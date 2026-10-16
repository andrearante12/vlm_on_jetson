"""Video decoding and the on-disk frame cache.

Large high-resolution videos are decoded once and the frames we need are cached as JPEGs
(downscaled to `max_side`), so neither the offline nor the real-time runner pays decode cost
inside the measured loop. The cache is keyed by source frame index:

    data/frames/<video_stem>/<max_side>/<frame_idx:07d>.jpg
    data/frames/<video_stem>/<max_side>/index.jsonl   {"idx", "t_ms", "path"}
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def t_ms(self, idx: int) -> float:
        return idx * 1000.0 / self.fps

    def idx_at(self, t_ms: float) -> int:
        """Nearest frame to t_ms."""
        return max(0, min(self.n_frames - 1, int(round(t_ms * self.fps / 1000.0))))

    def idx_at_floor(self, t_ms: float) -> int:
        """Last frame whose timestamp is <= t_ms."""
        return max(0, min(self.n_frames - 1, int(math.floor(t_ms * self.fps / 1000.0 + 1e-6))))


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    info = VideoInfo(
        path=str(path),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def resize_max_side(img, max_side: int | None):
    """Downscale a BGR ndarray so its longest side is <= max_side (never upscales)."""
    if not max_side:
        return img
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def indices_for_fps(info: VideoInfo, fps: float, duration_s: float | None = None) -> list[int]:
    """Source frame indices that sample the video at `fps` (capped at the native rate)."""
    fps = min(fps, info.fps)
    end_s = info.duration_s if duration_s is None else min(duration_s, info.duration_s)
    n = int(math.floor(end_s * fps + 1e-9))
    idxs = sorted({info.idx_at(k * 1000.0 / fps) for k in range(n)})
    return idxs


class FrameCache:
    def __init__(self, video: str | Path, root: str | Path = "data/frames", max_side: int | None = 1280):
        self.video = str(video)
        self.info = probe(video)
        self.max_side = max_side
        self.dir = Path(root) / Path(video).stem / str(max_side or "full")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"
        self.index: dict[int, dict] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                rec = json.loads(line)
                if (self.dir / rec["path"]).exists():
                    self.index[rec["idx"]] = rec

    def path(self, idx: int) -> Path:
        return self.dir / f"{idx:07d}.jpg"

    def ensure(self, idxs: list[int], progress: bool = True) -> None:
        """Decode the video sequentially and write any missing frames among `idxs`."""
        missing = sorted(set(idxs) - set(self.index))
        if not missing:
            return
        want = set(missing)
        last = missing[-1]
        cap = cv2.VideoCapture(self.video)
        idx = 0
        written = 0
        # Seek close to the first missing frame; sequential grab() after that is exact.
        if missing[0] > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, missing[0])
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if idx != missing[0]:  # backend can't seek precisely; restart from 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
        with open(self.index_path, "a") as idx_f:
            while idx <= last:
                if not cap.grab():
                    break
                if idx in want:
                    ok, frame = cap.retrieve()
                    if ok:
                        frame = resize_max_side(frame, self.max_side)
                        cv2.imwrite(str(self.path(idx)), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        rec = {"idx": idx, "t_ms": round(self.info.t_ms(idx), 3), "path": self.path(idx).name}
                        self.index[idx] = rec
                        idx_f.write(json.dumps(rec) + "\n")
                        written += 1
                        if progress and written % 200 == 0:
                            print(f"  [{Path(self.video).name}] cached {written}/{len(missing)} frames")
                idx += 1
        cap.release()

    def ensure_fps(self, fps: float, duration_s: float | None = None) -> list[int]:
        idxs = indices_for_fps(self.info, fps, duration_s)
        self.ensure(idxs)
        return [i for i in idxs if i in self.index]

    def load(self, idx: int) -> Image.Image:
        if idx not in self.index:
            self.ensure([idx], progress=False)
        return Image.open(self.path(idx)).convert("RGB")

    def latest_before(self, t_ms: float, candidates: list[int]) -> int | None:
        """Most recent frame among sorted `candidates` captured at or before t_ms."""
        pos = bisect.bisect_right(candidates, self.info.idx_at_floor(t_ms))
        return candidates[pos - 1] if pos else None
