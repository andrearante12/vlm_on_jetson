"""Latency microbenchmark: repeated inference on a few fixed frames, optionally at several input
resolutions. Isolates model speed from video content and is the cleanest Orin-vs-Thor comparison.

Config (experiment `latency:` block): n_warmup, n_reps, n_frames, image_sides (list of max-side px;
null = cached frame as-is). Model-side knobs (tiles, max_pixels, quantization) are swept with `sweep:`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..backends import GenConfig, VLMBackend
from ..tasks import Task
from ..video import FrameCache, indices_for_fps


def _resize(img, side):
    if not side:
        return img
    s = side / max(img.size)
    return img if s >= 1 else img.resize((round(img.width * s), round(img.height * s)))


def run_latency(backend: VLMBackend, task: Task, cache: FrameCache, cfg: dict, out_f) -> tuple[list[dict], dict]:
    lcfg = cfg["latency"]
    gen = GenConfig.from_cfg(cfg["gen"])
    prompt = task.prompt(cfg["model"])
    n_frames = int(lcfg.get("n_frames", 3))
    all_idx = indices_for_fps(cache.info, 1.0) or [0]
    step = max(1, len(all_idx) // n_frames)
    idxs = all_idx[::step][:n_frames]
    cache.ensure(idxs)
    frames = [cache.load(i) for i in idxs]
    video = Path(cache.video).stem
    timings, by_side = [], {}
    from ..metrics import latency_summary

    for side in lcfg.get("image_sides") or [None]:
        imgs = [_resize(f, side) for f in frames]
        for k in range(int(lcfg.get("n_warmup", 3))):
            backend.generate([imgs[k % len(imgs)]], prompt, gen)
        rows = []
        for k in range(int(lcfg.get("n_reps", 10))):
            img = imgs[k % len(imgs)]
            res = backend.generate([img], prompt, gen)
            row = res.to_dict() | {"text": None}
            out_f.write(json.dumps({"mode": "latency", "video": video, "idx": idxs[k % len(idxs)],
                                    "image_side": side, "image_size": img.size, "timing": row,
                                    "raw": res.text}) + "\n")
            out_f.flush()
            rows.append(res.to_dict())
        by_side[str(side or "native")] = latency_summary(rows)
        s = by_side[str(side or "native")]
        print(f"  [latency {video}] side={side or 'native'}: p50 {s.get('total_ms_p50')} ms, "
              f"ttft p50 {s.get('ttft_ms_p50')} ms, {s.get('tok_per_s_p50')} tok/s")
        timings += rows
    return timings, by_side
