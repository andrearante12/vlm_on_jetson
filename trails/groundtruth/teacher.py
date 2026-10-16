"""Pseudo-label video frames with a large "teacher" VLM.

The teacher labels every field at once (verdict, surface, obstacles with boxes, caption) so a single
labelling pass serves all three tasks. Teacher labels are tagged provenance="teacher:<model>" and are
outranked by human annotations on the same frame (see store.load_gt).

Caveats worth keeping in mind when reporting results:
  * scores against teacher labels measure *agreement with the teacher*, not ground truth; validate the
    teacher on a hand-checked subset before relying on it;
  * cloud teachers see the frames: sending footage to an external API must be acceptable for your data.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

from ..backends import GenConfig, get_backend
from ..schema import DEFAULT_OBSTACLE_CLASSES, DEFAULT_SURFACES, Vocab, grid_cell_from_bbox, normalize_bbox
from ..tasks.base import extract_json, parse_bool
from ..video import FrameCache
from .store import load_gt, merge_records

OBSTACLE_TYPES = list(DEFAULT_OBSTACLE_CLASSES) + ["other"]
SURFACE_TYPES = list(DEFAULT_SURFACES) + ["other"]

TEACHER_PROMPT = f"""You are labelling ground truth for evaluating an autonomous drone (small UAS) landing system.
The image is one frame of the drone's camera video.

Judge whether the drone could safely land in the area at the CENTER of the image right now. Treat it as
unsafe if the center area contains or is within roughly a rotor-span of people, animals, vehicles, trees,
poles, wires, water, steep slope, large rocks/debris, or other hazards, or if the surface is not firm
and roughly level.

Label:
- safe_to_land: true or false
- surface: the dominant surface at the image center, one of {SURFACE_TYPES}
- obstacles: every visible hazard that would matter for landing anywhere in the frame, each with
  type (one of {OBSTACLE_TYPES}) and a tight bounding box with integer coordinates normalised to
  0-1000 (x1, y1 = top-left; x2, y2 = bottom-right). Use an empty list if there are none.
- caption: 1-3 sentences describing the scene from a landing-safety perspective.

Return only the JSON object."""

TEACHER_SCHEMA = {
    "type": "object",
    "properties": {
        "safe_to_land": {"type": "boolean"},
        "surface": {"type": "string", "enum": SURFACE_TYPES},
        "obstacles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": OBSTACLE_TYPES},
                    "x1": {"type": "integer"}, "y1": {"type": "integer"},
                    "x2": {"type": "integer"}, "y2": {"type": "integer"},
                },
                "required": ["type", "x1", "y1", "x2", "y2"],
                "additionalProperties": False,
            },
        },
        "caption": {"type": "string"},
    },
    "required": ["safe_to_land", "surface", "obstacles", "caption"],
    "additionalProperties": False,
}

_obst = Vocab(DEFAULT_OBSTACLE_CLASSES)
_surf = Vocab(DEFAULT_SURFACES)


def parse_teacher(text: str) -> dict | None:
    obj = extract_json(text)
    if not isinstance(obj, dict) or parse_bool(obj.get("safe_to_land")) is None:
        return None
    obstacles = []
    for o in obj.get("obstacles") or []:
        if not isinstance(o, dict):
            continue
        coords = o.get("bbox") or [o.get("x1"), o.get("y1"), o.get("x2"), o.get("y2")]
        bbox = normalize_bbox(coords, fmt="norm1000") if None not in coords else None
        obstacles.append({"type": _obst.canon(o.get("type")) or "other", "bbox": bbox,
                          "grid_cell": grid_cell_from_bbox(bbox)})
    return {"safe_to_land": parse_bool(obj["safe_to_land"]), "surface": _surf.canon(obj.get("surface")),
            "obstacles": obstacles, "caption": obj.get("caption")}


def label_video(video: str, teacher_cfg: dict, gt_dir: str = "data/gt", frames_dir: str = "data/frames",
                fps: float = 1.0, max_side: int = 1280, concurrency: int = 4, max_frames: int | None = None,
                overwrite: bool = False) -> Path:
    cfg = dict(teacher_cfg)
    if cfg.get("backend") == "anthropic":
        cfg.setdefault("json_schema", TEACHER_SCHEMA)
    backend = get_backend(cfg)
    prov = f"teacher:{cfg.get('model') or cfg.get('model_id') or cfg.get('name')}"
    cache = FrameCache(video, frames_dir, max_side)
    idxs = cache.ensure_fps(fps)
    if max_frames:
        idxs = idxs[:max_frames]
    if not overwrite:
        done = set(load_gt(gt_dir, video, provenance_prefix=prov))
        idxs = [i for i in idxs if i not in done]
    print(f"[teacher] {Path(video).name}: labelling {len(idxs)} frames with {prov}")
    gen = GenConfig(max_new_tokens=int(cfg.get("max_new_tokens", 2048)), temperature=0.0)
    prompt = cfg.get("prompt", TEACHER_PROMPT)

    def work(idx: int) -> tuple[int, dict | None, str]:
        res = backend.generate([cache.load(idx)], prompt, gen)
        return idx, parse_teacher(res.text), res.meta.get("stop_reason", "")

    records, failed = [], []
    # Write in batches so an interrupted run keeps what it has (resumes by skipping labelled frames).
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, i) for i in idxs]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            try:
                idx, lab, stop = fut.result()
            except Exception as e:  # keep going; failed frames are reported and can be retried
                failed.append(str(e)[:200])
                continue
            if lab is None:
                failed.append(f"unparseable (stop_reason={stop})")
                continue
            records.append({"video": Path(video).stem, "idx": idx, "t_ms": round(cache.info.t_ms(idx), 3),
                            **lab, "provenance": prov})
            if len(records) % 25 == 0:
                merge_records(gt_dir, video, records)
                print(f"  {n}/{len(idxs)} labelled")
    path = merge_records(gt_dir, video, records)
    if failed:
        print(f"[teacher] {len(failed)} frames failed, e.g. {failed[0]!r}; rerun to retry them")
    backend.close()
    print(f"[teacher] wrote {path}")
    return path
