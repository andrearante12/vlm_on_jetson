"""Offline runner: run the model on every evaluation frame with no deadline.

This gives the accuracy ceiling of a model/config: what it would score if it were infinitely fast.
Evaluation frames are the GT-labelled frames (thinned to `gt_min_spacing_ms`), or frames sampled at
`sample_fps` when the video has no GT (latency/prediction dump only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..backends import GenConfig, VLMBackend
from ..tasks import Task
from ..video import FrameCache, indices_for_fps


def thin(idxs: list[int], t_ms, min_spacing_ms: float) -> list[int]:
    out, last = [], None
    for i in sorted(idxs):
        if last is None or t_ms(i) - last >= min_spacing_ms:
            out.append(i)
            last = t_ms(i)
    return out


def eval_frames(cache: FrameCache, gt: dict[int, dict], task: Task, ocfg: dict) -> list[int]:
    if gt and ocfg.get("use_gt_timestamps", True):
        idxs = [i for i, r in gt.items() if task.scorable(r)]
        idxs = thin(idxs, cache.info.t_ms, float(ocfg.get("gt_min_spacing_ms") or 0))
    else:
        idxs = indices_for_fps(cache.info, float(ocfg.get("sample_fps", 1.0)))
    if ocfg.get("max_frames"):
        idxs = idxs[: int(ocfg["max_frames"])]
    return idxs


def run_offline(backend: VLMBackend, task: Task, cache: FrameCache, gt: dict[int, dict], cfg: dict,
                out_f) -> tuple[list[tuple[dict | None, dict]], list[dict]]:
    ocfg = cfg["offline"]
    gen = GenConfig.from_cfg(cfg["gen"])
    prompt = task.prompt(cfg["model"])
    idxs = eval_frames(cache, gt, task, ocfg)
    cache.ensure(idxs)
    pairs, timings = [], []
    video = Path(cache.video).stem
    t_start = time.time()
    for n, idx in enumerate(idxs, 1):
        img = cache.load(idx)
        res = backend.generate([img], prompt, gen)
        ctx = {"input_size": res.meta.get("input_size") or img.size, "bbox_format": cfg["model"].get("bbox_format", "auto")}
        pred, ok = task.parse(res.text, ctx)
        rec = {"mode": "offline", "video": video, "idx": idx, "t_ms": round(cache.info.t_ms(idx), 3),
               "parse_ok": ok, "pred": pred, "raw": res.text, "timing": res.to_dict() | {"text": None}}
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        timings.append(res.to_dict())
        if idx in gt:
            pairs.append((pred, gt[idx]))
        if n % 10 == 0 or n == len(idxs):
            el = time.time() - t_start
            print(f"  [offline {video}] {n}/{len(idxs)} frames, {el / n * 1000:.0f} ms/frame avg")
    return pairs, timings
