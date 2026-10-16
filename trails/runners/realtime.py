"""Real-time replay runner: the video plays as if it were a live camera.

A simulated camera publishes frames at `camera_fps` on a clock. Whenever the model is idle it takes the
*latest* published frame (frames published while it was busy are dropped), runs inference, and emits a
decision at time `emit`. A decision stays in effect until the next one is emitted (hold_last).

Scoring is time-aligned: at each GT timestamp t, the decision in effect is the most recent one with
emit <= t, which may have been computed from a frame captured well before t. Comparing this to the
offline score of the same model isolates how much accuracy is lost purely to latency.

Clocks:
  wall       real elapsed time x `speed`; the honest on-device measurement.
  simulated  time advances by measured latency x `latency_scale` (+ extra_latency_ms). Lets you ask
             "what if inference were 3x faster" (e.g. projected Thor numbers) without the hardware.
"""

from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

from ..backends import GenConfig, VLMBackend
from ..metrics import realtime_summary
from ..tasks import Task
from ..video import FrameCache, indices_for_fps


def run_realtime(backend: VLMBackend, task: Task, cache: FrameCache, gt: dict[int, dict], cfg: dict,
                 out_f) -> tuple[list[tuple[dict | None, dict]], list[dict], dict]:
    rcfg = cfg["realtime"]
    gen = GenConfig.from_cfg(cfg["gen"])
    prompt = task.prompt(cfg["model"])
    info = cache.info
    video = Path(cache.video).stem
    cam_idxs = indices_for_fps(info, float(rcfg["camera_fps"]), rcfg.get("duration_s"))
    print(f"  [realtime {video}] caching {len(cam_idxs)} camera frames @ {rcfg['camera_fps']} fps")
    cache.ensure(cam_idxs)
    cam_idxs = [i for i in cam_idxs if i in cache.index]
    end_ms = info.t_ms(cam_idxs[-1]) + 1000.0 / float(rcfg["camera_fps"])

    simulated = rcfg.get("clock", "wall") == "simulated"
    speed = float(rcfg.get("speed", 1.0))
    scale = float(rcfg.get("latency_scale", 1.0))
    extra = float(rcfg.get("extra_latency_ms", 0.0))

    decisions: list[dict] = []
    timings: list[dict] = []
    sim_now = 0.0
    wall0 = time.perf_counter()

    def now_ms() -> float:
        return sim_now if simulated else (time.perf_counter() - wall0) * 1000 * speed

    while True:
        t_req = now_ms()
        if t_req >= end_ms:
            break
        idx = cache.latest_before(t_req, cam_idxs)
        if idx is None:  # before the first camera frame
            idx = cam_idxs[0]
        f0 = time.perf_counter()
        img = cache.load(idx)
        fetch_ms = (time.perf_counter() - f0) * 1000
        res = backend.generate([img], prompt, gen)
        if simulated:
            sim_now = t_req + (fetch_ms + res.total_ms) * scale + extra
        elif extra:
            time.sleep(extra / 1000 / speed)
        emit = now_ms()
        ctx = {"input_size": res.meta.get("input_size") or img.size, "bbox_format": cfg["model"].get("bbox_format", "auto")}
        pred, ok = task.parse(res.text, ctx)
        d = {"mode": "realtime", "video": video, "idx": idx, "capture_ms": round(info.t_ms(idx), 3),
             "request_ms": round(t_req, 3), "emit_ms": round(emit, 3), "fetch_ms": round(fetch_ms, 2),
             "parse_ok": ok, "pred": pred, "raw": res.text, "timing": res.to_dict() | {"text": None}}
        out_f.write(json.dumps(d) + "\n")
        out_f.flush()
        decisions.append(d)
        timings.append(res.to_dict())
    replay_s = (now_ms() if simulated else (time.perf_counter() - wall0) * 1000 * speed) / 1000

    pairs, staleness = time_aligned_pairs(decisions, gt, info.t_ms, task, end_ms, rcfg.get("decision_ttl_ms"))
    summary = realtime_summary(decisions, staleness, len(cam_idxs), min(replay_s, end_ms / 1000))
    summary["n_camera_frames"] = len(cam_idxs)
    print(f"  [realtime {video}] {summary['n_decisions']} decisions, "
          f"{summary.get('effective_decision_hz')} Hz, drop rate {summary.get('frame_drop_rate')}")
    # Raw lists let the caller aggregate percentiles across videos; it pops them before saving.
    summary["_decisions"], summary["_staleness"] = decisions, staleness
    return pairs, timings, summary


def time_aligned_pairs(decisions: list[dict], gt: dict[int, dict], t_ms, task: Task, end_ms: float,
                       ttl_ms: float | None = None):
    """Pair each GT frame (within the replayed span) with the decision in effect at its timestamp.

    A decision is in effect from its emit time until the next decision, or for at most `ttl_ms` if set
    (a stale decision then counts as no decision). Returns (pairs, staleness_ms per scored GT frame).
    """
    emits = [d["emit_ms"] for d in decisions]
    pairs, staleness = [], []
    for idx in sorted(gt):
        g = gt[idx]
        t = t_ms(idx)
        if not task.scorable(g) or t > end_ms:
            continue
        k = bisect.bisect_right(emits, t) - 1
        if k < 0 or (ttl_ms is not None and t - decisions[k]["emit_ms"] > ttl_ms):
            pairs.append((None, g))  # nothing (valid) emitted yet
            continue
        pairs.append((decisions[k]["pred"], g))
        staleness.append(t - decisions[k]["capture_ms"])
    return pairs, staleness
