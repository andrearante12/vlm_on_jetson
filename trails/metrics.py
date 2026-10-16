"""Latency and real-time metrics. Accuracy metrics live with each task (tasks/*.py)."""

from __future__ import annotations

import math


def percentile(vals: list[float], q: float) -> float | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def dist(vals: list[float | None], prefix: str) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    r = lambda x: None if x is None else round(x, 2)  # noqa: E731
    return {f"{prefix}_mean": r(sum(vals) / len(vals)), f"{prefix}_p50": r(percentile(vals, 50)),
            f"{prefix}_p90": r(percentile(vals, 90)), f"{prefix}_p99": r(percentile(vals, 99)),
            f"{prefix}_max": r(max(vals))}


def latency_summary(results: list[dict]) -> dict:
    """results: Result.to_dict() rows."""
    out = {"n_calls": len(results)}
    for key, name in (("total_ms", "total_ms"), ("preprocess_ms", "preprocess_ms"), ("ttft_ms", "ttft_ms"),
                      ("decode_ms", "decode_ms"), ("tok_per_s", "tok_per_s"), ("n_in_tokens", "in_tokens"),
                      ("n_out_tokens", "out_tokens")):
        out.update(dist([r.get(key) for r in results], name))
    peaks = [r["peak_mem_mb"] for r in results if r.get("peak_mem_mb")]
    if peaks:
        out["peak_gpu_mem_mb"] = round(max(peaks), 1)
    if out.get("total_ms_mean"):
        out["max_decision_hz"] = round(1000 / out["total_ms_mean"], 3)
    return out


def realtime_summary(decisions: list[dict], staleness_at_gt: list[float], n_camera_frames: int,
                     replay_s: float) -> dict:
    out = {"n_decisions": len(decisions), "replay_s": round(replay_s, 2),
           "effective_decision_hz": round(len(decisions) / replay_s, 3) if replay_s else None,
           "frames_processed_frac": round(len(decisions) / n_camera_frames, 4) if n_camera_frames else None}
    out["frame_drop_rate"] = None if out["frames_processed_frac"] is None else round(1 - out["frames_processed_frac"], 4)
    out.update(dist([d["emit_ms"] - d["capture_ms"] for d in decisions], "capture_to_decision_ms"))
    out.update(dist(staleness_at_gt, "staleness_at_gt_ms"))
    return out
