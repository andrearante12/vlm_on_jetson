"""Run one resolved experiment config end to end and write a self-contained results directory:

    results/<stamp>_<name>[_<variant>]/
        config.yaml      fully resolved config (model/task/platform inlined)
        platform.json    hardware/software snapshot (L4T, nvpmodel, versions)
        model.json       backend.describe() (model id, dtype, load time, #params)
        preds_<runner>.jsonl  every model call: raw text, parsed prediction, timings
        telemetry.csv    tegrastats samples for the whole run
        metrics.json     accuracy + latency + real-time + telemetry summaries
        report.md        human-readable summary
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import yaml

from .. import platform_info
from ..backends import GenConfig, get_backend
from ..config import ROOT
from ..groundtruth import load_gt
from ..metrics import latency_summary, realtime_summary
from ..report import write_run_report
from ..tasks import get_task
from ..telemetry import Telemetry
from ..video import FrameCache
from .latency import run_latency
from .offline import run_offline
from .realtime import run_realtime


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_experiment(cfg: dict) -> Path:
    videos = cfg["videos"]
    if not videos:
        raise SystemExit("no videos matched the experiment's `videos:` patterns (put footage in data/videos/)")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = cfg["name"] + (f"_{cfg['variant']}" if cfg.get("variant") else "")
    run_dir = _abs(cfg["results_dir"]) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    plat = platform_info.snapshot()
    (run_dir / "platform.json").write_text(json.dumps(plat, indent=2))
    print(f"[run] {name} -> {run_dir}")
    print(f"[run] platform {cfg['platform'].get('name')} | {plat.get('device_model')} | {plat.get('nvpmodel')}")

    task = get_task(cfg["task"])
    gt_dir, frames_dir = _abs(cfg["gt_dir"]), _abs(cfg["frames_dir"])
    max_side = cfg["image"].get("max_side")
    metrics: dict = {"name": cfg["name"], "variant": cfg.get("variant"), "platform": cfg["platform"].get("name"),
                     "task": task.name, "task_type": task.type, "model": cfg["model"].get("name"),
                     "run_dir": str(run_dir)}

    tcfg = cfg.get("telemetry", {})
    with Telemetry(run_dir / "telemetry.csv", tcfg.get("interval_ms", 500), tcfg.get("enabled", True)) as tel:
        t_load = time.time()
        backend = get_backend(cfg["model"])
        metrics["model_load_s"] = round(time.time() - t_load, 1)
        metrics["telemetry_load"] = tel.summary(t_load, time.time())
        (run_dir / "model.json").write_text(json.dumps(backend.describe(), indent=2, default=str))
        warmed = False
        try:
            for runner in cfg["runners"]:
                t0 = time.time()
                all_pairs, all_timings, per_video = [], [], {}
                rt_decisions, rt_stale, rt_cam, rt_replay = [], [], 0, 0.0
                with open(run_dir / f"preds_{runner}.jsonl", "w") as f:
                    for video in videos:
                        stem = Path(video).stem
                        cache = FrameCache(video, frames_dir, max_side)
                        gt = load_gt(gt_dir, video)
                        print(f"[{runner}] {stem}: {cache.info.width}x{cache.info.height} @ {cache.info.fps:.2f} fps, "
                              f"{cache.info.duration_s:.1f}s, {len(gt)} GT frames")
                        if not warmed and int(cfg.get("warmup", 2)) > 0:
                            first = cache.ensure_fps(1.0, duration_s=1.0) or [0]
                            backend.warmup(cache.load(first[0]), task.prompt(cfg["model"]),
                                           GenConfig.from_cfg(cfg["gen"]), int(cfg.get("warmup", 2)))
                            warmed = True
                        pv: dict = {"n_gt": len(gt)}
                        if runner == "offline":
                            pairs, timings = run_offline(backend, task, cache, gt, cfg, f)
                        elif runner == "realtime":
                            pairs, timings, rt = run_realtime(backend, task, cache, gt, cfg, f)
                            rt_decisions += rt.pop("_decisions")
                            rt_stale += rt.pop("_staleness")
                            rt_cam += rt["n_camera_frames"]
                            rt_replay += rt["replay_s"]
                            pv["realtime"] = rt
                        elif runner == "latency":
                            timings, by_side = run_latency(backend, task, cache, cfg, f)
                            pairs = []
                            pv["latency_by_side"] = by_side
                        else:
                            raise ValueError(f"unknown runner {runner!r}")
                        if pairs:
                            pv["accuracy"] = task.score(pairs)
                        pv["latency"] = latency_summary(timings)
                        per_video[stem] = pv
                        all_pairs += pairs
                        all_timings += timings
                m: dict = {"t_start": t0, "t_end": time.time(), "latency": latency_summary(all_timings),
                           "accuracy": task.score(all_pairs) if all_pairs else None,
                           "telemetry": tel.summary(t0, time.time()), "per_video": per_video}
                if runner == "realtime":
                    m["realtime"] = realtime_summary(rt_decisions, rt_stale, rt_cam, rt_replay)
                    m["realtime"]["clock"] = cfg["realtime"].get("clock")
                    m["realtime"]["latency_scale"] = cfg["realtime"].get("latency_scale")
                # Board energy per model call = mean board power x mean call latency (the runner's
                # wall time also includes frame caching, so energy_j / n_calls would overstate it).
                if m["telemetry"].get("power_total_mw_mean") and m["latency"].get("total_ms_mean"):
                    m["telemetry"]["energy_per_call_j"] = round(
                        m["telemetry"]["power_total_mw_mean"] / 1000 * m["latency"]["total_ms_mean"] / 1000, 3)
                metrics[runner] = m
        finally:
            backend.close()

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_run_report(run_dir, metrics, cfg)
    try:  # the viewer is a convenience; never fail a finished run because of it
        from ..viewer import build_index, build_view
        build_view(run_dir)
        build_index(run_dir.parent)
    except Exception as e:  # noqa: BLE001
        print(f"[run] viewer not built: {e!r}")
    print(f"[run] done: {run_dir / 'report.md'}")
    return run_dir
