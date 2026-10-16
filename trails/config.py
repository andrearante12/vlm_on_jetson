"""Experiment config loading.

An experiment YAML references (or inlines) a model, a task and a platform config:

    name: smoke
    platform: auto            # or orin_nano / thor / path to yaml
    model: fake               # name in configs/models/ or path or inline dict
    task: landing_json        # name in configs/tasks/ or path or inline dict
    videos: [data/videos/*.mp4]
    sweep:                    # optional: one run per entry, each a dict of dotted overrides
      - {model.max_tiles: 1}
      - {model.max_tiles: 6}

String references are resolved to dicts so the saved config.yaml of a run is fully self-contained.
"""

from __future__ import annotations

import copy
import glob
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

DEFAULTS: dict[str, Any] = {
    "name": "experiment",
    "platform": "auto",
    "videos": [],
    "gt_dir": "data/gt",
    "frames_dir": "data/frames",
    "results_dir": "results",
    "runners": ["offline"],
    "offline": {"sample_fps": 1.0, "use_gt_timestamps": True, "gt_min_spacing_ms": 0, "max_frames": None},
    "realtime": {
        "clock": "wall",          # wall: real elapsed time | simulated: advance by measured latency
        "camera_fps": 15.0,       # rate at which the simulated camera publishes frames
        "speed": 1.0,             # video playback speed multiplier (wall clock)
        "latency_scale": 1.0,     # simulated clock only: scale measured latency (what-if faster HW)
        "extra_latency_ms": 0.0,  # injected per-decision latency (e.g. downstream planner)
        "duration_s": None,       # cap replay length
        "decision_ttl_ms": None,  # a decision expires after this long (None: holds until the next one)
    },
    "latency": {"n_warmup": 3, "n_reps": 10, "n_frames": 3, "image_sides": [None]},
    "warmup": 2,
    "image": {"max_side": 1280},  # downscale applied when caching frames
    "gen": {"max_new_tokens": 256, "temperature": 0.0},
    "telemetry": {"enabled": True, "interval_ms": 500},
}


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_ref(ref: Any, kind: str) -> dict:
    """Turn a config reference (name, path, or inline dict) into a dict."""
    if isinstance(ref, dict):
        return copy.deepcopy(ref)
    if not isinstance(ref, str):
        raise ValueError(f"{kind}: expected name, path or mapping, got {ref!r}")
    p = Path(ref)
    candidates = [p, ROOT / p, CONFIG_DIR / kind / f"{ref}.yaml"]
    for c in candidates:
        if c.suffix in (".yaml", ".yml") and c.is_file():
            d = _load_yaml(c)
            d.setdefault("name", c.stem)
            return d
    raise FileNotFoundError(f"{kind} config {ref!r} not found (looked in {CONFIG_DIR / kind})")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_override(s: str) -> tuple[str, Any]:
    """'model.max_tiles=6' -> ('model.max_tiles', 6). Values are parsed as YAML scalars."""
    if "=" not in s:
        raise ValueError(f"override must look like key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def detect_platform() -> str:
    try:
        model = Path("/proc/device-tree/model").read_text().lower()
    except OSError:
        return "generic"
    if "thor" in model:
        return "thor"
    if "orin nano" in model:
        return "orin_nano"
    if "orin" in model:
        return "orin_agx"
    return "generic"


def load_experiment(path_or_dict: str | dict, overrides: list[str] | None = None,
                    platform: str | None = None) -> list[dict]:
    """Load an experiment and expand sweeps. Returns one fully-resolved config per run."""
    raw = _load_yaml(Path(path_or_dict)) if isinstance(path_or_dict, str) else copy.deepcopy(path_or_dict)
    cfg = deep_merge(DEFAULTS, raw)
    if platform:
        cfg["platform"] = platform
    if cfg["platform"] == "auto":
        cfg["platform"] = detect_platform()

    plat_file = CONFIG_DIR / "platforms" / f"{cfg['platform']}.yaml"
    plat = _resolve_ref(cfg["platform"], "platforms") if isinstance(cfg["platform"], dict) or plat_file.exists() \
        else {"name": cfg["platform"]}
    cfg["platform"] = plat
    cfg["videos"] = expand_videos(cfg["videos"])
    sweep = cfg.pop("sweep", None) or [{}]
    cli_overrides = [parse_override(o) for o in overrides or []]

    runs = []
    for variant in sweep:
        run = copy.deepcopy(cfg)
        # A variant may replace a whole model/task by name ({model: qwen2_5_vl_3b}) or tweak one field
        # ({model.max_tiles: 6}); whole-ref replacements are applied before resolving, field tweaks after.
        for k, v in list(variant.items()) + cli_overrides:
            if k in ("model", "task"):
                run[k] = v
        run["model"] = deep_merge(plat.get("model_defaults", {}), _resolve_ref(run["model"], "models"))
        run["task"] = _resolve_ref(run["task"], "tasks")
        for k, v in list(variant.items()) + cli_overrides:
            if k not in ("model", "task"):
                set_dotted(run, k, v)
        run["variant"] = _variant_name(variant) if variant else None
        runs.append(run)
    return runs


def _variant_name(variant: dict) -> str:
    def val(v):
        if isinstance(v, dict):
            return v.get("name", "inline")
        if isinstance(v, list):
            return "+".join(map(str, v))
        return v
    return "_".join(f"{k.split('.')[-1]}-{val(v)}" for k, v in variant.items()).replace("/", "-").replace(" ", "")


def expand_videos(patterns: list[str] | str) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    out: list[str] = []
    for pat in patterns:
        pat = os.path.expanduser(pat)
        if not os.path.isabs(pat) and not glob.glob(pat):
            pat = str(ROOT / pat)
        matches = sorted(glob.glob(pat))
        out.extend(matches if matches else [])
    return out
