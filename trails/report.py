"""Per-run Markdown reports and cross-run comparison tables (CSV + Markdown).

The comparison table is the main research artifact: one row per run (model x config x platform x
power mode) with offline accuracy, real-time accuracy, the gap between them, latency, decision rate
and energy, so Orin and Thor runs can be put side by side.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .tasks import get_task


def get_dotted(d: dict | None, key: str):
    for part in key.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def _headline(task_cfg: dict) -> tuple[str, ...]:
    try:
        return get_task(task_cfg).headline
    except Exception:
        return ()


def accuracy_gap(metrics: dict, keys: tuple[str, ...]) -> dict:
    off = (metrics.get("offline") or {}).get("accuracy")
    rt = (metrics.get("realtime") or {}).get("accuracy")
    if not off or not rt:
        return {}
    out = {}
    for k in keys:
        a, b = get_dotted(off, k), get_dotted(rt, k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = round(b - a, 4)
    return out


def write_run_report(run_dir: Path, metrics: dict, cfg: dict) -> Path:
    keys = _headline(cfg["task"])
    plat = json.loads((run_dir / "platform.json").read_text())
    model_id = cfg["model"].get("model_id") or cfg["model"].get("model")
    lines = [f"# {metrics['name']}" + (f" — {metrics['variant']}" if metrics.get("variant") else ""), "",
             f"- **Model:** {cfg['model'].get('name')}" + (f" (`{model_id}`)" if model_id else "")
             + f", backend {cfg['model'].get('backend')}",
             f"- **Task:** {metrics['task']} ({metrics['task_type']})",
             f"- **Platform:** {metrics['platform']} — {plat.get('device_model')} — "
             f"{(plat.get('nvpmodel') or 'power mode unknown').splitlines()[0]}",
             f"- **Videos:** {', '.join(Path(v).name for v in cfg['videos'])}",
             f"- **Model load:** {metrics.get('model_load_s')} s", ""]

    if keys and any(metrics.get(r, {}).get("accuracy") for r in ("offline", "realtime")):
        lines += ["## Accuracy", "", "| metric | offline (ceiling) | real-time | Δ (rt − off) |", "|---|---|---|---|"]
        gap = accuracy_gap(metrics, keys)
        for k in keys:
            lines.append(f"| {k} | {_fmt(get_dotted((metrics.get('offline') or {}).get('accuracy'), k))} | "
                         f"{_fmt(get_dotted((metrics.get('realtime') or {}).get('accuracy'), k))} | {_fmt(gap.get(k))} |")
        lines.append("")

    lat_keys = ["total_ms_p50", "total_ms_p90", "ttft_ms_p50", "decode_ms_p50", "preprocess_ms_p50", "tok_per_s_p50",
                "in_tokens_mean", "out_tokens_mean", "peak_gpu_mem_mb", "max_decision_hz"]
    runners = [r for r in ("offline", "realtime", "latency") if r in metrics]
    if runners:
        lines += ["## Latency", "", "| | " + " | ".join(runners) + " |", "|---" * (len(runners) + 1) + "|"]
        for k in lat_keys:
            lines.append(f"| {k} | " + " | ".join(_fmt(metrics[r]["latency"].get(k)) for r in runners) + " |")
        lines.append("")
    if "latency" in metrics:
        for stem, pv in metrics["latency"]["per_video"].items():
            sides = pv.get("latency_by_side") or {}
            if len(sides) > 1:
                lines += [f"### Resolution sweep ({stem})", "", "| max side | p50 ms | ttft p50 | in tokens |",
                          "|---|---|---|---|"]
                for side, s in sides.items():
                    lines.append(f"| {side} | {_fmt(s.get('total_ms_p50'))} | {_fmt(s.get('ttft_ms_p50'))} | "
                                 f"{_fmt(s.get('in_tokens_mean'))} |")
                lines.append("")
    if "realtime" in metrics:
        rt = metrics["realtime"]["realtime"]
        lines += ["## Real-time replay", "", f"Clock: {rt.get('clock')} (latency scale {rt.get('latency_scale')})", "",
                  "| | value |", "|---|---|"]
        for k in ("effective_decision_hz", "frame_drop_rate", "capture_to_decision_ms_p50", "capture_to_decision_ms_p90",
                  "staleness_at_gt_ms_p50", "staleness_at_gt_ms_p90", "staleness_at_gt_ms_max", "n_decisions"):
            lines.append(f"| {k} | {_fmt(rt.get(k))} |")
        lines.append("")
    lines += ["## Power / telemetry", "", "| | " + " | ".join(runners) + " |", "|---" * (len(runners) + 1) + "|"]
    for k in ("power_total_mw_mean", "power_total_mw_max", "energy_per_call_j", "gpu_util_pct_mean", "ram_used_mb_max",
              "temp_tj_c_max", "telemetry_source"):
        lines.append(f"| {k} | " + " | ".join(_fmt(metrics[r]["telemetry"].get(k)) for r in runners) + " |")
    lines.append("")
    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


COMPARE_FIELDS = ["run", "platform", "power_mode", "model", "variant", "task"]


def compare_runs(run_dirs: list[Path], out_prefix: Path) -> tuple[Path, Path]:
    rows = []
    for d in run_dirs:
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        cfg_task = {}
        try:
            import yaml
            cfg_task = yaml.safe_load((d / "config.yaml").read_text())["task"]
        except Exception:
            pass
        plat = json.loads((d / "platform.json").read_text()) if (d / "platform.json").exists() else {}
        power = (plat.get("nvpmodel") or "").splitlines()
        row = {"run": d.name, "platform": m.get("platform"), "power_mode": power[0].replace("NV Power Mode: ", "") if power else None,
               "model": m.get("model"), "variant": m.get("variant"), "task": m.get("task")}
        for k in _headline(cfg_task):
            row[f"off.{k}"] = get_dotted((m.get("offline") or {}).get("accuracy"), k)
            row[f"rt.{k}"] = get_dotted((m.get("realtime") or {}).get("accuracy"), k)
            row[f"gap.{k}"] = accuracy_gap(m, (k,)).get(k)
        src = m.get("offline") or m.get("realtime") or m.get("latency") or {}  # whole-video stats first
        lat = src.get("latency", {})
        for k in ("total_ms_p50", "total_ms_p90", "ttft_ms_p50", "tok_per_s_p50", "peak_gpu_mem_mb", "max_decision_hz"):
            row[k] = lat.get(k)
        rt = (m.get("realtime") or {}).get("realtime", {})
        for k in ("effective_decision_hz", "frame_drop_rate", "staleness_at_gt_ms_p50", "staleness_at_gt_ms_p90"):
            row[f"rt.{k}"] = rt.get(k)
        tel = src.get("telemetry", {})
        row["power_mw_mean"] = tel.get("power_total_mw_mean")
        row["energy_per_call_j"] = tel.get("energy_per_call_j")
        rows.append(row)
    fields = list(COMPARE_FIELDS)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_p = out_prefix.with_suffix(".csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # Markdown: drop columns that are empty for every run.
    cols = [c for c in fields if any(r.get(c) is not None for r in rows)]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    md_p = out_prefix.with_suffix(".md")
    md_p.write_text("\n".join(md) + "\n")
    return csv_p, md_p
