"""Self-contained HTML viewer for a run: `trails view <run_dir>` writes <run_dir>/view.html.

Everything (data, charts, frame thumbnails) is embedded in one file, so it opens offline, can be copied
to a laptop, or served to other devices with `trails serve`. No external scripts or fonts.

Sections: summary tiles, real-time decision timeline, frame-by-frame predictions (with filters and a
labelling mode that exports ground truth as CSV for `trails import-gt`), latency distributions,
resolution/tile sweep, and power/GPU telemetry.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import yaml
from PIL import Image

ASSET = Path(__file__).with_name("viewer_template.html")


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _thumb(path: Path, max_side: int, quality: int = 72) -> str | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _slim(rec: dict) -> dict:
    t = rec.get("timing") or {}
    out = {k: rec.get(k) for k in ("mode", "video", "idx", "t_ms", "capture_ms", "request_ms", "emit_ms",
                                  "parse_ok", "pred", "raw", "image_side")}
    out["timing"] = {k: t.get(k) for k in ("total_ms", "preprocess_ms", "ttft_ms", "decode_ms", "n_in_tokens",
                                          "n_out_tokens", "tok_per_s", "peak_mem_mb")}
    return {k: v for k, v in out.items() if v is not None}


def build_view(run_dir: str | Path, thumb_side: int = 640, rt_thumb_side: int = 400) -> Path:
    run_dir = Path(run_dir)
    from .config import ROOT

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    platform = json.loads((run_dir / "platform.json").read_text())
    model_info = json.loads((run_dir / "model.json").read_text()) if (run_dir / "model.json").exists() else {}

    runners = {r: [_slim(x) for x in _read_jsonl(run_dir / f"preds_{r}.jsonl")]
               for r in ("offline", "realtime", "latency")}

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    frames_root = _abs(cfg["frames_dir"])
    max_side = cfg.get("image", {}).get("max_side")
    gt_dir = _abs(cfg["gt_dir"])

    videos, gt, frames = {}, {}, {}
    from .groundtruth import load_gt
    from .video import probe

    for v in cfg["videos"]:
        stem = Path(v).stem
        try:
            i = probe(v)
            videos[stem] = {"fps": i.fps, "n_frames": i.n_frames, "width": i.width, "height": i.height,
                            "duration_s": i.duration_s, "file": Path(v).name}
        except OSError:
            videos[stem] = {"file": Path(v).name}
        g = load_gt(gt_dir, v)
        gt[stem] = {str(k): r for k, r in g.items()}
        fdir = frames_root / stem / str(max_side or "full")
        for rec in runners["offline"]:
            if rec["video"] == stem:
                key = f"{stem}:{rec['idx']}"
                frames.setdefault(key, _thumb(fdir / f"{rec['idx']:07d}.jpg", thumb_side))
        for rec in runners["realtime"]:
            if rec["video"] == stem:
                key = f"{stem}:{rec['idx']}"
                if key not in frames:
                    frames[key] = _thumb(fdir / f"{rec['idx']:07d}.jpg", rt_thumb_side)

    telemetry = []
    tp = run_dir / "telemetry.csv"
    if tp.exists():
        with open(tp) as f:
            for row in csv.DictReader(f):
                def num(k):
                    try:
                        return float(row[k]) if row.get(k) not in (None, "") else None
                    except ValueError:
                        return None
                telemetry.append({"t": num("t"), "power_w": None if num("power_total_mw") is None else num("power_total_mw") / 1000,
                                  "gpu": num("gpu_util_pct"), "ram_gb": None if num("ram_used_mb") is None else num("ram_used_mb") / 1024,
                                  "temp": num("temp_tj_c")})

    from .tasks import get_task
    headline = list(get_task(cfg["task"]).headline)

    data = {
        "run": run_dir.name,
        "meta": {
            "name": metrics.get("name"), "variant": metrics.get("variant"), "task": metrics.get("task"),
            "task_type": metrics.get("task_type"), "model": metrics.get("model"),
            "model_id": cfg["model"].get("model_id") or cfg["model"].get("model"),
            "backend": cfg["model"].get("backend"), "platform": metrics.get("platform"),
            "device": platform.get("device_model"),
            "power_mode": (platform.get("nvpmodel") or "").splitlines()[0] if platform.get("nvpmodel") else None,
            "timestamp": platform.get("timestamp"), "torch": platform.get("torch"),
            "transformers": platform.get("transformers"), "model_load_s": metrics.get("model_load_s"),
            "n_params_m": model_info.get("n_params_m"), "prompt": cfg["task"].get("prompt"),
            "gen": cfg.get("gen"), "realtime_cfg": cfg.get("realtime"),
        },
        "headline": headline,
        "metrics": {r: {k: metrics[r].get(k) for k in ("latency", "accuracy", "telemetry", "realtime", "t_start", "t_end")}
                    | {"per_video": {s: {k: pv.get(k) for k in ("latency_by_side",)} for s, pv in metrics[r].get("per_video", {}).items()}}
                    for r in ("offline", "realtime", "latency") if r in metrics},
        "runners": runners, "gt": gt, "videos": videos, "frames": frames, "telemetry": telemetry,
    }
    page = ASSET.read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = f"{metrics.get('model')} · {metrics.get('task')}" + (f" · {metrics['variant']}" if metrics.get("variant") else "")
    page = page.replace("__TITLE__", html.escape(title)).replace("/*__DATA__*/null", blob)
    out = run_dir / "view.html"
    out.write_text(page)
    return out


def build_index(results_dir: str | Path) -> Path:
    """results/index.html: one card per run that has a view.html (newest first)."""
    results_dir = Path(results_dir)
    rows = []
    views = (p for p in results_dir.rglob("view.html")
             if not any(part.startswith("_") for part in p.relative_to(results_dir).parts))  # skip _scratch dirs
    for d in sorted(views, key=lambda p: p.parent.name, reverse=True):
        run = d.parent
        try:
            m = json.loads((run / "metrics.json").read_text())
            plat = json.loads((run / "platform.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lat = (m.get("offline") or m.get("realtime") or m.get("latency") or {}).get("latency", {})
        rt = (m.get("realtime") or {}).get("realtime", {})
        acc = (m.get("offline") or {}).get("accuracy") or {}
        rows.append({
            "href": str(d.relative_to(results_dir)), "run": run.name, "model": m.get("model"), "task": m.get("task"),
            "variant": m.get("variant"), "platform": m.get("platform"),
            "power": (plat.get("nvpmodel") or "").splitlines()[0].replace("NV Power Mode: ", "") if plat.get("nvpmodel") else "",
            "p50": lat.get("total_ms_p50"), "hz": rt.get("effective_decision_hz"),
            "acc": (acc.get("safety") or {}).get("accuracy"), "runners": [r for r in ("offline", "realtime", "latency") if r in m],
        })
    page = INDEX_TEMPLATE.replace("/*__ROWS__*/[]", json.dumps(rows).replace("</", "<\\/"))
    out = results_dir / "index.html"
    out.write_text(page)
    return out


INDEX_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Trails runs</title>
<style>
:root{--bg:#f7f7f5;--surface:#fcfcfb;--line:#e4e3de;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--line:#2e2e2b;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--line:#2e2e2b;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--accent:#3987e5}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}p{color:var(--ink2);margin:0 0 20px}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--ink3);font-weight:600;font-size:12px}
td.num{text-align:right;font-variant-numeric:tabular-nums}a{color:var(--accent);text-decoration:none;font-weight:600}a:hover{text-decoration:underline}
.wrap{overflow-x:auto}.muted{color:var(--ink3)}
</style></head><body><main><h1>Trails benchmark runs</h1><p>Open a run to see its predictions, real-time timeline, latency and power.</p>
<div class="wrap"><table><thead><tr><th>Run</th><th>Model</th><th>Task</th><th>Platform</th><th>Power</th><th>Runners</th>
<th class="num">p50 latency</th><th class="num">Real-time Hz</th><th class="num">Safety acc.</th></tr></thead><tbody id="b"></tbody></table></div>
</main><script>
const R=/*__ROWS__*/[];const f=(v,d=0)=>v==null?'<span class="muted">–</span>':Number(v).toFixed(d);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('b').innerHTML=R.length?R.map(r=>`<tr><td><a href="${esc(r.href)}">${esc(r.run)}</a></td><td>${esc(r.model)}</td>
<td>${esc(r.task)}${r.variant?` <span class="muted">${esc(r.variant)}</span>`:''}</td><td>${esc(r.platform)}</td><td>${esc(r.power)}</td>
<td>${esc(r.runners.join(', '))}</td><td class="num">${r.p50==null?f(null):f(r.p50)+' ms'}</td><td class="num">${f(r.hz,2)}</td><td class="num">${f(r.acc,3)}</td></tr>`).join('')
:'<tr><td colspan="9" class="muted">No runs with a view yet. Run <code>trails view &lt;run_dir&gt;</code>.</td></tr>';
</script></body></html>"""
