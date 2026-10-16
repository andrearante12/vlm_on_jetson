"""Command line entry point: `trails <command>` (or `python -m trails`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, _resolve_ref, expand_videos, load_experiment


def cmd_info(a):
    from . import platform_info
    from .config import detect_platform
    print(json.dumps({"detected_platform": detect_platform(), **platform_info.snapshot()}, indent=2))


def cmd_extract(a):
    from .video import FrameCache
    for v in expand_videos(a.videos):
        c = FrameCache(v, a.frames_dir, a.max_side)
        i = c.info
        print(f"{Path(v).name}: {i.width}x{i.height} @ {i.fps:.2f} fps, {i.n_frames} frames ({i.duration_s:.1f}s)")
        idxs = c.ensure_fps(a.fps)
        print(f"  cached {len(idxs)} frames @ {a.fps} fps -> {c.dir}")


def cmd_import_gt(a):
    from .groundtruth import merge_records
    from .groundtruth.importers import IMPORTERS, detect_format
    fmt = a.format or detect_format(a.annotations)
    for v in expand_videos(a.video):
        recs = IMPORTERS[fmt](a.annotations, v, assume_complete=a.assume_complete, fps=a.fps)
        if not recs:
            print(f"{Path(v).name}: no matching annotations")
            continue
        p = merge_records(a.gt_dir, v, recs)
        n_obs = sum(len(r["obstacles"] or []) for r in recs)
        n_safe = sum(r["safe_to_land"] is not None for r in recs)
        print(f"{Path(v).name}: {len(recs)} frames ({n_safe} with landing verdict, {n_obs} obstacles) -> {p}")


def cmd_teacher(a):
    from .groundtruth.teacher import label_video
    tcfg = _resolve_ref(a.teacher, "models")
    for v in expand_videos(a.videos):
        label_video(v, tcfg, gt_dir=a.gt_dir, frames_dir=a.frames_dir, fps=a.fps, max_side=a.max_side,
                    concurrency=a.concurrency, max_frames=a.max_frames, overwrite=a.overwrite)


def cmd_run(a):
    from .runners import run_experiment
    runs = load_experiment(a.experiment, a.set, a.platform)
    if a.videos:
        for r in runs:
            r["videos"] = expand_videos(a.videos)
    if a.dry_run:
        import yaml
        print(yaml.safe_dump(runs, sort_keys=False))
        return
    dirs = [run_experiment(r) for r in runs]
    if len(dirs) > 1:
        from .report import compare_runs
        csv_p, md_p = compare_runs(dirs, dirs[-1].parent / f"compare_{runs[0]['name']}")
        print(f"[sweep] comparison: {md_p}")


def cmd_report(a):
    from .report import compare_runs
    dirs = []
    for p in a.runs:
        p = Path(p)
        dirs += sorted(d for d in p.iterdir() if (d / "metrics.json").exists()) if not (p / "metrics.json").exists() else [p]
    csv_p, md_p = compare_runs(dirs, Path(a.out))
    print(md_p.read_text())
    print(f"wrote {csv_p} and {md_p}")


def cmd_view(a):
    from .config import ROOT as root
    from .viewer import build_index, build_view
    dirs = []
    for p in a.runs:
        p = Path(p)
        dirs += [p] if (p / "metrics.json").exists() else sorted(d.parent for d in p.rglob("metrics.json"))
    for d in dirs:
        print(f"wrote {build_view(d)}")
    print(f"wrote {build_index(Path(a.results) if a.results else root / 'results')}")


def cmd_serve(a):
    import functools
    import http.server
    import socket

    from .viewer import build_index
    root = Path(a.dir)
    build_index(root)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    try:
        ips = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")]
    except OSError:
        ips = []
    print(f"Serving {root} on port {a.port}. Open from another device:")
    for ip in ips or ["<jetson-ip>"]:
        print(f"  http://{ip}:{a.port}/")
    http.server.ThreadingHTTPServer((a.host, a.port), handler).serve_forever()


def main(argv=None):
    import os
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(prog="trails", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="print platform snapshot")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("extract", help="decode videos and cache downscaled frames")
    s.add_argument("videos", nargs="+")
    s.add_argument("--fps", type=float, default=1.0)
    s.add_argument("--max-side", type=int, default=1280)
    s.add_argument("--frames-dir", default=str(ROOT / "data/frames"))
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("import-gt", help="convert existing annotations (csv/coco/cvat/jsonl) to canonical GT")
    s.add_argument("annotations")
    s.add_argument("--video", nargs="+", required=True, help="video file(s) the annotations refer to")
    s.add_argument("--format", choices=["csv", "coco", "cvat", "jsonl"])
    s.add_argument("--assume-complete", action="store_true",
                   help="frames without boxes (sampled at --fps) are labelled as having no obstacles")
    s.add_argument("--fps", type=float, default=1.0)
    s.add_argument("--gt-dir", default=str(ROOT / "data/gt"))
    s.set_defaults(fn=cmd_import_gt)

    s = sub.add_parser("teacher", help="pseudo-label frames with a large teacher VLM")
    s.add_argument("videos", nargs="+")
    s.add_argument("--teacher", default="claude_teacher", help="model config name/path (configs/models/)")
    s.add_argument("--fps", type=float, default=1.0)
    s.add_argument("--max-side", type=int, default=1280)
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--max-frames", type=int)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--gt-dir", default=str(ROOT / "data/gt"))
    s.add_argument("--frames-dir", default=str(ROOT / "data/frames"))
    s.set_defaults(fn=cmd_teacher)

    s = sub.add_parser("run", help="run an experiment config")
    s.add_argument("experiment")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    s.add_argument("--platform", help="override platform (orin_nano, thor, ...)")
    s.add_argument("--videos", nargs="+", help="override the experiment's videos")
    s.add_argument("--dry-run", action="store_true", help="print resolved configs and exit")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="compare runs: results dirs or a parent results/ directory")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", default=str(ROOT / "results/compare"))
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("view", help="build the HTML viewer (view.html) for runs, plus results/index.html")
    s.add_argument("runs", nargs="+", help="run dirs, or a parent dir to build every run under it")
    s.add_argument("--results", help="where to write index.html (default: results/)")
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("serve", help="serve results/ over HTTP so other devices on the network can open the viewer")
    s.add_argument("--dir", default=str(ROOT / "results"))
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8800)
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
