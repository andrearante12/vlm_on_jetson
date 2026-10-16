"""Canonical GT files: data/gt/<video_stem>.jsonl, one schema.py record per labelled frame."""

from __future__ import annotations

import json
from pathlib import Path


def gt_path(gt_dir: str | Path, video: str | Path) -> Path:
    return Path(gt_dir) / f"{Path(video).stem}.jsonl"


def load_gt(gt_dir: str | Path, video: str | Path, provenance_prefix: str | None = None) -> dict[int, dict]:
    """Returns {frame_idx: record}. If several provenances label the same frame, annotation beats teacher."""
    p = gt_path(gt_dir, video)
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if provenance_prefix and not str(rec.get("provenance", "")).startswith(provenance_prefix):
            continue
        prev = out.get(rec["idx"])
        if prev is None or _rank(rec) >= _rank(prev):
            out[rec["idx"]] = rec
    return out


def _rank(rec: dict) -> int:
    prov = str(rec.get("provenance", ""))
    return 2 if prov.startswith("annotation") else 1 if prov.startswith("teacher") else 0


def write_gt(gt_dir: str | Path, video: str | Path, records: list[dict]) -> Path:
    p = gt_path(gt_dir, video)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in sorted(records, key=lambda r: (r["idx"], str(r.get("provenance")))):
            f.write(json.dumps(r) + "\n")
    return p


def merge_records(gt_dir: str | Path, video: str | Path, new: list[dict]) -> Path:
    """Add/replace records keyed by (idx, provenance), keeping other provenances intact."""
    p = gt_path(gt_dir, video)
    existing: dict[tuple, dict] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["idx"], r.get("provenance"))] = r
    for r in new:
        existing[(r["idx"], r.get("provenance"))] = r
    return write_gt(gt_dir, video, list(existing.values()))
