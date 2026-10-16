"""Convert existing annotations into canonical GT records.

Supported inputs:
  csv    one row per obstacle (or one row with empty obstacle_type for "no obstacles"):
           video, frame | t_ms | t_s, safe_to_land, surface, caption, obstacle_type, x1, y1, x2, y2, grid_cell
         Box coords may be normalised (0-1) or pixels (detected; pixels are divided by the video size).
  coco   COCO detection JSON. Frame index comes from images[].frame_index / frame_id, or the last number
         in file_name. Optional per-image keys safe_to_land / surface / caption are carried over.
  cvat   "CVAT for video 1.1" XML: <track label=..><box frame=.. xtl ytl xbr ybr outside=..>, and
         <tag label="safe"|"unsafe" frame=..> for the landing verdict.
  jsonl  already canonical; validated and copied.

Only frames that appear in the annotation are emitted, unless assume_complete=True, in which case
every frame sampled at `fps` gets a record (frames without boxes become "labelled: no obstacles").
"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from ..schema import Vocab, canon_grid_cell, grid_cell_from_bbox, normalize_bbox, DEFAULT_OBSTACLE_CLASSES, \
    DEFAULT_SURFACES
from ..tasks.base import parse_bool
from ..video import VideoInfo, indices_for_fps, probe

_obst = Vocab(DEFAULT_OBSTACLE_CLASSES)
_surf = Vocab(DEFAULT_SURFACES)


def _rec(video: str, info: VideoInfo, idx: int, prov: str) -> dict:
    return {"video": Path(video).stem, "idx": idx, "t_ms": round(info.t_ms(idx), 3), "safe_to_land": None,
            "surface": None, "obstacles": None, "caption": None, "provenance": prov}


def _box(coords, info: VideoInfo) -> list[float] | None:
    vals = [float(c) for c in coords]
    if max(vals) <= 1.0:
        return normalize_bbox(vals, fmt="norm1")
    return normalize_bbox(vals, input_size=(info.width, info.height), fmt="pixels")


def _add_obstacle(rec: dict, label: str, bbox: list[float] | None, cell: str | None = None) -> None:
    rec["obstacles"] = rec["obstacles"] or []
    typ = _obst.canon(label)
    if typ:
        rec["obstacles"].append({"type": typ, "bbox": bbox, "grid_cell": canon_grid_cell(cell) or grid_cell_from_bbox(bbox)})


def _complete(records: dict[int, dict], video: str, info: VideoInfo, fps: float, prov: str) -> dict[int, dict]:
    for idx in indices_for_fps(info, fps):
        r = records.setdefault(idx, _rec(video, info, idx, prov))
        if r["obstacles"] is None:
            r["obstacles"] = []
    return records


def import_csv(path: str, video: str, prov: str = "annotation:csv", **_) -> list[dict]:
    info = probe(video)
    recs: dict[int, dict] = {}
    stem = Path(video).stem
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("video") and Path(row["video"]).stem != stem:
                continue
            if row.get("frame"):
                idx = int(float(row["frame"]))
            elif row.get("t_ms"):
                idx = info.idx_at(float(row["t_ms"]))
            elif row.get("t_s"):
                idx = info.idx_at(float(row["t_s"]) * 1000)
            else:
                raise ValueError("csv rows need one of frame / t_ms / t_s")
            r = recs.setdefault(idx, _rec(video, info, idx, prov))
            if row.get("safe_to_land"):
                r["safe_to_land"] = parse_bool(row["safe_to_land"])
            if row.get("surface"):
                r["surface"] = _surf.canon(row["surface"])
            if row.get("caption"):
                r["caption"] = row["caption"]
            if r["obstacles"] is None:
                r["obstacles"] = []
            if row.get("obstacle_type"):
                coords = [row.get(k) for k in ("x1", "y1", "x2", "y2")]
                bbox = _box(coords, info) if all(c not in (None, "") for c in coords) else None
                _add_obstacle(r, row["obstacle_type"], bbox, row.get("grid_cell"))
    return list(recs.values())


def import_coco(path: str, video: str, prov: str = "annotation:coco", assume_complete: bool = False,
                fps: float = 1.0, **_) -> list[dict]:
    info = probe(video)
    data = json.loads(Path(path).read_text())
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    stem = Path(video).stem
    recs: dict[int, dict] = {}
    img_idx: dict[int, int] = {}
    img_size: dict[int, tuple[int, int]] = {}
    for im in data["images"]:
        if im.get("video") and Path(im["video"]).stem != stem:
            continue
        idx = im.get("frame_index", im.get("frame_id"))
        if idx is None:
            nums = re.findall(r"\d+", Path(im["file_name"]).stem)
            if not nums:
                continue
            idx = int(nums[-1])
        idx = int(idx)
        img_idx[im["id"]] = idx
        img_size[im["id"]] = (im.get("width") or info.width, im.get("height") or info.height)
        r = recs.setdefault(idx, _rec(video, info, idx, prov))
        r["obstacles"] = r["obstacles"] or []  # an annotated image with no boxes = "no obstacles"
        if "safe_to_land" in im:
            r["safe_to_land"] = parse_bool(im["safe_to_land"])
        if im.get("surface"):
            r["surface"] = _surf.canon(im["surface"])
        if im.get("caption"):
            r["caption"] = im["caption"]
    for a in data.get("annotations", []):
        if a["image_id"] not in img_idx:
            continue
        x, y, w, h = a["bbox"]
        bbox = normalize_bbox([x, y, x + w, y + h], input_size=img_size[a["image_id"]], fmt="pixels")
        _add_obstacle(recs[img_idx[a["image_id"]]], cats.get(a["category_id"], "other"), bbox)
    if assume_complete:
        _complete(recs, video, info, fps, prov)
    return list(recs.values())


def import_cvat(path: str, video: str, prov: str = "annotation:cvat", assume_complete: bool = False,
                fps: float = 1.0, **_) -> list[dict]:
    info = probe(video)
    root = ET.parse(path).getroot()
    w = int(root.findtext(".//original_size/width") or info.width)
    h = int(root.findtext(".//original_size/height") or info.height)
    recs: dict[int, dict] = {}
    boxes: dict[int, list] = defaultdict(list)
    for track in root.iter("track"):
        label = track.get("label", "other")
        for b in track.iter("box"):
            if b.get("outside") == "1":
                continue
            frame = int(b.get("frame"))
            coords = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            boxes[frame].append((label, normalize_bbox(coords, input_size=(w, h), fmt="pixels")))
    for img in root.iter("image"):  # "CVAT for images" export of extracted frames
        frame = int(img.get("id"))
        for b in img.iter("box"):
            coords = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            boxes[frame].append((b.get("label", "other"),
                                 normalize_bbox(coords, input_size=(int(img.get("width", w)), int(img.get("height", h))),
                                                fmt="pixels")))
    for frame, items in boxes.items():
        r = recs.setdefault(frame, _rec(video, info, frame, prov))
        r["obstacles"] = []
        for label, bb in items:
            _add_obstacle(r, label, bb)
    for tag in root.iter("tag"):
        frame = int(tag.get("frame", -1))
        if frame < 0:
            continue
        r = recs.setdefault(frame, _rec(video, info, frame, prov))
        lab = tag.get("label", "").lower()
        if lab in ("safe", "safe_to_land", "landable"):
            r["safe_to_land"] = True
        elif lab in ("unsafe", "not_safe", "no_landing"):
            r["safe_to_land"] = False
    if assume_complete:
        _complete(recs, video, info, fps, prov)
    return list(recs.values())


def import_jsonl(path: str, video: str, prov: str | None = None, **_) -> list[dict]:
    info = probe(video)
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "idx" not in r:
            r["idx"] = info.idx_at(float(r["t_ms"]))
        base = _rec(video, info, r["idx"], prov or r.get("provenance") or "annotation:jsonl")
        base.update({k: v for k, v in r.items() if k in base})
        if base["obstacles"]:
            for o in base["obstacles"]:
                o["type"] = _obst.canon(o.get("type"))
                o.setdefault("bbox", None)
                o["grid_cell"] = canon_grid_cell(o.get("grid_cell")) or grid_cell_from_bbox(o["bbox"])
        out.append(base)
    return out


IMPORTERS = {"csv": import_csv, "coco": import_coco, "cvat": import_cvat, "jsonl": import_jsonl}


def detect_format(path: str) -> str:
    suf = Path(path).suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf == ".xml":
        return "cvat"
    if suf == ".jsonl":
        return "jsonl"
    if suf == ".json":
        return "coco"
    raise ValueError(f"cannot infer annotation format of {path}; pass --format")
