"""Canonical per-frame record shared by ground truth and predictions.

    {
      "video": "<stem>", "idx": 120, "t_ms": 4000.0,
      "safe_to_land": true | false | null,
      "surface": "grass" | ... | null,
      "obstacles": [{"type": "person", "grid_cell": "center" | null, "bbox": [x1,y1,x2,y2] | null}],
      "caption": "..." | null,
      "provenance": "annotation:coco" | "teacher:claude-opus-5-5" | "model:internvl3_1b"
    }

bbox is normalised to [0, 1] in (x1, y1, x2, y2) order. Fields that are null are "not labelled"
and are skipped by scorers rather than counted as wrong.
"""

from __future__ import annotations

import re

GRID_CELLS = [
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]

DEFAULT_OBSTACLE_CLASSES: dict[str, list[str]] = {
    "person": ["person", "people", "pedestrian", "human", "man", "woman", "child", "hiker", "crowd"],
    "vehicle": ["vehicle", "car", "truck", "bus", "van", "motorcycle", "bicycle", "bike", "atv", "tractor"],
    "tree": ["tree", "trees", "branch", "branches", "canopy", "foliage", "bush", "shrub"],
    "building": ["building", "house", "structure", "shed", "roof", "wall", "tent"],
    "pole": ["pole", "post", "tower", "mast", "lamp post", "light pole", "sign"],
    "wire": ["wire", "power line", "powerline", "cable", "wires", "power lines"],
    "water": ["water", "river", "lake", "pond", "stream", "puddle", "creek"],
    "rock": ["rock", "rocks", "boulder", "stone", "stones", "debris", "log"],
    "animal": ["animal", "dog", "cow", "horse", "deer", "bird", "livestock"],
    "fence": ["fence", "railing", "barrier", "gate"],
    "slope": ["slope", "steep slope", "cliff", "ditch", "hill", "uneven terrain", "drop-off"],
}

DEFAULT_SURFACES: dict[str, list[str]] = {
    "grass": ["grass", "lawn", "field", "meadow", "turf"],
    "dirt": ["dirt", "soil", "mud", "earth", "trail"],
    "gravel": ["gravel", "pebbles"],
    "pavement": ["pavement", "asphalt", "concrete", "road", "parking lot", "tarmac", "paved"],
    "sand": ["sand", "beach"],
    "rock": ["rock", "rocky", "stone"],
    "water": ["water"],
    "snow": ["snow", "ice"],
    "vegetation": ["vegetation", "bushes", "shrubs", "forest", "crops", "tall grass"],
    "roof": ["roof", "rooftop"],
}


class Vocab:
    """Maps free-form model labels onto a closed class set (unknown -> 'other')."""

    def __init__(self, classes: dict[str, list[str]]):
        self.classes = list(classes)
        self._lookup: dict[str, str] = {}
        for canon, syns in classes.items():
            self._lookup[_norm(canon)] = canon
            for s in syns:
                self._lookup[_norm(s)] = canon
        # Longest synonyms first so "power line" wins over "line" in substring matching.
        self._by_len = sorted(self._lookup, key=len, reverse=True)

    def canon(self, label: str | None) -> str | None:
        if label is None:
            return None
        n = _norm(str(label))
        if not n:
            return None
        if n in self._lookup:
            return self._lookup[n]
        if n.endswith("s") and n[:-1] in self._lookup:
            return self._lookup[n[:-1]]
        for syn in self._by_len:
            if re.search(rf"\b{re.escape(syn)}\b", n):
                return self._lookup[syn]
        return "other"

    def mentioned(self, text: str) -> set[str]:
        """Canonical classes mentioned anywhere in free text."""
        n = _norm(text)
        found = set()
        for syn in self._by_len:
            if re.search(rf"\b{re.escape(syn)}s?\b", n):
                found.add(self._lookup[syn])
        return found


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.strip().lower())


_CELL_ALIASES = {
    "upper": "top", "lower": "bottom", "mid": "middle", "centre": "center", "central": "center",
    "middle": "middle", "left": "left", "right": "right", "top": "top", "bottom": "bottom", "center": "center",
}


def canon_grid_cell(cell: str | None) -> str | None:
    if not cell:
        return None
    words = [_CELL_ALIASES.get(w, w) for w in re.split(r"[\s_\-]+", str(cell).strip().lower()) if w]
    vert = next((w for w in words if w in ("top", "bottom")), None)
    horiz = next((w for w in words if w in ("left", "right")), None)
    if vert and horiz:
        return f"{vert}-{horiz}"
    if vert:
        return f"{vert}-center"
    if horiz:
        return f"middle-{horiz}"
    if "center" in words or "middle" in words:
        return "center"
    return None


def grid_cell_from_bbox(bbox: list[float] | None) -> str | None:
    if not bbox:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    col = min(2, max(0, int(cx * 3)))
    row = min(2, max(0, int(cy * 3)))
    return GRID_CELLS[row * 3 + col]


def normalize_bbox(b, input_size: tuple[int, int] | None = None, fmt: str = "auto") -> list[float] | None:
    """Convert a model bbox to normalised [0,1] xyxy.

    fmt: 'norm1' (already 0-1), 'norm1000' (Qwen2-VL/InternVL convention), 'pixels' (absolute pixels of
    the image the model saw; needs input_size=(w, h)), or 'auto' (guess from magnitude).
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in list(b)[:4])
    except (TypeError, ValueError):
        return None
    m = max(x1, y1, x2, y2)
    if fmt == "auto":
        fmt = "norm1" if m <= 1.0 else ("pixels" if (m > 1000 and input_size) else "norm1000")
    if fmt == "norm1000":
        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    elif fmt == "pixels":
        if not input_size:
            return None
        w, h = input_size
        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    clip = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    box = [clip(x1), clip(y1), clip(x2), clip(y2)]
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        return None
    return [round(v, 4) for v in box]


def empty_record(video: str, idx: int, t_ms: float, provenance: str) -> dict:
    return {"video": video, "idx": idx, "t_ms": t_ms, "safe_to_land": None, "surface": None,
            "obstacles": None, "caption": None, "provenance": provenance}
