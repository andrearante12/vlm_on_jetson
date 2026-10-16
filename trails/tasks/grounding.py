"""Obstacle bounding boxes, scored by IoU matching against GT boxes.

Parses the three output conventions small VLMs actually produce:
  * JSON list  [{"label": "tree", "bbox": [x1, y1, x2, y2]}]   (our default prompt, 0-1000 coords)
  * Qwen2.5-VL [{"label": "tree", "bbox_2d": [x1, y1, x2, y2]}]  (absolute pixels of the resized input)
  * InternVL   <ref>tree</ref><box>[[x1, y1, x2, y2]]</box>      (0-1000 coords)
VLMs give no calibrated scores, so instead of mAP we report P/R/F1 at an IoU threshold (class-aware
and class-agnostic) plus mean IoU of matched boxes.
"""

from __future__ import annotations

import re

from ..schema import normalize_bbox
from .base import Task, extract_json, prf
from .landing_json import parse_obstacles

_INTERNVL_BOX = re.compile(r"<ref>(.*?)</ref>\s*<box>\s*\[(.*?)\]\s*</box>", re.S)


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: list[dict], gts: list[dict], thr: float, class_aware: bool) -> tuple[int, int, int, list[float]]:
    cands = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if class_aware and p["type"] != g["type"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= thr:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    tp = len(ious)
    return tp, len(preds) - tp, len(gts) - tp, ious


class GroundingTask(Task):
    type = "grounding"
    required_gt = ("obstacles",)
    headline = ("class_agnostic.f1", "class_aware.f1", "class_agnostic.mean_iou_matched", "no_decision_rate")

    def parse(self, text, ctx=None):
        ctx = ctx or {}
        obstacles: list[dict] = []
        for label, coords in _INTERNVL_BOX.findall(text or ""):
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", coords)]
            typ = self.obstacles.canon(label)
            for k in range(0, len(nums) - 3, 4):  # one <box> may hold several [[..],[..]]
                bb = normalize_bbox(nums[k:k + 4], fmt="norm1000")
                if bb and typ:
                    obstacles.append({"type": typ, "bbox": bb, "grid_cell": None})
        if not obstacles:
            obj = extract_json(text or "")
            if isinstance(obj, dict):
                obj = obj.get("obstacles") or obj.get("objects") or [obj]
            obstacles = [o for o in parse_obstacles(obj, self.obstacles, ctx) if o["bbox"]]
            ok = obj is not None
        else:
            ok = True
        return {"obstacles": obstacles}, ok

    def scorable(self, gt):
        obs = gt.get("obstacles")
        return obs is not None and all(o.get("bbox") for o in obs)

    def score(self, pairs):
        pairs = [(p, g) for p, g in pairs if self.scorable(g)]
        thr = float(self.cfg.get("iou_threshold", 0.5))
        out: dict = {"n_frames": len(pairs), "iou_threshold": thr,
                     "no_decision_rate": round(sum(p is None for p, _ in pairs) / len(pairs), 4) if pairs else None}
        for name, aware in (("class_aware", True), ("class_agnostic", False)):
            tp = fp = fn = 0
            ious: list[float] = []
            for p, g in pairs:
                a, b, c, i = greedy_match((p or {}).get("obstacles") or [], g["obstacles"], thr, aware)
                tp, fp, fn = tp + a, fp + b, fn + c
                ious += i
            out[name] = prf(tp, fp, fn)
            out[name]["mean_iou_matched"] = round(sum(ious) / len(ious), 4) if ious else None
        return out
