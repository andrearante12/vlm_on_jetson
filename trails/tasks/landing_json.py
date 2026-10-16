"""Structured JSON output: safe_to_land + surface + obstacle list with coarse 3x3 grid location."""

from __future__ import annotations

from collections import Counter

from ..schema import canon_grid_cell, grid_cell_from_bbox, normalize_bbox
from .base import Task, extract_json, parse_bool, prf, safety_metrics


def parse_obstacles(raw, vocab, ctx: dict | None) -> list[dict]:
    ctx = ctx or {}
    out = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return out
    for o in raw:
        if isinstance(o, str):
            o = {"type": o}
        if not isinstance(o, dict):
            continue
        typ = vocab.canon(o.get("type") or o.get("label") or o.get("name") or o.get("class"))
        if typ is None:
            continue
        bb = o.get("bbox") or o.get("bbox_2d") or o.get("box")
        bbox = normalize_bbox(bb, ctx.get("input_size"), ctx.get("bbox_format", "auto")) if bb else None
        cell = canon_grid_cell(o.get("grid_cell") or o.get("location") or o.get("position")) or grid_cell_from_bbox(bbox)
        out.append({"type": typ, "grid_cell": cell, "bbox": bbox})
    return out


class LandingJsonTask(Task):
    type = "landing_json"
    required_gt = ("safe_to_land",)
    headline = ("safety.accuracy", "safety.unsafe_f1", "safety.false_safe_rate", "obstacle_type.f1",
                "parse_fail_rate", "no_decision_rate")

    def parse(self, text, ctx=None):
        obj = extract_json(text)
        if isinstance(obj, list):  # model returned only the obstacle list
            obj = {"obstacles": obj}
        if not isinstance(obj, dict):
            return {"safe_to_land": None, "surface": None, "obstacles": None}, False
        safe = parse_bool(obj.get("safe_to_land", obj.get("safe")))
        surface = self.surfaces.canon(obj.get("surface"))
        obstacles = parse_obstacles(obj.get("obstacles", obj.get("hazards", [])), self.obstacles, ctx)
        conf = obj.get("confidence")
        pred = {"safe_to_land": safe, "surface": surface, "obstacles": obstacles,
                "confidence": conf if isinstance(conf, (int, float)) else None}
        return pred, safe is not None

    def score(self, pairs):
        pairs = [(p, g) for p, g in pairs if self.scorable(g)]
        out = {"n_frames": len(pairs),
               "no_decision_rate": round(sum(p is None for p, _ in pairs) / len(pairs), 4) if pairs else None,
               "parse_fail_rate": round(sum(p is not None and p.get("safe_to_land") is None for p, _ in pairs)
                                        / len(pairs), 4) if pairs else None}
        out["safety"] = safety_metrics(pairs)

        surf = [(p.get("surface") if p else None, g["surface"]) for p, g in pairs if g.get("surface")]
        out["surface_accuracy"] = round(sum(a == b for a, b in surf) / len(surf), 4) if surf else None

        # Obstacles: only frames where GT lists obstacles (an empty list = labelled "none").
        obs_pairs = [(p, g) for p, g in pairs if g.get("obstacles") is not None]
        tp = fp = fn = 0
        tpc = fpc = fnc = 0
        presence_ok = 0
        per_class: dict[str, list[int]] = {}
        for p, g in obs_pairs:
            pobs = (p or {}).get("obstacles") or []
            gtypes = Counter(o["type"] for o in g["obstacles"])
            ptypes = Counter(o["type"] for o in pobs)
            m = gtypes & ptypes
            tp += sum(m.values())
            fp += sum((ptypes - gtypes).values())
            fn += sum((gtypes - ptypes).values())
            for cls, cnt in gtypes.items():
                rec = per_class.setdefault(cls, [0, 0])
                rec[0] += min(cnt, ptypes.get(cls, 0))
                rec[1] += cnt
            gcells = Counter((o["type"], o.get("grid_cell")) for o in g["obstacles"] if o.get("grid_cell"))
            if gcells:
                pcells = Counter((o["type"], o.get("grid_cell")) for o in pobs)
                mc = gcells & pcells
                tpc += sum(mc.values())
                fpc += sum((pcells - gcells).values())
                fnc += sum((gcells - pcells).values())
            presence_ok += int(bool(pobs) == bool(g["obstacles"]))
        if obs_pairs:
            out["obstacle_presence_accuracy"] = round(presence_ok / len(obs_pairs), 4)
            out["obstacle_type"] = prf(tp, fp, fn)
            out["obstacle_type_and_cell"] = prf(tpc, fpc, fnc) if tpc + fpc + fnc else None
            out["obstacle_recall_per_class"] = {c: round(a / b, 4) for c, (a, b) in sorted(per_class.items())}
        return out
