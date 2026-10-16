"""Task = prompt + parser + scorer for one output mode.

Tasks are configured from configs/tasks/<name>.yaml; `type` selects the implementation and every
other key (prompt, vocab, iou threshold, ...) is passed through as task config.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schema import DEFAULT_OBSTACLE_CLASSES, DEFAULT_SURFACES, Vocab


class Task:
    type: str = "base"
    #: GT fields that must be non-null for a GT record to be scored by this task
    required_gt: tuple[str, ...] = ()
    #: dotted metric keys shown in reports and comparison tables
    headline: tuple[str, ...] = ()

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", self.type)
        self.obstacles = Vocab(cfg.get("obstacle_classes") or DEFAULT_OBSTACLE_CLASSES)
        self.surfaces = Vocab(cfg.get("surface_classes") or DEFAULT_SURFACES)

    def prompt(self, model_cfg: dict | None = None) -> str:
        """Model configs may override a task prompt (e.g. model-specific grounding syntax)."""
        override = ((model_cfg or {}).get("prompts") or {}).get(self.type)
        return (override or self.cfg["prompt"]).strip()

    def parse(self, text: str, ctx: dict | None = None) -> tuple[dict, bool]:
        """Return (partial canonical record, parse_ok)."""
        raise NotImplementedError

    def scorable(self, gt: dict) -> bool:
        return all(gt.get(f) is not None for f in self.required_gt)

    def score(self, pairs: list[tuple[dict | None, dict]]) -> dict[str, Any]:
        """pairs: (pred or None if the model produced no decision in time, gt)."""
        raise NotImplementedError


def get_task(cfg: dict) -> Task:
    from .caption import CaptionTask
    from .grounding import GroundingTask
    from .landing_json import LandingJsonTask

    types = {t.type: t for t in (LandingJsonTask, CaptionTask, GroundingTask)}
    t = cfg.get("type", cfg.get("name"))
    if t not in types:
        raise ValueError(f"unknown task type {t!r}; expected one of {sorted(types)}")
    return types[t](cfg)


# ---------- shared parsing helpers ----------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any | None:
    """Find the first parseable JSON object/array in model output (fences, prose, trailing junk)."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            frag = cand[start:i + 1]
                            obj = _loads_lenient(frag)
                            if obj is not None:
                                return obj
                            break
                start = cand.find(opener, start + 1)
    return None


def _loads_lenient(s: str) -> Any | None:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Common small-model slips: single quotes, Python literals, trailing commas.
    fixed = re.sub(r",\s*([}\]])", r"\1", s)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    if '"' not in fixed:
        fixed = fixed.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def parse_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "safe", "y", "1"):
            return True
        if s in ("false", "no", "unsafe", "n", "0"):
            return False
    return None


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    if tp:
        f = 2 * p * r / (p + r)
    else:
        f = 0.0 if fp + fn else None  # no hits but something to find (or spurious hits) -> F1 = 0
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f), "tp": tp, "fp": fp, "fn": fn}


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def safety_metrics(pairs: list[tuple[dict | None, dict]]) -> dict:
    """Binary safe/unsafe metrics. 'Unsafe' is the positive class: missing a hazard is the costly error.

    A missing prediction (no decision / parse failure) is counted as *not unsafe*, i.e. the system
    failed to warn, which is the conservative way to score a landing-safety monitor.
    """
    tp = fp = fn = tn = 0
    false_safe = n_unsafe = 0
    correct = n = 0
    for pred, gt in pairs:
        g = gt.get("safe_to_land")
        if g is None:
            continue
        p = None if pred is None else pred.get("safe_to_land")
        n += 1
        correct += int(p is not None and p == g)
        pred_unsafe = p is False
        gt_unsafe = g is False
        tp += pred_unsafe and gt_unsafe
        fp += pred_unsafe and not gt_unsafe
        fn += (not pred_unsafe) and gt_unsafe
        tn += (not pred_unsafe) and not gt_unsafe
        if gt_unsafe:
            n_unsafe += 1
            false_safe += int(p is True)
    out = {"n": n, "accuracy": _r(correct / n) if n else None}
    out.update({f"unsafe_{k}": v for k, v in prf(tp, fp, fn).items() if k in ("precision", "recall", "f1")})
    out["false_safe_rate"] = _r(false_safe / n_unsafe) if n_unsafe else None
    out["gt_unsafe_frac"] = _r(n_unsafe / n) if n else None
    return out
