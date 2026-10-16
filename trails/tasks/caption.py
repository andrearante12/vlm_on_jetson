"""Free-text scene description, scored against a reference caption.

The prompt asks the model to end with 'VERDICT: SAFE' or 'VERDICT: UNSAFE' so the free-text mode can
also be scored on the landing decision, and hazard mentions are checked against GT obstacle classes.
BLEU-4 / ROUGE-L are reported for comparability with DriveLM-style evaluations (incl. LiteVLM), but
they mostly measure phrasing similarity; hazard recall and the verdict are the operationally useful ones.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import Task, prf, safety_metrics

_VERDICT = re.compile(r"verdict\s*[:\-]?\s*\**\s*(safe|unsafe|not safe)", re.I)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def bleu4(hyp: str, ref: str) -> float:
    h, r = _tokens(hyp), _tokens(ref)
    if not h or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, 5):
        hn = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        overlap = sum((hn & rn).values())
        total = max(1, sum(hn.values()))
        log_p += math.log((overlap + 1) / (total + 1)) / 4  # add-one smoothing (Lin & Och 2004)
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(log_p)


def rouge_l(hyp: str, ref: str) -> float:
    h, r = _tokens(hyp), _tokens(ref)
    if not h or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in h:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return 2 * p * rc / (p + rc)


class CaptionTask(Task):
    type = "caption"
    required_gt = ()
    headline = ("safety.accuracy", "safety.false_safe_rate", "hazard_mention.f1", "rouge_l", "bleu4", "no_decision_rate")

    def parse(self, text, ctx=None):
        text = (text or "").strip()
        m = _VERDICT.search(text)
        safe = None if not m else m.group(1).lower() == "safe"
        caption = _VERDICT.sub("", text).strip() if text else None
        return {"caption": caption or None, "safe_to_land": safe,
                "mentioned": sorted(self.obstacles.mentioned(caption or ""))}, bool(caption)

    def scorable(self, gt):
        return gt.get("caption") is not None or gt.get("obstacles") is not None or gt.get("safe_to_land") is not None

    def score(self, pairs):
        pairs = [(p, g) for p, g in pairs if self.scorable(g)]
        out: dict = {"n_frames": len(pairs),
                     "no_decision_rate": round(sum(p is None for p, _ in pairs) / len(pairs), 4) if pairs else None}
        refs = [(p, g) for p, g in pairs if g.get("caption")]
        if refs:
            out["bleu4"] = round(sum(bleu4((p or {}).get("caption") or "", g["caption"]) for p, g in refs) / len(refs), 4)
            out["rouge_l"] = round(sum(rouge_l((p or {}).get("caption") or "", g["caption"]) for p, g in refs) / len(refs), 4)
        tp = fp = fn = 0
        for p, g in pairs:
            if g.get("obstacles") is None:
                continue
            gset = {o["type"] for o in g["obstacles"]}
            pset = set((p or {}).get("mentioned") or [])
            tp += len(gset & pset)
            fp += len(pset - gset)
            fn += len(gset - pset)
        if tp + fp + fn:
            out["hazard_mention"] = prf(tp, fp, fn)
        out["verdict_parse_rate"] = round(sum(bool(p) and p.get("safe_to_land") is not None for p, _ in pairs)
                                          / len(pairs), 4) if pairs else None
        out["safety"] = safety_metrics(pairs)
        return out
