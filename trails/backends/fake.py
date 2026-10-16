"""Deterministic stand-in model: fixed latency, canned response. Used for tests and for exercising
the runners/scoring on a new machine before any model weights are downloaded."""

from __future__ import annotations

import random
import time

from .base import GenConfig, Result, VLMBackend


class FakeBackend(VLMBackend):
    name = "fake"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.latency_ms = float(cfg.get("latency_ms", 300))
        self.jitter_ms = float(cfg.get("jitter_ms", 0))
        self.response = cfg.get("response", '{"safe_to_land": true, "surface": "grass", "obstacles": []}')
        self.sleep = cfg.get("sleep", True)  # False: report latency without waiting (simulated clock)
        self._rng = random.Random(cfg.get("seed", 0))
        self.calls = 0

    def generate(self, images, prompt, gen: GenConfig) -> Result:
        self.calls += 1
        lat = max(0.0, self.latency_ms + self._rng.uniform(-self.jitter_ms, self.jitter_ms))
        if self.sleep:
            time.sleep(lat / 1000)
        text = self.response(images, prompt) if callable(self.response) else self.response
        return Result(text=text, total_ms=lat, preprocess_ms=0.0, ttft_ms=lat * 0.6, decode_ms=lat * 0.4,
                      n_in_tokens=0, n_out_tokens=len(text.split()))

    def describe(self):
        d = super().describe()
        d.pop("response", None)
        return d
