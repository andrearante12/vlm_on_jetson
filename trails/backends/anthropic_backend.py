"""Claude as a teacher/labeler (not a benchmark subject: it runs in the cloud, so its latency is
meaningless for on-board real-time evaluation).

Uses structured outputs (output_config.format) with the label JSON schema so every response parses,
and the server-side refusal fallback so an occasional safety-classifier decline is retried on a
fallback model instead of losing the frame.

Model config keys: model (default claude-opus-5-5), effort (low|medium|high|xhigh|max, default high),
json_schema (optional dict; enables structured outputs), max_tokens, jpeg_quality, max_retries
"""

from __future__ import annotations

import time

import anthropic

from .base import GenConfig, Result, VLMBackend
from .openai_compat import encode_jpeg


class AnthropicBackend(VLMBackend):
    name = "anthropic"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.client = anthropic.Anthropic(max_retries=int(cfg.get("max_retries", 5)))
        self.model = cfg.get("model", "claude-opus-5-5")

    def generate(self, images, prompt, gen: GenConfig) -> Result:
        t0 = time.perf_counter()
        q = int(self.cfg.get("jpeg_quality", 90))
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": encode_jpeg(im, q)}} for im in images]
        content.append({"type": "text", "text": prompt})
        output_config: dict = {"effort": self.cfg.get("effort", "high")}
        if self.cfg.get("json_schema"):
            output_config["format"] = {"type": "json_schema", "schema": self.cfg["json_schema"]}
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=int(self.cfg.get("max_tokens", 16000)),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config=output_config,
            messages=[{"role": "user", "content": content}],
        )
        t1 = time.perf_counter()
        meta = {"model": resp.model, "stop_reason": resp.stop_reason, "request_id": resp._request_id,
                "input_size": images[0].size if images else None}
        if resp.stop_reason == "refusal":
            meta["refusal_category"] = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            text = ""
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
        return Result(text=text.strip(), total_ms=(t1 - t0) * 1000, n_in_tokens=resp.usage.input_tokens,
                      n_out_tokens=resp.usage.output_tokens, meta=meta)
