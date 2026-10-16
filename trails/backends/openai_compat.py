"""OpenAI-compatible chat-completions client.

One backend for every inference server: llama.cpp `llama-server` (GGUF, fits 3B Q4 VLMs into the
8 GB Orin Nano), vLLM and TensorRT-LLM `trtllm-serve` (Thor), Ollama (`/v1`). Streaming is used so
time-to-first-token can be measured client-side.

Model config keys: base_url (e.g. http://localhost:8080/v1), model (server-side name), api_key_env,
jpeg_quality, timeout_s, extra_body (merged into the request, e.g. {"top_k": 1})
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

import requests

from .base import GenConfig, Result, VLMBackend


def encode_jpeg(img, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class OpenAICompatBackend(VLMBackend):
    name = "openai"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.url = cfg.get("base_url", "http://localhost:8080/v1").rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "") or "none"
        self.session.headers["Authorization"] = f"Bearer {key}"

    def generate(self, images, prompt, gen: GenConfig) -> Result:
        t0 = time.perf_counter()
        q = int(self.cfg.get("jpeg_quality", 90))
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg(im, q)}"}}
                   for im in images]
        content.append({"type": "text", "text": prompt})
        body = {"model": self.cfg.get("model", "default"), "messages": [{"role": "user", "content": content}],
                "max_tokens": gen.max_new_tokens, "temperature": gen.temperature, "stream": True,
                "stream_options": {"include_usage": True}, **(self.cfg.get("extra_body") or {})}
        t1 = time.perf_counter()
        first = None
        chunks: list[str] = []
        n_chunks = 0
        usage = {}
        with self.session.post(self.url, json=body, stream=True, timeout=float(self.cfg.get("timeout_s", 300))) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if first is None:
                            first = time.perf_counter()
                        chunks.append(piece)
                        n_chunks += 1
        t2 = time.perf_counter()
        first = first or t2
        meta = {"input_size": images[0].size if images else None}
        if usage.get("prompt_tokens_details"):
            meta["prompt_tokens_details"] = usage["prompt_tokens_details"]
        return Result(text="".join(chunks).strip(), total_ms=(t2 - t0) * 1000, preprocess_ms=(t1 - t0) * 1000,
                      ttft_ms=(first - t1) * 1000, decode_ms=(t2 - first) * 1000,
                      n_in_tokens=usage.get("prompt_tokens"), n_out_tokens=usage.get("completion_tokens", n_chunks),
                      meta=meta)

    def close(self):
        self.session.close()
