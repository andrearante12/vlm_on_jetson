"""Backend interface: every model, local or remote, is driven through `generate()`.

Timing breakdown (all milliseconds, wall clock on the calling thread):
  preprocess  image -> tensors / encoded request (incl. host->device copy)
  ttft        end of preprocess -> first generated token (vision encoder + LLM prefill; for HTTP
              backends also network + server queueing)
  decode      first token -> last token
  total       start of generate() -> return
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from PIL import Image


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0

    @classmethod
    def from_cfg(cls, d: dict | None) -> "GenConfig":
        d = d or {}
        return cls(max_new_tokens=int(d.get("max_new_tokens", 256)), temperature=float(d.get("temperature", 0.0)))


@dataclass
class Result:
    text: str
    total_ms: float
    preprocess_ms: float | None = None
    ttft_ms: float | None = None
    decode_ms: float | None = None
    n_in_tokens: int | None = None
    n_out_tokens: int | None = None
    peak_mem_mb: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def tok_per_s(self) -> float | None:
        if self.n_out_tokens and self.decode_ms and self.n_out_tokens > 1:
            return (self.n_out_tokens - 1) / (self.decode_ms / 1000)
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tok_per_s"] = None if self.tok_per_s is None else round(self.tok_per_s, 2)
        return d


class VLMBackend:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, images: list[Image.Image], prompt: str, gen: GenConfig) -> Result:
        raise NotImplementedError

    def warmup(self, image: Image.Image, prompt: str, gen: GenConfig, n: int = 2) -> None:
        for _ in range(n):
            self.generate([image], prompt, gen)

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        """Static info recorded alongside results (model id, dtype, quantization...)."""
        return {"backend": self.name, **{k: v for k, v in self.cfg.items() if k not in ("prompts",)}}


def get_backend(cfg: dict) -> VLMBackend:
    kind = cfg.get("backend")
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(cfg)
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(cfg)
    if kind == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(cfg)
    if kind == "fake":
        from .fake import FakeBackend
        return FakeBackend(cfg)
    raise ValueError(f"unknown backend {kind!r} (expected hf | openai | anthropic | fake)")
