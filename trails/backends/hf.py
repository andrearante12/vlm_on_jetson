"""In-process HuggingFace transformers backend.

Two families:
  auto      AutoProcessor + AutoModelForImageTextToText with the model's chat template. Covers
            Qwen2.5-VL, InternVL3 "-hf" checkpoints, SmolVLM, Gemma 3, etc. Preferred.
  internvl  The original OpenGVLab trust_remote_code checkpoints (InternVL2.5-1B/2B, the models
            LiteVLM is built on) using the reference dynamic-tiling preprocessing. `max_tiles` is the
            knob that corresponds to LiteVLM's patch selection (fewer 448x448 patches -> fewer tokens).

Model config keys:
  model_id, family (auto|internvl), dtype (float16|bfloat16), device (cuda), attn_implementation,
  processor_kwargs (passed to from_pretrained, e.g. min_pixels/max_pixels for Qwen),
  processor_call_kwargs (passed per call, e.g. max_patches for InternVL-hf),
  max_tiles / image_size / use_thumbnail (internvl family), load_kwargs (extra from_pretrained args)
"""

from __future__ import annotations

import time

from PIL import Image

from .base import GenConfig, Result, VLMBackend

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _patch_sdpa_gqa():
    """NVIDIA's Jetson torch builds can be 2.5 *pre-releases* (e.g. 2.5.0a0+nv24.08) that lack
    scaled_dot_product_attention(enable_gqa=...), while transformers treats them as >= 2.5 and passes
    it, which surfaces as a misleading "Missing **kwargs" TypeError. Fall back to repeat_kv instead."""
    import torch
    import torch.nn.functional as F

    q = torch.zeros(1, 1, 1, 2)
    try:
        F.scaled_dot_product_attention(q, q, q, enable_gqa=False)
    except TypeError:
        import transformers.integrations.sdpa_attention as sdpa

        sdpa._is_torch_greater_or_equal_than_2_5 = False


def _disable_allocator_warmup():
    """transformers pre-allocates one contiguous block the size of the whole model before loading.
    On Jetson (NvMap, unified memory) a single ~2 GB block fails even when that much memory is free
    in smaller pieces, so skip the warmup on integrated GPUs; loading still works, just un-pooled."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).is_integrated:
        import transformers.modeling_utils as mu

        if hasattr(mu, "caching_allocator_warmup"):
            mu.caching_allocator_warmup = lambda *a, **k: None


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _TimingStreamer:
    """Records the wall time of each streamer.put(). generate() puts the prompt first, then one
    call per generated token, so times[1] is the first generated token."""

    def __init__(self):
        self.times: list[float] = []

    def put(self, value):
        self.times.append(time.perf_counter())

    def end(self):
        pass


class HFBackend(VLMBackend):
    name = "hf"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import torch

        self.torch = torch
        self.family = cfg.get("family", "auto")
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, cfg.get("dtype", "float16"))
        _patch_sdpa_gqa()
        _disable_allocator_warmup()
        # Load straight onto the device: on Jetson CPU and GPU share memory, so a CPU copy followed by
        # .to("cuda") briefly needs twice the weights and OOMs the 8 GB Orin Nano.
        load_kwargs = dict(cfg.get("load_kwargs") or {})
        if cfg.get("attn_implementation"):
            load_kwargs["attn_implementation"] = cfg["attn_implementation"]
        t0 = time.perf_counter()
        if self.family == "internvl":
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(
                cfg["model_id"], torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True,
                use_flash_attn=False, device_map=self.device, **load_kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True, use_fast=False)
        elif self.family == "auto":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(cfg["model_id"], **(cfg.get("processor_kwargs") or {}))
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg["model_id"], torch_dtype=self.dtype, low_cpu_mem_usage=True, device_map=self.device,
                **load_kwargs).eval()
        else:
            raise ValueError(f"unknown hf family {self.family!r}")
        _sync()
        self.load_s = time.perf_counter() - t0

    # ----- InternVL (remote code) preprocessing: reference implementation from the model card -----

    def _internvl_pixels(self, image: Image.Image):
        import numpy as np

        size = int(self.cfg.get("image_size", 448))
        tiles = _dynamic_preprocess(image, max_num=int(self.cfg.get("max_tiles", 12)), image_size=size,
                                    use_thumbnail=bool(self.cfg.get("use_thumbnail", True)))
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arrs = [((np.asarray(t.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0 - mean) / std)
                .transpose(2, 0, 1) for t in tiles]
        return self.torch.from_numpy(np.stack(arrs)).to(self.device, self.dtype), len(tiles)

    def generate(self, images, prompt, gen: GenConfig) -> Result:
        torch = self.torch
        gen_kwargs = {"max_new_tokens": gen.max_new_tokens, "do_sample": gen.temperature > 0}
        if gen.temperature > 0:
            gen_kwargs["temperature"] = gen.temperature
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        streamer = _TimingStreamer()
        meta: dict = {"input_size": images[0].size if images else None}
        t0 = time.perf_counter()

        with torch.inference_mode():
            if self.family == "internvl":
                pix, n = zip(*(self._internvl_pixels(im) for im in images))
                pixel_values = torch.cat(pix)
                meta["n_tiles"] = list(n)
                question = "".join("<image>\n" for _ in images) + prompt
                kw = {"num_patches_list": list(n)} if len(images) > 1 else {}
                _sync()
                t1 = time.perf_counter()
                text = self.model.chat(self.tokenizer, pixel_values, question,
                                       dict(gen_kwargs, streamer=streamer), **kw)
                _sync()
                t2 = time.perf_counter()
                n_out = max(0, len(streamer.times) - 1)
                n_in = None
            else:
                messages = [{"role": "user", "content": [{"type": "image"} for _ in images]
                             + [{"type": "text", "text": prompt}]}]
                chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=[chat], images=images, return_tensors="pt",
                                        **(self.cfg.get("processor_call_kwargs") or {}))
                inputs = inputs.to(self.device, dtype=self.dtype)
                if "image_grid_thw" in inputs:  # Qwen2/2.5-VL: bbox coords are in resized-input pixels
                    patch = getattr(self.processor.image_processor, "patch_size", 14)
                    _t, gh, gw = (int(x) for x in inputs["image_grid_thw"][0])
                    meta["input_size"] = (gw * patch, gh * patch)
                if "pixel_values" in inputs:
                    meta["pixel_values_shape"] = list(inputs["pixel_values"].shape)
                n_in = int(inputs["input_ids"].shape[1])
                _sync()
                t1 = time.perf_counter()
                out = self.model.generate(**inputs, **gen_kwargs, streamer=streamer)
                _sync()
                t2 = time.perf_counter()
                new = out[0, n_in:]
                n_out = int(new.shape[0])
                text = self.processor.decode(new, skip_special_tokens=True)

        first = streamer.times[1] if len(streamer.times) > 1 else t2
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        return Result(text=text.strip(), total_ms=(t2 - t0) * 1000, preprocess_ms=(t1 - t0) * 1000,
                      ttft_ms=(first - t1) * 1000, decode_ms=(t2 - first) * 1000,
                      n_in_tokens=n_in, n_out_tokens=n_out, peak_mem_mb=peak, meta=meta)

    def describe(self):
        d = super().describe()
        d["load_s"] = round(self.load_s, 1)
        d["n_params_m"] = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)
        return d

    def close(self):
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """InternVL dynamic-resolution tiling: pick the grid (i x j <= max_num) closest to the aspect ratio."""
    w, h = image.size
    aspect = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda r: r[0] * r[1])
    best, best_diff, area = (1, 1), float("inf"), w * h
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff, best = diff, r
    tw, th = image_size * best[0], image_size * best[1]
    resized = image.resize((tw, th))
    tiles = [resized.crop(((i % best[0]) * image_size, (i // best[0]) * image_size,
                           (i % best[0] + 1) * image_size, (i // best[0] + 1) * image_size))
             for i in range(best[0] * best[1])]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles
