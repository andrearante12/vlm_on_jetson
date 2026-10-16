# Moving the harness to Jetson AGX Thor

The code is platform-agnostic: the same configs and commands run on Orin and Thor. What changes is
the container base image, the platform defaults (`configs/platforms/thor.yaml`: bf16, 128 GB), and
which models fit. This guide gets a fresh Thor to the point where it reproduces the Orin baseline and
then goes beyond it.

## 0. What moves how

| What | How | Notes |
|---|---|---|
| Code, configs, docs, Orin baseline metrics | `git clone` | everything in this repo |
| Videos (`data/videos/`) | `scripts/transfer_data.sh` (rsync) | kept out of git: large, and footage may be private |
| Ground truth (`data/gt/`), raw annotations | `scripts/transfer_data.sh` | small; kept out of git by default, add if you want them versioned |
| Past run pages (`results/`) | `WITH_RESULTS=1 scripts/transfer_data.sh` | optional; `view.html` embeds frames |
| Frame cache (`data/frames/`) | not moved | rebuilt automatically |
| Model weights | not moved | downloaded on first use into `~/.cache/huggingface` |

## 1. Prepare the Thor (host)

```bash
# JetPack 7 should already be flashed. Check device, L4T, power mode, docker:
git clone git@github.com:andrearante12/vlm_on_jetson.git ~/vlm_on_jetson
cd ~/vlm_on_jetson
scripts/check_platform.sh

# Benchmark settings (check the mode ids first: nvpmodel -q --verbose)
sudo nvpmodel -m 0          # MAXN on Thor
sudo jetson_clocks

# Docker must be able to use the GPU
sudo apt install -y nvidia-container-toolkit   # usually preinstalled with JetPack
docker run --rm --runtime nvidia nvcr.io/nvidia/pytorch:25.08-py3 \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Bring the data over (run on the Orin)

```bash
cd ~/nsf_trails            # or wherever the harness lives on the Orin
scripts/transfer_data.sh andre@<thor-ip> ~/vlm_on_jetson
```

## 3. Build and smoke-test (on the Thor)

```bash
docker compose -f docker/compose.yaml --profile thor build
docker/run.sh info                                     # should report detected_platform: thor
docker/run.sh run configs/experiments/smoke.yaml       # fake model, ~1 min, no weights needed
```

**Base image.** `docker/Dockerfile.thor` uses `nvcr.io/nvidia/pytorch:25.08-py3`, the arm64 SBSA image. Its
CUDA must be supported by the installed JetPack 7 driver. If `torch.cuda.is_available()` is False in
the container, use a newer release (25.09 and 25.10 are verified available for arm64):
`docker compose -f docker/compose.yaml --profile thor build --build-arg BASE=nvcr.io/nvidia/pytorch:25.10-py3`.

**First-run checks** (things that could not be verified before the hardware arrived):
- `results/<run>/telemetry.csv`: check that `power_total_mw` is filled. Thor's tegrastats rail names may
  differ from Orin's `VDD_IN`. If none of the known names match, the parser sums every rail it
  finds, which can double-count. Fix the name list in `trails/telemetry.py` (`parse_tegrastats_line`).
- `trails/backends/hf.py` contains two Jetson workarounds:
  - a fallback for a missing SDPA `enable_gqa` argument in pre-release torch builds;
  - skipping the transformers allocator warm-up on integrated GPUs.

  Both turn themselves off when they aren't needed, but if you see an odd load or attention error,
  look there first.

## 4. Reproduce the Orin baseline on Thor

```bash
docker/run.sh run configs/experiments/thor_first_look.yaml
docker/run.sh report baselines/orin_nano results/ --out results/compare_orin_vs_thor
docker/run.sh serve          # viewer on http://<thor-ip>:8800/
```
`thor_first_look.yaml` is the Orin `dji_first_look` experiment unchanged except for the platform. The
model runs in bf16 on Thor and fp16 on Orin. The comparison table puts the two platforms side by side.

## 5. Go beyond what fits on the Orin

```bash
# Larger models in plain transformers (no server needed)
docker/run.sh run configs/experiments/thor_model_compare.yaml

# Optimised serving: start vLLM, then the *_vllm model config talks to it
docker compose -f docker/compose.yaml --profile vllm up -d vllm-thor
docker compose -f docker/compose.yaml logs -f vllm-thor   # wait for "Application startup complete"
docker/run.sh run configs/experiments/thor_vllm.yaml
```
Comparing `qwen2_5_vl_7b` (transformers) with `qwen2_5_vl_7b_vllm` (vLLM) on the same video measures
how much of the latency comes from the software stack. On the Orin that was the dominant cost (see
`docs/ORIN_BASELINE.md`). FP8 and TensorRT‑LLM (`nvcr.io/nvidia/tensorrt-llm/release`) plug in the same
way: serve the model and add an `openai`-backend model config pointing at the server.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False in container | base image CUDA newer than the driver supports, or the container wasn't started with the nvidia runtime | newer/older `BASE` tag; use `docker/run.sh` (sets `runtime: nvidia`) |
| `NVML_SUCCESS == r INTERNAL ASSERT` while loading | out of GPU memory (Jetson reports OOM this way) | free memory; drop caches: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `Missing **kwargs ... LlamaModel.forward` | torch pre-release without SDPA `enable_gqa` | handled by `_patch_sdpa_gqa` in `hf.py`; update transformers/torch |
| No power numbers | tegrastats not mounted or rail names differ | see first-run checks above |
| Viewer not reachable from other devices | firewall or wrong network | `trails serve` binds 0.0.0.0:8800; check `ufw status` and the IP |
