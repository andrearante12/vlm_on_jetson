# vlm_on_jetson — VLM landing-zone evaluation harness for Jetson

Benchmarks small vision-language models for UAS landing-zone and obstacle assessment on NVIDIA
Jetson. It answers two questions for every model and config:

1. **How accurate is it?** This is the offline score: every evaluation frame is processed with no deadline.
2. **How much of that accuracy survives real time?** The video is replayed as a live camera. The model
   always grabs the newest frame, and each decision is scored at the moment it would actually be in effect.

Everything runs in containers with per-platform configs, so the same experiments run unchanged on
Jetson Orin and Jetson AGX Thor.

- **Moving to or setting up a Thor:** [`docs/THOR_SETUP.md`](docs/THOR_SETUP.md)
- **Orin Nano reference results (InternVL3‑1B, DJI footage):** [`docs/ORIN_BASELINE.md`](docs/ORIN_BASELINE.md),
  raw metrics in `baselines/orin_nano/`

> **About LiteVLM (arXiv 2506.07416):** this is an NVIDIA *inference pipeline*, not a released model.
> It is InternVL2.5‑1B/2B finetuned on DriveLM driving QA, with patch selection, token pruning and Eagle‑2
> speculative decoding, built on custom TensorRT for DRIVE Thor. No code or weights are public.
> This harness therefore benchmarks its **base models** (`internvl2_5_1b`) and newer small VLMs. LiteVLM's
> ideas become experiment variants: fewer image patches (`latency_sweep.yaml`), FP8/INT4 quantization
> and TensorRT‑LLM/vLLM serving on Thor.

## Layout

```
configs/platforms/   orin_nano, thor, orin_agx, generic   (per-platform model defaults)
configs/models/      internvl3_1b, internvl3_2b, internvl2_5_1b, qwen2_5_vl_3b(_gguf), fake, *_teacher
configs/tasks/       landing_json (structured), caption (free text), grounding (boxes) — prompts live here
configs/experiments/ smoke, baseline, latency_sweep, realtime_whatif, model_compare, dji_first_look,
                     thor_first_look, thor_model_compare, thor_vllm
trails/              the package (video cache, backends, tasks, GT, runners, telemetry, reports, viewer)
docker/              Dockerfile.orin, Dockerfile.thor, compose.yaml (+ llama.cpp and vLLM services), run.sh
scripts/             check_platform.sh (pre-run checklist), transfer_data.sh (copy footage/GT to another board)
docs/                THOR_SETUP.md, ORIN_BASELINE.md
baselines/           committed reference metrics (no footage) for cross-platform comparison
tests/               pytest suite (no GPU or weights needed)
data/videos/         your footage            data/gt/      canonical ground truth (jsonl per video)
data/annotations/    raw annotation exports  data/frames/  decoded-frame cache (safe to delete)
results/             one directory per run + comparison tables
```

## Quick start

```bash
# 1. Build the container for this board (docker/run.sh auto-detects Orin vs Thor)
docker compose -f docker/compose.yaml --profile orin build      # or --profile thor
docker/run.sh info                                   # platform snapshot: L4T, power mode, torch/CUDA

# 2. Put videos in data/videos/, then check the pipeline with the fake model (no weights needed)
docker/run.sh run configs/experiments/smoke.yaml

# 3. Get ground truth (either or both, see below), then run the real benchmark
docker/run.sh run configs/experiments/baseline.yaml
docker/run.sh report results/                        # comparison table across all runs
```

Without Docker, any Python ≥3.10 environment with a CUDA-enabled torch for the board works:
`pip install -e ".[hf,teacher,dev]"`, then `trails ...` or `python -m trails ...`. Run the tests with
`pytest`; they need no GPU or weights.

**Before benchmarking**, run `scripts/check_platform.sh`, then on the host set a MAXN power mode
(`sudo nvpmodel -m <id>`: MAXN SUPER is 2 on the Orin Nano Super and MAXN is 0 on Thor; check
`nvpmodel -q --verbose`) and run `sudo jetson_clocks`. Close other GPU and memory users, since the
Orin Nano shares 8 GB between CPU and GPU. Every run records the power mode and tegrastats power,
utilisation and temperature, so runs at different power modes stay distinguishable.

## Viewing results

Every run writes `results/<run>/view.html`. It's a self-contained page with:
- summary tiles
- the real-time decision timeline
- every frame with the model's verdict and obstacles drawn on it
- latency charts and power/GPU charts

It also has a **labelling mode**: open a frame, press `S` (safe) or `U` (unsafe), tag obstacles, then
use *Export my labels (CSV)* and load the file with `trails import-gt`. `results/index.html` lists all runs.

To view from another device on the network:
```bash
trails serve                    # serves results/ on 0.0.0.0:8800 and prints the URLs
# then open http://<jetson-ip>:8800/ on a laptop or phone
trails view results/<run>       # rebuild a run's page (e.g. after adding GT)
```

## Ground truth

All GT is stored as `data/gt/<video_stem>.jsonl`, one record per labelled frame:
`{idx, t_ms, safe_to_land, surface, obstacles: [{type, bbox (0-1 xyxy), grid_cell}], caption, provenance}`.
A `null` field means "not labelled" and is skipped by scorers, not counted as wrong. When a human
annotation and a teacher label exist for the same frame, the human one wins.

**Existing annotations:**
```bash
docker/run.sh import-gt data/annotations/flight1.xml --video data/videos/flight1.mp4          # CVAT video XML
docker/run.sh import-gt data/annotations/flight1.json --video data/videos/flight1.mp4         # COCO
docker/run.sh import-gt data/annotations/labels.csv  --video data/videos/*.mp4                 # CSV
```
- **CSV columns:** `video, frame|t_ms|t_s, safe_to_land, surface, caption, obstacle_type, x1, y1, x2, y2, grid_cell`.
  Use one row per obstacle, or one row with an empty `obstacle_type` for "no obstacles".
- **CVAT:** `<tag label="safe|unsafe">` gives the landing verdict.
- **`--assume-complete`:** treats unannotated sampled frames as obstacle-free.

**Teacher model (pseudo-labels):**
```bash
export ANTHROPIC_API_KEY=...        # Claude (claude-opus-5-5): frames are sent to the Anthropic API
docker/run.sh teacher data/videos/*.mp4 --fps 1 --teacher claude_teacher
# or keep footage on-prem: a big open VLM behind vLLM (e.g. on Thor), see configs/models/local_teacher.yaml
docker/run.sh teacher data/videos/*.mp4 --fps 1 --teacher local_teacher
```
The teacher labels the verdict, surface, obstacles with boxes and a caption in one pass, so a single
labelling run serves all three tasks. Labelling resumes where it stopped. Scores against teacher labels
measure **agreement with the teacher**. Hand-check a sample of the teacher's labels before treating them as
ground truth, and treat its boxes with extra care: Claude's localisation is much weaker than its scene judgement.

## Output modes (tasks)

| task | model outputs | main metrics |
|---|---|---|
| `landing_json` | `{safe_to_land, surface, obstacles:[{type, grid_cell}]}` | safety accuracy, **unsafe recall/F1**, **false-safe rate**, obstacle P/R/F1 (type, type+3×3 cell), parse-fail rate |
| `caption` | free text ending in `VERDICT: SAFE/UNSAFE` | verdict accuracy, hazard-mention P/R/F1, ROUGE-L / BLEU-4 (DriveLM-style) |
| `grounding` | boxes (JSON, Qwen `bbox_2d`, or InternVL `<ref><box>`) | P/R/F1 at IoU≥0.5 (class-aware and class-agnostic), mean IoU |

"Unsafe" is the positive class. A missing decision counts as a failure to warn.

## Real-time evaluation

`realtime` replays each video as a camera at `camera_fps`. While the model is busy, frames are dropped.
At every GT timestamp the decision in effect is the latest one *emitted* before that moment, which may
come from a much older frame. The report shows, side by side:
- offline accuracy (the ceiling), real-time accuracy, and the **gap** between them
- effective decision rate (Hz), frame drop rate, capture→decision latency, and **staleness** at GT time
- latency breakdown (preprocess / TTFT = vision + prefill / decode, tok/s), peak GPU memory, board power, energy per call

`clock: simulated` with `latency_scale: 0.5` asks "what if inference were 2× faster?". `realtime_whatif.yaml`
uses it to project Thor-class speedups before the hardware arrives. `decision_ttl_ms` makes old
decisions expire.

## Moving to Thor

See [`docs/THOR_SETUP.md`](docs/THOR_SETUP.md). In short:
1. Clone the repo and run `scripts/transfer_data.sh` from the Orin.
2. Build with `--profile thor` and run `thor_first_look.yaml`, which repeats the Orin baseline.
3. Compare with `trails report baselines/orin_nano results/`.
4. Try the larger models (`thor_model_compare.yaml`) and vLLM serving (`thor_vllm.yaml`).

## Adding a model

Create `configs/models/<name>.yaml`:
- **In-process transformers:** `backend: hf`, `family: auto` works for any `AutoModelForImageTextToText`
  checkpoint; `family: internvl` is for original InternVL remote-code checkpoints.
- **Any OpenAI-compatible server:** `backend: openai`. This covers llama.cpp, vLLM, TensorRT‑LLM and Ollama.

Model configs can override task prompts under `prompts: {grounding: "..."}`.
