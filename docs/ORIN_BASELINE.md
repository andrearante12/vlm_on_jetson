# Orin Nano baseline (reference for the Thor comparison)

Raw metrics, configs and per-run reports: `baselines/orin_nano/`. The comparison table is
`baselines/orin_nano/compare.md`.

## Setup
- **Device:** Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB shared memory),
  L4T R36.4.7, CUDA 12.6, torch 2.5.0a0+nv24.08, transformers 4.57.6. Power mode **25W** (not MAXN).
- **Model:** InternVL3‑1B (`OpenGVLab/InternVL3-1B-hf`, 938 M params), fp16, in-process transformers.
- **Video:** `DJI_0168.MP4`, 1920×1080 at 29.97 fps, 115 s. Top-down drone approach to an H landing pad,
  with people, a dog and a parked car moving nearby.
- **Runs:** offline at 1 fps (114 frames) and a real-time replay with a 15 fps camera on a wall clock.
  The latency microbenchmark also covers input sides 448/896/1280 px.
- **Ground truth:** none yet, so accuracy is not scored. Quality notes below come from inspecting frames.

## Results

| Output mode | Valid outputs | Latency p50 / p90 | First token p50 | Decode | Real-time rate | Frames dropped | Board power |
|---|---|---|---|---|---|---|---|
| Structured JSON | 114/114 | 4.4 s / 16.1 s | 0.67 s | 10.3 tok/s | 0.14 Hz | 99.1 % | 7.8 W |
| Free text + verdict | 114/114 (verdict line in 24) | 5.7 s / 7.1 s | 0.60 s | 10.6 tok/s | 0.18 Hz | 98.8 % | 7.9 W |
| Bounding boxes | 114/114 | 3.8 s / 4.0 s | 0.60 s | 10.7 tok/s | 0.27 Hz | 98.2 % | 8.3 W |

Peak GPU memory was 1.9 GB and the model took about 30 s to load.

## Findings
1. **Decode dominates latency.** The first token takes about 0.65 s; the rest is generating the answer at
   about 10 tok/s, so a 160-token JSON answer takes 16 s. The hardware ceiling for a 1.9 GB fp16 model at
   102 GB/s is about 50 tok/s, so most of the decode time is software overhead: Python-driven HF
   `generate` on a weak CPU, plus the 25W mode and concurrent load.
2. **Input resolution does not change the cost for InternVL.** There were 951 input tokens at 448, 896 and
   1280 px, because the processor tiles the image into fixed 448 px crops. Tile count (`max_patches`) is
   the visual-token knob, the analogue of LiteVLM's patch selection.
3. **Real time is not viable on this board with this stack.** Each decision is based on a frame 6–16 s old,
   and over 98 % of camera frames are never seen.
4. **Quality is weak** (qualitative, no GT):
   - The model calls 105 of 114 frames safe, including a frame with a person and dog standing on the pad.
   - In some frames it fills every grid cell with obstacles, copying the option lists in the prompt.
   - The free-text mode notices the people, dog and car more often, but still concludes "no hazards" and
     usually omits the verdict line.
5. **The prompt definition matters.** "Area at the centre of the image" does not match footage where the
   pad is off-centre; task prompts should target the pad or the area below the drone.

## Expected on Thor (to be measured)
- **Same model and stack:** a 2–4× speed-up from bandwidth (~273 GB/s vs 102 GB/s) and a much faster CPU,
  which cuts per-token overhead.
- **Optimised serving (vLLM / TensorRT‑LLM, FP8) with compact outputs:** 0.1–0.3 s per decision.
  For scale, LiteVLM reports 530 ms (FP16) and 163 ms (optimised, FP8) for a 2B InternVL on DRIVE Thor.
- **Bigger impact on accuracy:** 7–8B models (Qwen2.5‑VL‑7B, InternVL3‑8B) fit and should fix much of the
  hazard-detection weakness.
