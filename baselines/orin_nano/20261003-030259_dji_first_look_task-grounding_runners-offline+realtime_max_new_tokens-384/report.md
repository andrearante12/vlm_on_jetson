# dji_first_look — task-grounding_runners-offline+realtime_max_new_tokens-384

- **Model:** internvl3_1b (`OpenGVLab/InternVL3-1B-hf`), backend hf
- **Task:** grounding (grounding)
- **Platform:** orin_nano — NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super — NV Power Mode: 25W
- **Videos:** DJI_0168.MP4
- **Model load:** 31.0 s

## Latency

| | offline | realtime |
|---|---|---|
| total_ms_p50 | 3754.5 | 3740.7 |
| total_ms_p90 | 3966.2 | 3931.0 |
| ttft_ms_p50 | 601.7 | 601.2 |
| decode_ms_p50 | 3105.5 | 3091.9 |
| preprocess_ms_p50 | 45.0 | 46.6 |
| tok_per_s_p50 | 10.7 | 10.6 |
| in_tokens_mean | 872.0 | 872.0 |
| out_tokens_mean | 36.2 | 33.8 |
| peak_gpu_mem_mb | 1891.8 | 1891.8 |
| max_decision_hz | 0.252 | 0.267 |

## Real-time replay

Clock: wall (latency scale 1.0)

| | value |
|---|---|
| effective_decision_hz | 0.270 |
| frame_drop_rate | 0.982 |
| capture_to_decision_ms_p50 | 3777.4 |
| capture_to_decision_ms_p90 | 3976.2 |
| staleness_at_gt_ms_p50 | – |
| staleness_at_gt_ms_p90 | – |
| staleness_at_gt_ms_max | – |
| n_decisions | 31 |

## Power / telemetry

| | offline | realtime |
|---|---|---|
| power_total_mw_mean | 8312.9 | 8390.6 |
| power_total_mw_max | 13904 | 13747 |
| energy_per_call_j | 33.0 | 31.4 |
| gpu_util_pct_mean | 55.9 | 55.5 |
| ram_used_mb_max | 6593 | 6478 |
| temp_tj_c_max | 54.5 | 55.0 |
| telemetry_source | tegrastats | tegrastats |
