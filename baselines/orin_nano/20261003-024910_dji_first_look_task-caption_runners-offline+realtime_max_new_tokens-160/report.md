# dji_first_look — task-caption_runners-offline+realtime_max_new_tokens-160

- **Model:** internvl3_1b (`OpenGVLab/InternVL3-1B-hf`), backend hf
- **Task:** caption (caption)
- **Platform:** orin_nano — NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super — NV Power Mode: 25W
- **Videos:** DJI_0168.MP4
- **Model load:** 30.5 s

## Latency

| | offline | realtime |
|---|---|---|
| total_ms_p50 | 5666.7 | 5747.3 |
| total_ms_p90 | 7086.1 | 6591.0 |
| ttft_ms_p50 | 602.2 | 600.9 |
| decode_ms_p50 | 5015.1 | 5089.5 |
| preprocess_ms_p50 | 45.1 | 48.6 |
| tok_per_s_p50 | 10.6 | 10.6 |
| in_tokens_mean | 872.0 | 872.0 |
| out_tokens_mean | 54.7 | 54.0 |
| peak_gpu_mem_mb | 1891.8 | 1891.8 |
| max_decision_hz | 0.174 | 0.176 |

## Real-time replay

Clock: wall (latency scale 1.0)

| | value |
|---|---|
| effective_decision_hz | 0.183 |
| frame_drop_rate | 0.988 |
| capture_to_decision_ms_p50 | 5801.4 |
| capture_to_decision_ms_p90 | 6638.7 |
| staleness_at_gt_ms_p50 | – |
| staleness_at_gt_ms_p90 | – |
| staleness_at_gt_ms_max | – |
| n_decisions | 21 |

## Power / telemetry

| | offline | realtime |
|---|---|---|
| power_total_mw_mean | 7876.0 | 7874.1 |
| power_total_mw_max | 13865 | 13826 |
| energy_per_call_j | 45.2 | 44.6 |
| gpu_util_pct_mean | 56.0 | 56.7 |
| ram_used_mb_max | 6696 | 6662 |
| temp_tj_c_max | 54.0 | 53.8 |
| telemetry_source | tegrastats | tegrastats |
