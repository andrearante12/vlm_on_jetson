# dji_first_look — task-landing_json

- **Model:** internvl3_1b (`OpenGVLab/InternVL3-1B-hf`), backend hf
- **Task:** landing_json (landing_json)
- **Platform:** orin_nano — NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super — NV Power Mode: 25W
- **Videos:** DJI_0168.MP4
- **Model load:** 34.5 s

## Latency

| | offline | realtime | latency |
|---|---|---|---|
| total_ms_p50 | 4363.6 | 5741.8 | 16176.8 |
| total_ms_p90 | 16059.5 | 15832.4 | 16335.3 |
| ttft_ms_p50 | 627.8 | 622.6 | 669.7 |
| decode_ms_p50 | 3686.8 | 5069.8 | 15467.7 |
| preprocess_ms_p50 | 45.7 | 48.1 | 35.5 |
| tok_per_s_p50 | 10.5 | 10.6 | 10.3 |
| in_tokens_mean | 951.0 | 951.0 | 951.0 |
| out_tokens_mean | 62.4 | 68.9 | 141.2 |
| peak_gpu_mem_mb | 1892.1 | 1892.1 | 1892.1 |
| max_decision_hz | 0.153 | 0.141 | 0.070 |

### Resolution sweep (DJI_0168)

| max side | p50 ms | ttft p50 | in tokens |
|---|---|---|---|
| 448 | 16119.9 | 669.1 | 951.0 |
| 896 | 16156.5 | 674.2 | 951.0 |
| 1280 | 16235.0 | 668.2 | 951.0 |

## Real-time replay

Clock: wall (latency scale 1.0)

| | value |
|---|---|
| effective_decision_hz | 0.139 |
| frame_drop_rate | 0.991 |
| capture_to_decision_ms_p50 | 5796.6 |
| capture_to_decision_ms_p90 | 15911.1 |
| staleness_at_gt_ms_p50 | – |
| staleness_at_gt_ms_p90 | – |
| staleness_at_gt_ms_max | – |
| n_decisions | 16 |

## Power / telemetry

| | offline | realtime | latency |
|---|---|---|---|
| power_total_mw_mean | 7817.6 | 7224.9 | 7497.8 |
| power_total_mw_max | 13747 | 13943 | 13669 |
| energy_per_call_j | 51.2 | 51.2 | 107.2 |
| gpu_util_pct_mean | 55.3 | 38.3 | 56.2 |
| ram_used_mb_max | 6085 | 5932 | 6030 |
| temp_tj_c_max | 53.8 | 53.3 | 52.7 |
| telemetry_source | tegrastats | tegrastats | tegrastats |
