import json

import cv2

from trails.config import load_experiment
from trails.groundtruth import load_gt, merge_records
from trails.groundtruth.importers import import_coco, import_csv, import_cvat
from trails.groundtruth.teacher import parse_teacher
from trails.telemetry import parse_tegrastats_line
from trails.video import FrameCache, indices_for_fps, probe

TEGRA = ("10-02-2026 00:36:39 RAM 5898/7620MB (lfb 5x2MB) SWAP 3772/3810MB (cached 2MB) "
         "CPU [95%@1344,86%@1344,100%@1344,98%@1344,88%@1344,90%@1344] GR3D_FREQ 37% cpu@52.843C "
         "soc2@50.156C gpu@51.187C tj@52.843C VDD_IN 7210mW/7210mW VDD_CPU_GPU_CV 2408mW/2408mW VDD_SOC 1661mW/1661mW")


def test_tegrastats_parse():
    r = parse_tegrastats_line(TEGRA)
    assert r["ram_used_mb"] == 5898 and r["ram_total_mb"] == 7620
    assert r["gpu_util_pct"] == 37 and r["power_total_mw"] == 7210
    assert r["temp_tj_c"] == 52.843 and r["cpu_util_pct"] == 92.8


def test_probe_and_cache(synthetic_video, tmp_path):
    info = probe(synthetic_video)
    assert info.n_frames == 100 and round(info.fps) == 10
    assert indices_for_fps(info, 1.0) == list(range(0, 100, 10))
    cache = FrameCache(synthetic_video, tmp_path, max_side=160)
    cache.ensure([0, 37, 99])
    assert cache.load(37).size == (160, 120)
    # Seeked frame must really be frame 37: compare with a sequential decode (codec shifts intensities,
    # but neighbouring frames differ by ~2 levels per frame, so an off-by-one seek would show).
    cap = cv2.VideoCapture(synthetic_video)
    for _ in range(38):
        ok, ref = cap.read()
    ref_px = int(cv2.resize(ref, (160, 120), interpolation=cv2.INTER_AREA)[2, 75, 2])
    px = cache.load(37).getpixel((75, 2))
    assert abs(px[0] - ref_px) <= 1
    again = FrameCache(synthetic_video, tmp_path, max_side=160)  # index persisted on disk
    assert set(again.index) == {0, 37, 99}
    assert cache.latest_before(3750, [0, 10, 20, 30, 40]) == 30


def test_import_csv(synthetic_video, tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("frame,safe_to_land,surface,obstacle_type,x1,y1,x2,y2\n"
                 "10,false,lawn,person,32,24,64,48\n10,false,lawn,tree,,,,\n20,true,lawn,,,,,\n")
    recs = {r["idx"]: r for r in import_csv(str(p), synthetic_video)}
    assert recs[10]["safe_to_land"] is False and recs[10]["surface"] == "grass"
    assert recs[10]["obstacles"][0] == {"type": "person", "bbox": [0.1, 0.1, 0.2, 0.2], "grid_cell": "top-left"}
    assert recs[10]["obstacles"][1]["bbox"] is None
    assert recs[20]["obstacles"] == [] and recs[20]["safe_to_land"] is True


def test_import_coco(synthetic_video, tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "frame_000030.jpg", "width": 320, "height": 240, "safe_to_land": False},
                   {"id": 2, "file_name": "frame_000040.jpg", "width": 320, "height": 240}],
        "annotations": [{"image_id": 1, "category_id": 3, "bbox": [160, 120, 32, 24]}],
        "categories": [{"id": 3, "name": "car"}]}))
    recs = {r["idx"]: r for r in import_coco(str(p), synthetic_video)}
    assert recs[30]["obstacles"][0]["type"] == "vehicle"
    assert recs[30]["obstacles"][0]["bbox"] == [0.5, 0.5, 0.6, 0.6]
    assert recs[40]["obstacles"] == []


def test_import_cvat(synthetic_video, tmp_path):
    p = tmp_path / "a.xml"
    p.write_text("""<annotations><meta><task><original_size><width>320</width><height>240</height></original_size></task></meta>
      <track id="0" label="tree"><box frame="5" xtl="0" ytl="0" xbr="32" ybr="24" outside="0"/>
                                 <box frame="6" xtl="0" ytl="0" xbr="32" ybr="24" outside="1"/></track>
      <tag frame="5" label="unsafe"/></annotations>""")
    recs = {r["idx"]: r for r in import_cvat(str(p), synthetic_video, assume_complete=True, fps=1.0)}
    assert recs[5]["safe_to_land"] is False and recs[5]["obstacles"][0]["type"] == "tree"
    assert 6 not in recs  # outside box is ignored and frame 6 is not on the 1 fps grid
    assert recs[10]["obstacles"] == []  # assume_complete fills sampled frames


def test_gt_store_merge_and_priority(synthetic_video, tmp_path):
    base = {"video": "synthetic", "t_ms": 0, "surface": None, "obstacles": [], "caption": None}
    merge_records(tmp_path, synthetic_video, [dict(base, idx=0, safe_to_land=True, provenance="teacher:x")])
    merge_records(tmp_path, synthetic_video, [dict(base, idx=0, safe_to_land=False, provenance="annotation:csv")])
    assert load_gt(tmp_path, synthetic_video)[0]["safe_to_land"] is False  # human beats teacher
    assert load_gt(tmp_path, synthetic_video, "teacher")[0]["safe_to_land"] is True


def test_parse_teacher_structured():
    t = parse_teacher('{"safe_to_land": false, "surface": "grass", "caption": "x",'
                      ' "obstacles": [{"type": "tree", "x1": 0, "y1": 0, "x2": 300, "y2": 300}]}')
    assert t["obstacles"][0] == {"type": "tree", "bbox": [0.0, 0.0, 0.3, 0.3], "grid_cell": "top-left"}


def test_experiment_sweep_resolution():
    runs = load_experiment("configs/experiments/baseline.yaml", ["realtime.camera_fps=5"], platform="orin_nano")
    assert [r["task"]["type"] for r in runs] == ["landing_json", "caption", "grounding"]
    assert runs[1]["gen"]["max_new_tokens"] == 160 and runs[0]["gen"]["max_new_tokens"] == 256
    assert all(r["realtime"]["camera_fps"] == 5 for r in runs)
    assert runs[0]["model"]["dtype"] == "float16" and runs[0]["model"]["model_id"].startswith("OpenGVLab/")
    runs = load_experiment("configs/experiments/model_compare.yaml", platform="thor")
    assert [r["model"]["name"] for r in runs] == ["internvl3_1b", "internvl2_5_1b", "qwen2_5_vl_3b_gguf"]
    assert runs[0]["model"]["dtype"] == "bfloat16"
