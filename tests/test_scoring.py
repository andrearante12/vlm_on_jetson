import pytest

from trails.tasks import get_task
from trails.tasks.caption import bleu4, rouge_l
from trails.tasks.grounding import iou

LJ = get_task({"type": "landing_json", "prompt": "x"})
GR = get_task({"type": "grounding", "prompt": "x"})
CAP = get_task({"type": "caption", "prompt": "x"})


def gt(safe, obstacles=(), caption=None):
    return {"safe_to_land": safe, "surface": "grass", "caption": caption,
            "obstacles": [{"type": t, "grid_cell": c, "bbox": b} for t, c, b in obstacles]}


def test_safety_metrics_known_answer():
    pairs = [
        ({"safe_to_land": True, "surface": "grass", "obstacles": []}, gt(True)),            # TN
        ({"safe_to_land": False, "surface": "grass", "obstacles": []}, gt(False)),          # TP
        ({"safe_to_land": True, "surface": "dirt", "obstacles": []}, gt(False)),            # FN, false safe
        (None, gt(False)),                                                                  # no decision -> FN
        ({"safe_to_land": False, "surface": "grass", "obstacles": []}, gt(True)),           # FP
    ]
    m = LJ.score(pairs)
    s = m["safety"]
    assert s["n"] == 5
    assert s["accuracy"] == pytest.approx(2 / 5, abs=1e-4)
    assert s["unsafe_precision"] == pytest.approx(1 / 2, abs=1e-4)
    assert s["unsafe_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["false_safe_rate"] == pytest.approx(1 / 3, abs=1e-4)  # the no-decision frame is not a "safe" claim
    assert m["no_decision_rate"] == pytest.approx(0.2)
    assert m["surface_accuracy"] == pytest.approx(3 / 5, abs=1e-4)  # None pred and the dirt pred are wrong


def test_obstacle_prf_multiset_and_cells():
    p = {"safe_to_land": False, "obstacles": [{"type": "tree", "grid_cell": "center"},
                                              {"type": "tree", "grid_cell": "top-left"},
                                              {"type": "person", "grid_cell": "center"}]}
    g = gt(False, [("tree", "center", None), ("vehicle", "top-left", None)])
    m = LJ.score([(p, g)])
    assert m["obstacle_type"]["tp"] == 1 and m["obstacle_type"]["fp"] == 2 and m["obstacle_type"]["fn"] == 1
    assert m["obstacle_type_and_cell"]["tp"] == 1
    assert m["obstacle_recall_per_class"] == {"tree": 1.0, "vehicle": 0.0}


def test_unlabelled_gt_skipped():
    m = LJ.score([({"safe_to_land": True}, {"safe_to_land": None, "obstacles": None})])
    assert m["n_frames"] == 0


def test_grounding_iou_matching():
    g = gt(False, [("tree", None, [0.0, 0.0, 0.5, 0.5]), ("person", None, [0.6, 0.6, 0.8, 0.8])])
    p = {"obstacles": [{"type": "tree", "bbox": [0.0, 0.0, 0.5, 0.45]},    # IoU 0.9, right class
                       {"type": "vehicle", "bbox": [0.6, 0.6, 0.8, 0.8]},  # right box, wrong class
                       {"type": "rock", "bbox": [0.9, 0.9, 1.0, 1.0]}]}    # spurious
    m = GR.score([(p, g)])
    assert m["class_aware"]["tp"] == 1 and m["class_aware"]["fp"] == 2 and m["class_aware"]["fn"] == 1
    assert m["class_agnostic"]["tp"] == 2 and m["class_agnostic"]["fp"] == 1
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]) == 0.0


def test_text_metrics_bounds():
    assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)
    assert rouge_l("x y", "a b") == 0.0
    assert bleu4("the field is clear of obstacles", "the field is clear of obstacles") == pytest.approx(1.0)
    assert 0 < bleu4("the field is clear", "a field that is mostly clear") < 1


def test_caption_hazard_mentions():
    p, _ = CAP.parse("Two people stand near a tree. VERDICT: UNSAFE")
    m = CAP.score([(p, gt(False, [("person", None, None), ("wire", None, None)], caption="People and wires."))])
    assert m["hazard_mention"]["tp"] == 1 and m["hazard_mention"]["fn"] == 1 and m["hazard_mention"]["fp"] == 1
    assert m["safety"]["accuracy"] == 1.0
