from trails.schema import Vocab, DEFAULT_OBSTACLE_CLASSES, canon_grid_cell, grid_cell_from_bbox, normalize_bbox
from trails.tasks import get_task
from trails.tasks.base import extract_json

LJ = get_task({"type": "landing_json", "prompt": "x"})
CAP = get_task({"type": "caption", "prompt": "x"})
GR = get_task({"type": "grounding", "prompt": "x"})


def test_extract_json_fenced_with_prose():
    text = 'Sure! Here it is:\n```json\n{"safe_to_land": false, "obstacles": [{"type": "tree"}]}\n```\nHope that helps.'
    assert extract_json(text) == {"safe_to_land": False, "obstacles": [{"type": "tree"}]}


def test_extract_json_lenient_python_literals_and_trailing_comma():
    assert extract_json("{'safe_to_land': True, 'obstacles': [],}") == {"safe_to_land": True, "obstacles": []}


def test_extract_json_braces_inside_strings():
    assert extract_json('{"caption": "a {weird} one", "safe_to_land": "yes"}')["caption"] == "a {weird} one"


def test_landing_json_parse_normalises_labels_and_cells():
    pred, ok = LJ.parse('{"safe_to_land": "no", "surface": "Asphalt", "obstacles": ['
                        '{"type": "Power Lines", "grid_cell": "upper left"}, {"type": "people", "grid_cell": "middle"}]}')
    assert ok
    assert pred["safe_to_land"] is False
    assert pred["surface"] == "pavement"
    assert [(o["type"], o["grid_cell"]) for o in pred["obstacles"]] == [("wire", "top-left"), ("person", "center")]


def test_landing_json_garbage_is_parse_failure():
    pred, ok = LJ.parse("I cannot determine this.")
    assert not ok and pred["safe_to_land"] is None


def test_caption_verdict_and_mentions():
    pred, ok = CAP.parse("A grassy field with two hikers near a tree.\nVERDICT: UNSAFE")
    assert ok and pred["safe_to_land"] is False
    assert set(pred["mentioned"]) == {"person", "tree"}
    assert "VERDICT" not in pred["caption"]


def test_grounding_internvl_box_syntax():
    pred, ok = GR.parse("<ref>tree</ref><box>[[100, 200, 300, 400]]</box> and <ref>car</ref><box>[[0,0,500,500],[600,600,700,700]]</box>")
    assert ok
    assert pred["obstacles"][0] == {"type": "tree", "bbox": [0.1, 0.2, 0.3, 0.4], "grid_cell": None}
    assert [o["type"] for o in pred["obstacles"]] == ["tree", "vehicle", "vehicle"]


def test_grounding_qwen_pixel_boxes():
    pred, ok = GR.parse('```json\n[{"bbox_2d": [56, 28, 112, 84], "label": "person"}]\n```',
                        {"input_size": (560, 280), "bbox_format": "pixels"})
    assert ok
    assert pred["obstacles"][0]["bbox"] == [0.1, 0.1, 0.2, 0.3]


def test_normalize_bbox_auto_and_degenerate():
    assert normalize_bbox([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
    assert normalize_bbox([100, 200, 300, 400]) == [0.1, 0.2, 0.3, 0.4]
    assert normalize_bbox([300, 400, 100, 200]) == [0.1, 0.2, 0.3, 0.4]  # swapped corners
    assert normalize_bbox([5, 5, 5, 9]) is None


def test_grid_helpers():
    assert grid_cell_from_bbox([0.4, 0.4, 0.6, 0.6]) == "center"
    assert grid_cell_from_bbox([0.0, 0.8, 0.1, 1.0]) == "bottom-left"
    assert canon_grid_cell("Lower Right") == "bottom-right"
    assert canon_grid_cell("top") == "top-center"
    assert canon_grid_cell("nowhere") is None


def test_vocab():
    v = Vocab(DEFAULT_OBSTACLE_CLASSES)
    assert v.canon("Trees") == "tree"
    assert v.canon("a parked car") == "vehicle"
    assert v.canon("spaceship") == "other"
    assert v.mentioned("power lines over the road, a dog") == {"wire", "animal"}
