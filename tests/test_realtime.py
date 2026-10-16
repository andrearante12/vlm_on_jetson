import io

import pytest

from trails.backends.fake import FakeBackend
from trails.config import DEFAULTS, deep_merge
from trails.runners.realtime import run_realtime, time_aligned_pairs
from trails.tasks import get_task
from trails.video import FrameCache

TASK = get_task({"type": "landing_json", "prompt": "x"})


def make_cfg(**rt):
    return deep_merge(DEFAULTS, {"model": {"backend": "fake"}, "realtime": rt})


def test_time_aligned_pairs_hold_and_ttl():
    decisions = [{"emit_ms": 300, "capture_ms": 0, "pred": "A"}, {"emit_ms": 900, "capture_ms": 300, "pred": "B"}]
    gt = {0: {"safe_to_land": True}, 5: {"safe_to_land": True}, 10: {"safe_to_land": True}, 30: {"safe_to_land": True}}
    t_ms = lambda i: i * 100.0  # noqa: E731
    pairs, stale = time_aligned_pairs(decisions, gt, t_ms, TASK, end_ms=5000)
    assert [p for p, _ in pairs] == [None, "A", "B", "B"]   # t=0: nothing yet; t=500: A; t=1000, 3000: B
    assert stale == [500, 700, 2700]
    pairs, _ = time_aligned_pairs(decisions, gt, t_ms, TASK, end_ms=5000, ttl_ms=1000)
    assert [p for p, _ in pairs] == [None, "A", "B", None]  # B expired by t=3000
    pairs, _ = time_aligned_pairs(decisions, gt, t_ms, TASK, end_ms=1500)
    assert len(pairs) == 3                                     # GT after replay end is not scored


def test_simulated_clock_drop_rate_and_staleness(synthetic_video, tmp_path):
    """Fake 300 ms model on a 10 fps camera: one decision per 300 ms of video -> ~1/3 of frames
    processed, each decision emitted 300 ms after its capture time."""
    cache = FrameCache(synthetic_video, tmp_path, max_side=None)
    backend = FakeBackend({"latency_ms": 300, "sleep": False})
    cfg = make_cfg(clock="simulated", camera_fps=10, duration_s=6)
    gt = {i: {"safe_to_land": True, "obstacles": []} for i in range(0, 60, 10)}
    pairs, timings, s = run_realtime(backend, TASK, cache, gt, cfg, io.StringIO())
    # Fetch time is real (tiny) so allow a little slack around the ideal 20 decisions in 6 s.
    assert 18 <= s["n_decisions"] <= 20
    assert s["frame_drop_rate"] == pytest.approx(1 - s["n_decisions"] / 60, abs=1e-3)
    assert s["capture_to_decision_ms_p50"] == pytest.approx(300, abs=110)  # + up to 1 frame period of wait
    assert len(pairs) == 6 and pairs[0][0] is None  # GT at t=0 precedes the first decision
    assert all(p is not None for p, _ in pairs[1:])


def test_latency_scale_projects_faster_hardware(synthetic_video, tmp_path):
    cache = FrameCache(synthetic_video, tmp_path, max_side=None)
    base = run_realtime(FakeBackend({"latency_ms": 400, "sleep": False}), TASK, cache, {},
                        make_cfg(clock="simulated", camera_fps=10, duration_s=4), io.StringIO())[2]
    fast = run_realtime(FakeBackend({"latency_ms": 400, "sleep": False}), TASK, cache, {},
                        make_cfg(clock="simulated", camera_fps=10, duration_s=4, latency_scale=0.25), io.StringIO())[2]
    assert fast["n_decisions"] >= 3.5 * base["n_decisions"]


def test_wall_clock_short_run(synthetic_video, tmp_path):
    cache = FrameCache(synthetic_video, tmp_path, max_side=None)
    s = run_realtime(FakeBackend({"latency_ms": 200}), TASK, cache, {},
                     make_cfg(clock="wall", camera_fps=10, duration_s=2), io.StringIO())[2]
    assert 8 <= s["n_decisions"] <= 11
