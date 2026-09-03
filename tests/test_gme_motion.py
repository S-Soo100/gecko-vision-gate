from __future__ import annotations

from gecko_vision_gate.gme_contracts import TrackPoint
from gecko_vision_gate import gme_motion
import cv2
import numpy as np

from gecko_vision_gate.gme_motion import (
    aggregate_states,
    classify_track_motion,
    detect_camera_motion,
    detect_exposure_change,
)


def _p(track: str, ts: float, x: float) -> TrackPoint:
    return TrackPoint(track, ts, (x, 0.1, 0.2, 0.2), 0.9, "observed")


def test_track_motion_is_body_length_normalized():
    assert classify_track_motion(_p("g1", 0.0, 0.1), _p("g1", 1.0, 0.3), threshold_body_lengths=0.2) == "moving"
    assert classify_track_motion(_p("g1", 0.0, 0.1), _p("g1", 1.0, 0.105), threshold_body_lengths=0.2) == "static"


def test_slow_motion_uses_centered_three_second_displacement():
    points = tuple(
        _p("g1", index / 10, 0.1 + 0.001 * index)
        for index in range(61)
    )
    frames = tuple(
        (point.timestamp_sec, {point.track_id: "static"}, False)
        for point in points
    )

    promoted = gme_motion.promote_slow_motion(
        frames,
        points,
        window_sec=3.0,
        threshold_body_lengths=0.08,
        max_track_gap_sec=0.2,
    )

    states = {round(timestamp, 1): track_states["g1"] for timestamp, track_states, _ in promoted}
    assert states[3.0] == "moving"
    assert aggregate_states(promoted, duration_sec=6.1).candidate_moving_sec_any_gecko > 0


def test_slow_motion_does_not_promote_stationary_bbox_jitter():
    points = tuple(
        _p("g1", index / 10, 0.1 + (0.003 if index % 2 else -0.003))
        for index in range(61)
    )
    frames = tuple(
        (point.timestamp_sec, {point.track_id: "static"}, False)
        for point in points
    )

    promoted = gme_motion.promote_slow_motion(
        frames,
        points,
        window_sec=3.0,
        threshold_body_lengths=0.08,
        max_track_gap_sec=0.2,
    )

    assert all(track_states["g1"] == "static" for _, track_states, _ in promoted)


def test_two_simultaneous_geckos_union_user_time_but_sum_internal_time():
    frames = (
        (0.0, {"g1": "moving", "g2": "moving"}, False),
        (10.0, {"g1": "moving", "g2": "moving"}, False),
    )
    summary = aggregate_states(frames, duration_sec=10.0)
    assert summary.candidate_moving_sec_any_gecko == 10.0
    assert summary.moving_gecko_seconds == 20.0
    assert summary.intervals[0].state == "moving"


def test_camera_motion_overrides_local_track_state():
    frames = (
        (0.0, {"g1": "moving"}, True),
        (2.0, {"g1": "moving"}, True),
    )
    summary = aggregate_states(frames, duration_sec=2.0)
    assert summary.candidate_moving_sec_any_gecko == 0.0
    assert summary.camera_motion_sec == 2.0
    assert summary.intervals[0].state == "camera_motion"


def test_global_translation_is_camera_motion_but_brightness_jump_is_separate_quality_signal():
    rng = np.random.default_rng(12)
    first = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    matrix = np.float32([[1, 0, 8], [0, 1, 0]])
    shifted = cv2.warpAffine(first, matrix, (160, 120))
    assert detect_camera_motion(first, shifted, threshold_norm=0.01) is True
    brighter = np.clip(first.astype(np.int16) + 80, 0, 255).astype(np.uint8)
    assert detect_exposure_change(first, brighter, threshold=0.15) is True
