from __future__ import annotations

import numpy as np
import pytest

from gecko_vision_gate.gme_contracts import Detection, TrackPoint
from gecko_vision_gate.gme_tracker import MultiGeckoTracker, interpolate_short_gaps


def _d(ts: float, x: float, y: float, conf: float = 0.9) -> Detection:
    return Detection(ts, (x, y, 20.0, 20.0), conf, "gecko")


def test_anchor_association_keeps_two_geckos_separate():
    tracker = MultiGeckoTracker()
    first = tracker.update_anchor((_d(0.0, 10, 10), _d(0.0, 70, 70)), (100, 100))
    second = tracker.update_anchor((_d(0.5, 12, 10), _d(0.5, 68, 70)), (100, 100))
    assert [p.track_id for p in first] == ["g0001", "g0002"]
    assert [p.track_id for p in second] == ["g0001", "g0002"]
    assert all(p.provenance == "observed" for p in second)


def test_anchor_tie_is_deterministic_by_detection_position():
    tracker = MultiGeckoTracker()
    points = tracker.update_anchor((_d(0.0, 60, 5), _d(0.0, 5, 5)), (100, 100))
    assert [(p.track_id, p.bbox_norm[0]) for p in points] == [
        ("g0001", 0.05),
        ("g0002", 0.6),
    ]


def test_bbox_crossing_frame_edge_is_clamped_to_normalized_contract():
    tracker = MultiGeckoTracker()
    point = tracker.update_anchor((_d(0.0, 95, 95),), (100, 100))[0]
    assert point.bbox_norm == pytest.approx((0.95, 0.95, 0.05, 0.05))


def test_optical_flow_failure_requests_redetection_instead_of_inventing_point():
    tracker = MultiGeckoTracker(min_flow_points=4)
    frame = np.zeros((100, 100), dtype=np.uint8)
    tracker.update_anchor((_d(0.0, 10, 10),), frame.shape)
    result = tracker.update_tracked(frame, frame, 1 / 30, frame.shape)
    assert result.points == ()
    assert result.needs_redetection is True


def test_only_short_bidirectional_gap_is_interpolated():
    points = (
        TrackPoint("g1", 0.0, (0.1, 0.1, 0.2, 0.2), 0.9, "observed"),
        TrackPoint("g1", 1.0, (0.2, 0.1, 0.2, 0.2), 0.9, "observed"),
        TrackPoint("g2", 0.0, (0.1, 0.1, 0.2, 0.2), 0.9, "observed"),
        TrackPoint("g2", 2.0, (0.2, 0.1, 0.2, 0.2), 0.9, "observed"),
    )
    out = interpolate_short_gaps(points, step_sec=0.5, max_gap_sec=1.0)
    assert any(p.track_id == "g1" and p.timestamp_sec == 0.5 and p.provenance == "interpolated" for p in out)
    assert not any(p.track_id == "g2" and p.provenance == "interpolated" for p in out)
