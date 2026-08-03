from __future__ import annotations

import pytest
from dataclasses import replace

from gecko_vision_gate.gme_contracts import (
    ArtifactIdentity,
    Detection,
    GMEAnalysis,
    GMEConfig,
    StateInterval,
    TrackingQuality,
    TrackPoint,
)


def test_detection_rejects_invalid_values():
    with pytest.raises(ValueError):
        Detection(0.0, (-1.0, 2.0, 3.0, 4.0), 0.8, "gecko")
    with pytest.raises(ValueError):
        Detection(0.0, (1.0, 2.0, 3.0, 4.0), 1.1, "gecko")


def test_track_point_requires_normalized_bbox_and_known_provenance():
    p = TrackPoint("g1", 1.5, (0.1, 0.2, 0.3, 0.4), 0.9, "observed")
    assert p.bbox_norm == (0.1, 0.2, 0.3, 0.4)
    with pytest.raises(ValueError):
        TrackPoint("g1", 1.5, (0.9, 0.2, 0.3, 0.4), 0.9, "tracked")
    with pytest.raises(ValueError):
        TrackPoint("g1", 1.5, (0.1, 0.2, 0.3, 0.4), 0.9, "guessed")


def test_state_intervals_are_half_open_and_non_overlapping():
    first = StateInterval(0.0, 1.0, "static", ("g1",))
    second = StateInterval(1.0, 2.0, "moving", ("g1",))
    analysis = GMEAnalysis.minimal(
        duration_sec=2.0,
        intervals=(first, second),
        tracking_quality=TrackingQuality.empty(),
        artifact_identity=ArtifactIdentity.test_identity(),
    )
    assert analysis.candidate_moving_sec_any_gecko == 1.0
    with pytest.raises(ValueError):
        GMEAnalysis.minimal(
            duration_sec=2.0,
            intervals=(first, StateInterval(0.5, 1.5, "moving", ("g1",))),
            tracking_quality=TrackingQuality.empty(),
            artifact_identity=ArtifactIdentity.test_identity(),
        )


def test_config_rejects_unbounded_or_negative_settings():
    assert GMEConfig().analysis_fps == 30.0
    with pytest.raises(ValueError):
        GMEConfig(analysis_fps=31.0)
    with pytest.raises(ValueError):
        GMEConfig(max_interpolation_gap_sec=-1.0)


def test_analysis_rejects_impossible_time_accounting():
    base = GMEAnalysis.minimal(
        duration_sec=1.0,
        intervals=(StateInterval(0.0, 1.0, "static", ("g1",)),),
        tracking_quality=TrackingQuality.empty(),
        artifact_identity=ArtifactIdentity.test_identity(),
    )
    with pytest.raises(ValueError):
        replace(base, candidate_moving_sec_any_gecko=2.0)
    with pytest.raises(ValueError):
        replace(base, unknown_sec=1.0)
