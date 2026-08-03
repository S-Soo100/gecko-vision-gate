from __future__ import annotations

import gzip
import json

from gecko_vision_gate.gme_contracts import (
    ArtifactIdentity,
    GMEAnalysis,
    StateInterval,
    TrackingQuality,
    TrackPoint,
)
from gecko_vision_gate.gme_serialization import serialize_artifacts


def _analysis() -> GMEAnalysis:
    point = TrackPoint("g1", 0.0, (0.1, 0.1, 0.2, 0.2), 0.9, "observed")
    return GMEAnalysis.minimal(
        duration_sec=1.0,
        intervals=(StateInterval(0.0, 1.0, "moving", ("g1",)),),
        tracking_quality=TrackingQuality.empty(),
        artifact_identity=ArtifactIdentity.test_identity(),
        track_points=(point,),
    )


def test_serialization_is_deterministic_and_separates_debug():
    first = serialize_artifacts(_analysis())
    second = serialize_artifacts(_analysis())
    assert first.permanent_gzip == second.permanent_gzip
    assert first.permanent_sha256 == second.permanent_sha256
    permanent = json.loads(gzip.decompress(first.permanent_gzip))
    debug = json.loads(gzip.decompress(first.debug_gzip))
    assert permanent["schema_version"] == "gme-artifact-v1"
    assert "track_points" in permanent
    assert "frame_debug" not in permanent
    assert "frame_debug" in debug
