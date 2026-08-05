from __future__ import annotations

from dataclasses import dataclass

from gecko_vision_gate.gme_contracts import (
    ArtifactIdentity,
    GMEAnalysis,
    TrackPoint,
    TrackingQuality,
)
from gecko_vision_gate.gme_presence import (
    analyze_presence,
    analyze_presence_with_gate,
    decide_presence,
)


@dataclass
class FakeDetector:
    model_name: str = "rf-detr-nano"
    model_version: str = "gecko_v2 (checkpoint_best_ema)"
    checkpoint_sha256: str = "a" * 64
    schema_version: str = "detector-v1"
    threshold: float = 0.5

    def detect(self, frame_bgr, timestamp_sec):
        return ()


def _analysis(
    *,
    status: str = "ok",
    analyzed_frame_count: int = 30,
    track_points: tuple[TrackPoint, ...] = (),
    unknown_sec: float = 0.0,
    camera_motion_sec: float = 0.0,
) -> GMEAnalysis:
    return GMEAnalysis(
        status=status,
        duration_sec=1.0 if status == "ok" else 0.0,
        decoded_frame_count=30 if status == "ok" else 0,
        analyzed_frame_count=analyzed_frame_count if status == "ok" else 0,
        source_fps=30.0 if status == "ok" else None,
        intervals=(),
        track_points=track_points,
        candidate_moving_sec_any_gecko=0.0,
        moving_gecko_seconds=0.0,
        visible_sec=0.0,
        unknown_sec=unknown_sec,
        camera_motion_sec=camera_motion_sec,
        max_simultaneous_geckos=1 if track_points else 0,
        tracking_quality=TrackingQuality.empty(),
        artifact_identity=ArtifactIdentity.test_identity(),
    )


def test_direct_observation_is_detected_candidate():
    observed = TrackPoint("g1", 0.0, (0.1, 0.1, 0.2, 0.2), 0.9, "observed")

    result = decide_presence(_analysis(track_points=(observed,), unknown_sec=0.3), detector=FakeDetector())

    assert result.decision == "detected_candidate"
    assert result.reason_code == "direct_observation"
    assert result.observed_point_count == 1


def test_clean_analysis_without_observation_is_not_observed():
    result = decide_presence(_analysis(), detector=FakeDetector())

    assert result.decision == "not_observed"
    assert result.reason_code == "clean_analysis_without_direct_observation"


def test_unknown_without_observation_is_unresolved():
    result = decide_presence(_analysis(unknown_sec=0.1), detector=FakeDetector())

    assert result.decision == "unresolved"
    assert result.reason_code == "unknown_region_without_direct_observation"


def test_camera_motion_without_observation_is_unresolved():
    result = decide_presence(_analysis(camera_motion_sec=0.1), detector=FakeDetector())

    assert result.decision == "unresolved"
    assert result.reason_code == "camera_motion_without_direct_observation"


def test_failed_or_empty_analysis_is_unresolved():
    failed = decide_presence(_analysis(status="decode_error"), detector=FakeDetector())
    empty = decide_presence(_analysis(analyzed_frame_count=0), detector=FakeDetector())

    assert (failed.decision, failed.reason_code) == ("unresolved", "analysis_decode_error")
    assert (empty.decision, empty.reason_code) == ("unresolved", "no_analyzed_frames")


def test_result_contains_reproducibility_fields_without_source_path(monkeypatch):
    detector = FakeDetector()
    monkeypatch.setattr("gecko_vision_gate.gme_presence.analyze_clip", lambda *_a, **_k: _analysis())

    result = analyze_presence("/private/source/secret-clip.mp4", detector=detector)
    payload = result.to_dict()

    assert payload["schema_version"] == "gme-presence-v1"
    assert payload["checkpoint_sha256"] == "a" * 64
    assert payload["threshold"] == 0.5
    assert "secret-clip" not in repr(payload)
    assert "video_path" not in payload


def test_gate_convenience_wrapper_builds_detector_and_forwards_config(monkeypatch):
    detector = FakeDetector()
    calls: dict[str, object] = {}

    def fake_build_detector(**kwargs):
        calls["detector"] = kwargs
        return detector

    def fake_analyze_presence(video_path, *, detector, config):
        calls["analysis"] = (video_path, detector, config)
        return decide_presence(_analysis(), detector=detector)

    monkeypatch.setattr("gecko_vision_gate.gme_presence.build_detector", fake_build_detector)
    monkeypatch.setattr("gecko_vision_gate.gme_presence.analyze_presence", fake_analyze_presence)

    result = analyze_presence_with_gate(
        "/input.mp4",
        checkpoint="/models/checkpoint_best_ema.pth",
        threshold=0.5,
        model_size="nano",
    )

    assert result.decision == "not_observed"
    assert calls["detector"] == {
        "checkpoint": "/models/checkpoint_best_ema.pth",
        "threshold": 0.5,
        "model_size": "nano",
    }
    assert calls["analysis"][0] == "/input.mp4"
    assert calls["analysis"][1] is detector

