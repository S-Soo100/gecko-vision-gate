"""촬영 서버용 GME 3상태 presence 계약.

`not_observed`는 게코 부재가 아니라 현재 모델의 직접 관측이 없었다는 뜻이다.
이 모듈은 저장 경로 선택을 돕지만 삭제·GT·행동·VLM route를 결정하지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .gme_contracts import Detector, GMEAnalysis, GMEConfig
from .gme_detector import build_detector
from .gme_engine import analyze_clip

PRESENCE_SCHEMA_VERSION = "gme-presence-v1"

PresenceDecision = Literal["detected_candidate", "not_observed", "unresolved"]


@dataclass(frozen=True, slots=True)
class PresenceResult:
    schema_version: str
    decision: PresenceDecision
    reason_code: str
    analysis_status: str
    model_name: str
    model_version: str
    checkpoint_sha256: str
    detector_schema_version: str
    threshold: float
    engine_schema_version: str
    algorithm_version: str
    detector_identity: str
    duration_sec: float
    decoded_frame_count: int
    analyzed_frame_count: int
    observed_point_count: int
    candidate_moving_sec_any_gecko: float
    unknown_sec: float
    camera_motion_sec: float
    max_simultaneous_geckos: int

    def to_dict(self) -> dict[str, object]:
        """경로·credential 없이 JSON 직렬화 가능한 사전을 반환한다."""

        return asdict(self)


def decide_presence(analysis: GMEAnalysis, *, detector: Detector) -> PresenceResult:
    """GME 분석을 보수적인 3상태 계약으로 변환한다."""

    observed_point_count = sum(point.provenance == "observed" for point in analysis.track_points)
    if analysis.status != "ok":
        decision: PresenceDecision = "unresolved"
        reason_code = f"analysis_{analysis.status}"
    elif analysis.analyzed_frame_count == 0:
        decision = "unresolved"
        reason_code = "no_analyzed_frames"
    elif observed_point_count > 0:
        decision = "detected_candidate"
        reason_code = "direct_observation"
    elif analysis.unknown_sec > 0:
        decision = "unresolved"
        reason_code = "unknown_region_without_direct_observation"
    elif analysis.camera_motion_sec > 0:
        decision = "unresolved"
        reason_code = "camera_motion_without_direct_observation"
    else:
        decision = "not_observed"
        reason_code = "clean_analysis_without_direct_observation"

    identity = analysis.artifact_identity
    return PresenceResult(
        schema_version=PRESENCE_SCHEMA_VERSION,
        decision=decision,
        reason_code=reason_code,
        analysis_status=analysis.status,
        model_name=detector.model_name,
        model_version=detector.model_version,
        checkpoint_sha256=detector.checkpoint_sha256,
        detector_schema_version=detector.schema_version,
        threshold=detector.threshold,
        engine_schema_version=identity.engine_schema_version,
        algorithm_version=identity.algorithm_version,
        detector_identity=identity.detector_identity,
        duration_sec=analysis.duration_sec,
        decoded_frame_count=analysis.decoded_frame_count,
        analyzed_frame_count=analysis.analyzed_frame_count,
        observed_point_count=observed_point_count,
        candidate_moving_sec_any_gecko=analysis.candidate_moving_sec_any_gecko,
        unknown_sec=analysis.unknown_sec,
        camera_motion_sec=analysis.camera_motion_sec,
        max_simultaneous_geckos=analysis.max_simultaneous_geckos,
    )


def analyze_presence(
    video_path: str | Path,
    *,
    detector: Detector,
    config: GMEConfig = GMEConfig(),
) -> PresenceResult:
    """이미 생성된 detector로 영상을 분석하고 3상태 결과를 반환한다."""

    analysis = analyze_clip(video_path, detector=detector, config=config)
    return decide_presence(analysis, detector=detector)


def analyze_presence_with_gate(
    video_path: str | Path,
    *,
    checkpoint: str,
    threshold: float = 0.5,
    model_size: str = "nano",
    config: GMEConfig = GMEConfig(),
) -> PresenceResult:
    """RF-DETR Gate detector 생성까지 포함한 최소 서버 호출 함수."""

    detector = build_detector(checkpoint=checkpoint, threshold=threshold, model_size=model_size)
    return analyze_presence(video_path, detector=detector, config=config)

