"""Gecko Motion Engine의 detector-agnostic 불변 계약.

GME shadow 결과는 사용자 정답이 아니라 재현 가능한 candidate 측정치다. 이 모듈은
검출기·추적기·worker가 같은 형태를 쓰게 하고 잘못된 수치를 저장 경계 전에 거부한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

Provenance = Literal["observed", "tracked", "interpolated", "unknown"]
State = Literal["moving", "static", "not_visible", "unknown", "camera_motion"]

PROVENANCES = frozenset({"observed", "tracked", "interpolated", "unknown"})
STATES = frozenset({"moving", "static", "not_visible", "unknown", "camera_motion"})


def _finite(value: float, *, minimum: float = 0.0) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


@dataclass(frozen=True, slots=True)
class Detection:
    timestamp_sec: float
    bbox_xywh: tuple[float, float, float, float]
    confidence: float
    class_name: str

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox_xywh
        if not _finite(self.timestamp_sec) or not all(_finite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError("invalid detection geometry")
        if not _finite(self.confidence) or self.confidence > 1:
            raise ValueError("invalid detection confidence")
        if not self.class_name.strip():
            raise ValueError("blank class name")


class Detector(Protocol):
    model_name: str
    model_version: str
    checkpoint_sha256: str
    schema_version: str
    threshold: float

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float) -> tuple[Detection, ...]: ...


@dataclass(frozen=True, slots=True)
class TrackPoint:
    track_id: str
    timestamp_sec: float
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    provenance: Provenance

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox_norm
        if not self.track_id.strip() or not _finite(self.timestamp_sec):
            raise ValueError("invalid track identity/time")
        if not all(_finite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("bbox_norm must be inside [0,1]")
        if not _finite(self.confidence) or self.confidence > 1:
            raise ValueError("invalid track confidence")
        if self.provenance not in PROVENANCES:
            raise ValueError("invalid point provenance")


@dataclass(frozen=True, slots=True)
class StateInterval:
    start_sec: float
    end_sec: float
    state: State
    track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _finite(self.start_sec) or not _finite(self.end_sec) or self.end_sec <= self.start_sec:
            raise ValueError("interval must be positive and half-open")
        if self.state not in STATES:
            raise ValueError("invalid GME state")
        if tuple(sorted(set(self.track_ids))) != self.track_ids:
            raise ValueError("track_ids must be unique and sorted")


@dataclass(frozen=True, slots=True)
class TrackingQuality:
    fragmentation_count: int
    possible_id_switch_count: int
    detection_gap_count: int
    position_jump_count: int
    provenance_seconds: tuple[tuple[str, float], ...]
    multi_gecko_separation_ratio: float | None

    @classmethod
    def empty(cls) -> "TrackingQuality":
        return cls(0, 0, 0, 0, tuple((p, 0.0) for p in sorted(PROVENANCES)), None)

    def __post_init__(self) -> None:
        counts = (self.fragmentation_count, self.possible_id_switch_count, self.detection_gap_count, self.position_jump_count)
        if any(not isinstance(v, int) or v < 0 for v in counts):
            raise ValueError("tracking counts must be non-negative integers")
        if self.multi_gecko_separation_ratio is not None and not (0 <= self.multi_gecko_separation_ratio <= 1):
            raise ValueError("invalid separation ratio")
        for key, value in self.provenance_seconds:
            if key not in PROVENANCES or not _finite(value):
                raise ValueError("invalid provenance duration")


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    engine_schema_version: str
    algorithm_version: str
    detector_identity: str

    @classmethod
    def test_identity(cls) -> "ArtifactIdentity":
        return cls("gme-shadow-v1", "gme-motion-v0", "test-detector")

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.engine_schema_version, self.algorithm_version, self.detector_identity)):
            raise ValueError("artifact identity fields must be nonblank")


@dataclass(frozen=True, slots=True)
class GMEConfig:
    analysis_fps: float = 30.0
    anchor_interval_sec: float = 0.5
    detection_window_frames: int = 1
    detection_min_positive_frames: int = 1
    detector_every_analysis_frame: bool = False
    max_interpolation_gap_sec: float = 1.0
    moving_threshold_body_lengths: float = 0.08
    slow_motion_window_sec: float = 3.0
    slow_motion_max_track_gap_sec: float = 0.25
    camera_motion_threshold_norm: float = 0.015
    tracker_confidence_floor: float = 0.35
    timestamp_overlay_height_ratio: float = 0.12

    @classmethod
    def v26(cls) -> "GMEConfig":
        return cls(
            analysis_fps=10.0,
            anchor_interval_sec=0.1,
            detection_window_frames=5,
            detection_min_positive_frames=3,
            detector_every_analysis_frame=True,
        )

    def __post_init__(self) -> None:
        if not (0 < self.analysis_fps <= 30):
            raise ValueError("analysis_fps must be in (0,30]")
        if not (0 < self.anchor_interval_sec <= 10):
            raise ValueError("invalid anchor interval")
        if (
            not isinstance(self.detection_window_frames, int)
            or isinstance(self.detection_window_frames, bool)
            or self.detection_window_frames <= 0
        ):
            raise ValueError("invalid detection window")
        if (
            not isinstance(self.detection_min_positive_frames, int)
            or isinstance(self.detection_min_positive_frames, bool)
            or not 1 <= self.detection_min_positive_frames <= self.detection_window_frames
        ):
            raise ValueError("invalid detection minimum")
        if not isinstance(self.detector_every_analysis_frame, bool):
            raise ValueError("detector_every_analysis_frame must be bool")
        if not (0 <= self.max_interpolation_gap_sec <= 1.0):
            raise ValueError("interpolation gap must be in [0,1]")
        if not (0 <= self.tracker_confidence_floor <= 1):
            raise ValueError("invalid tracker confidence floor")
        if (
            self.moving_threshold_body_lengths < 0
            or not 0 < self.slow_motion_window_sec <= 10
            or not 0 < self.slow_motion_max_track_gap_sec <= 1
            or self.camera_motion_threshold_norm < 0
        ):
            raise ValueError("motion thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class GMEAnalysis:
    status: str
    duration_sec: float
    decoded_frame_count: int
    analyzed_frame_count: int
    source_fps: float | None
    intervals: tuple[StateInterval, ...]
    track_points: tuple[TrackPoint, ...]
    candidate_moving_sec_any_gecko: float
    moving_gecko_seconds: float
    visible_sec: float
    unknown_sec: float
    camera_motion_sec: float
    max_simultaneous_geckos: int
    tracking_quality: TrackingQuality
    artifact_identity: ArtifactIdentity
    frame_debug: tuple[dict, ...] = field(default_factory=tuple, repr=False)

    @classmethod
    def minimal(
        cls,
        *,
        duration_sec: float,
        intervals: tuple[StateInterval, ...],
        tracking_quality: TrackingQuality,
        artifact_identity: ArtifactIdentity,
        track_points: tuple[TrackPoint, ...] = (),
    ) -> "GMEAnalysis":
        moving = sum(i.end_sec - i.start_sec for i in intervals if i.state == "moving")
        unknown = sum(i.end_sec - i.start_sec for i in intervals if i.state == "unknown")
        camera = sum(i.end_sec - i.start_sec for i in intervals if i.state == "camera_motion")
        visible = sum(i.end_sec - i.start_sec for i in intervals if i.state in {"moving", "static"})
        gecko_seconds = sum((i.end_sec - i.start_sec) * len(i.track_ids) for i in intervals if i.state == "moving")
        return cls(
            status="ok", duration_sec=duration_sec, decoded_frame_count=0, analyzed_frame_count=0,
            source_fps=None, intervals=intervals, track_points=track_points,
            candidate_moving_sec_any_gecko=moving, moving_gecko_seconds=gecko_seconds,
            visible_sec=visible, unknown_sec=unknown, camera_motion_sec=camera,
            max_simultaneous_geckos=max((len(i.track_ids) for i in intervals), default=0),
            tracking_quality=tracking_quality, artifact_identity=artifact_identity,
        )

    def __post_init__(self) -> None:
        if self.status not in {"ok", "no_decodable_frames", "invalid_metadata", "decode_error"}:
            raise ValueError("invalid analysis status")
        if not _finite(self.duration_sec) or self.decoded_frame_count < 0 or self.analyzed_frame_count < 0:
            raise ValueError("invalid media counts")
        previous_end = 0.0
        for interval in self.intervals:
            if interval.start_sec < previous_end - 1e-9 or interval.end_sec > self.duration_sec + 1e-6:
                raise ValueError("state intervals overlap or exceed duration")
            previous_end = interval.end_sec
        for value in (
            self.candidate_moving_sec_any_gecko, self.moving_gecko_seconds, self.visible_sec,
            self.unknown_sec, self.camera_motion_sec,
        ):
            if not _finite(value):
                raise ValueError("invalid analysis duration")
        epsilon = 0.001
        if self.candidate_moving_sec_any_gecko > self.visible_sec + epsilon:
            raise ValueError("candidate moving time cannot exceed visible time")
        if self.moving_gecko_seconds + epsilon < self.candidate_moving_sec_any_gecko:
            raise ValueError("gecko-seconds cannot be below any-gecko moving time")
        if self.visible_sec + self.unknown_sec + self.camera_motion_sec > self.duration_sec + epsilon:
            raise ValueError("state time accounting exceeds clip duration")
