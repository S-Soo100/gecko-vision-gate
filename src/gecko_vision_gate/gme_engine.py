"""한 번의 순차 decode로 GME shadow 결과를 만드는 코어 엔진."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from .gme_contracts import ArtifactIdentity, Detector, GMEAnalysis, GMEConfig, TrackingQuality
from .gme_motion import aggregate_states, classify_track_motion, detect_camera_motion, detect_exposure_change
from .gme_temporal import AnalysisClock, TemporalDetectionGate
from .gme_tracker import MultiGeckoTracker, interpolate_short_gaps

ENGINE_SCHEMA_VERSION = "gme-shadow-v1"
ALGORITHM_VERSION = "gme-motion-v0"


def detector_identity(detector: Detector) -> str:
    """검출기 실행 계약 전체를 GME 원장용 단일 지문으로 묶는다."""
    execution_identity = getattr(detector, "execution_identity", None)
    if execution_identity is not None:
        return str(execution_identity)
    raw = "|".join((detector.model_name, detector.model_version, detector.checkpoint_sha256, detector.schema_version, str(detector.threshold)))
    return hashlib.sha256(raw.encode()).hexdigest()


def analyze_clip(
    video_path: str | Path,
    *,
    detector: Detector,
    config: GMEConfig = GMEConfig(),
    tracker: MultiGeckoTracker | None = None,
) -> GMEAnalysis:
    cap = cv2.VideoCapture(str(video_path))
    identity = ArtifactIdentity(ENGINE_SCHEMA_VERSION, ALGORITHM_VERSION, detector_identity(detector))
    decoded = analyzed = 0
    source_fps: float | None = None
    points = []
    frame_states = []
    frame_debug = []
    previous_gray = None
    previous_by_track = {}
    tracker = tracker or MultiGeckoTracker()
    need_redetection = True
    next_anchor = 0.0
    last_timestamp = 0.0
    analysis_clock = AnalysisClock(max_analysis_fps=config.analysis_fps)
    temporal_gate = TemporalDetectionGate(
        window_frames=config.detection_window_frames,
        min_positive_frames=config.detection_min_positive_frames,
    )
    try:
        if not cap.isOpened():
            return GMEAnalysis("invalid_metadata", 0.0, 0, 0, None, (), (), 0, 0, 0, 0, 0, 0, TrackingQuality.empty(), identity)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps != fps:
            return GMEAnalysis("invalid_metadata", 0.0, 0, 0, None, (), (), 0, 0, 0, 0, 0, 0, TrackingQuality.empty(), identity)
        source_fps = fps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = decoded
            decoded += 1
            timestamp = frame_index / fps
            last_timestamp = timestamp
            if not analysis_clock.accept(frame_index, fps):
                continue
            analyzed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_motion = False
            exposure_change = False
            if previous_gray is not None:
                exposure_change = detect_exposure_change(previous_gray, gray)
                camera_motion = detect_camera_motion(
                    previous_gray, gray, threshold_norm=config.camera_motion_threshold_norm,
                    overlay_height_ratio=config.timestamp_overlay_height_ratio,
                )
                if exposure_change:
                    camera_motion = False
            if exposure_change:
                # 전환 프레임 optical-flow/bbox는 다음 프레임의 움직임 기준으로 쓰지 않는다.
                tracker.reset()
                temporal_gate.reset()
                previous_by_track.clear()
                current = ()
                need_redetection = True
            elif config.detector_every_analysis_frame or need_redetection or timestamp + 1e-9 >= next_anchor:
                detections = detector.detect(frame, timestamp)
                accepted = temporal_gate.push(detections)
                current = tracker.update_anchor(accepted, frame.shape)
                need_redetection = False
                next_anchor = timestamp + config.anchor_interval_sec
            elif previous_gray is not None:
                update = tracker.update_tracked(previous_gray, gray, timestamp, frame.shape)
                current = update.points
                need_redetection = update.needs_redetection
            else:
                current = ()
            points.extend(current)
            current_states = {}
            for point in current:
                previous = previous_by_track.get(point.track_id)
                current_states[point.track_id] = (
                    classify_track_motion(previous, point, threshold_body_lengths=config.moving_threshold_body_lengths)
                    if previous is not None else "static"
                )
                previous_by_track[point.track_id] = point
            frame_states.append((timestamp, current_states, camera_motion))
            frame_debug.append({"timestamp_sec": timestamp, "exposure_change": exposure_change, "camera_motion": camera_motion})
            previous_gray = gray
    finally:
        cap.release()
    if decoded == 0:
        return GMEAnalysis("no_decodable_frames", 0.0, 0, 0, source_fps, (), (), 0, 0, 0, 0, 0, 0, TrackingQuality.empty(), identity)
    duration = max(last_timestamp + 1.0 / source_fps, decoded / source_fps)
    interpolated = interpolate_short_gaps(tuple(points), step_sec=1.0 / min(source_fps, config.analysis_fps), max_gap_sec=config.max_interpolation_gap_sec)
    summary = aggregate_states(tuple(frame_states), duration_sec=duration)
    provenance_counts = {name: 0.0 for name in ("observed", "tracked", "interpolated", "unknown")}
    step = 1.0 / min(source_fps, config.analysis_fps)
    for point in interpolated:
        provenance_counts[point.provenance] += step
    quality = TrackingQuality(
        tracker.fragmentation_count, tracker.possible_id_switch_count, tracker.detection_gap_count,
        tracker.position_jump_count, tuple(sorted(provenance_counts.items())), None,
    )
    return GMEAnalysis(
        "ok", duration, decoded, analyzed, source_fps, summary.intervals, interpolated,
        summary.candidate_moving_sec_any_gecko, summary.moving_gecko_seconds, summary.visible_sec,
        summary.unknown_sec, summary.camera_motion_sec,
        max((len(states) for _, states, _ in frame_states), default=0), quality, identity, tuple(frame_debug),
    )
