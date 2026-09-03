"""게코 track 변화와 카메라 변화를 분리해 candidate 상태 구간을 만든다."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from .gme_contracts import StateInterval, TrackPoint


def classify_track_motion(previous: TrackPoint, current: TrackPoint, *, threshold_body_lengths: float) -> str:
    if previous.track_id != current.track_id:
        return "unknown"
    px, py, pw, ph = previous.bbox_norm
    cx, cy, cw, ch = current.bbox_norm
    body = max(float(np.hypot((pw + cw) / 2, (ph + ch) / 2)), 1e-9)
    displacement = float(np.hypot((cx + cw / 2) - (px + pw / 2), (cy + ch / 2) - (py + ph / 2)))
    return "moving" if displacement / body >= threshold_body_lengths else "static"


def promote_slow_motion(
    frames,
    points: tuple[TrackPoint, ...],
    *,
    window_sec: float,
    threshold_body_lengths: float,
    max_track_gap_sec: float,
):
    """직전 프레임만으로 놓치는 느린 이동을 중심 시간창의 순이동으로 보완한다."""
    if window_sec <= 0 or not frames or not points:
        return tuple(frames)

    by_track: dict[str, list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    moving_keys: set[tuple[str, float]] = set()
    minimum_span = window_sec * 0.8
    half_window = window_sec / 2
    for track_id, track_points in by_track.items():
        track_points.sort(key=lambda point: point.timestamp_sec)
        segments: list[list[TrackPoint]] = []
        for point in track_points:
            if not segments or point.timestamp_sec - segments[-1][-1].timestamp_sec > max_track_gap_sec:
                segments.append([point])
            else:
                segments[-1].append(point)
        for segment in segments:
            times = [point.timestamp_sec for point in segment]
            for point in segment:
                left_index = bisect_left(times, point.timestamp_sec - half_window)
                right_index = bisect_right(times, point.timestamp_sec + half_window) - 1
                if right_index <= left_index:
                    continue
                left = segment[left_index]
                right = segment[right_index]
                if right.timestamp_sec - left.timestamp_sec + 1e-9 < minimum_span:
                    continue
                if classify_track_motion(
                    left,
                    right,
                    threshold_body_lengths=threshold_body_lengths,
                ) == "moving":
                    moving_keys.add((track_id, point.timestamp_sec))

    promoted = []
    for timestamp, track_states, camera_motion in frames:
        updated = {
            track_id: (
                "moving"
                if state == "static" and (track_id, timestamp) in moving_keys
                else state
            )
            for track_id, state in track_states.items()
        }
        promoted.append((timestamp, updated, camera_motion))
    return tuple(promoted)


def detect_camera_motion(previous_gray, current_gray, *, threshold_norm: float, overlay_height_ratio: float = 0.12) -> bool:
    h, w = previous_gray.shape[:2]
    mask = np.full_like(previous_gray, 255, dtype=np.uint8)
    mask[int(h * (1 - overlay_height_ratio)):, :] = 0
    old = cv2.goodFeaturesToTrack(previous_gray, maxCorners=120, qualityLevel=0.01, minDistance=8, mask=mask)
    if old is None or len(old) < 8:
        return False
    new, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, old, None)
    if new is None or status is None:
        return False
    valid = status.reshape(-1) == 1
    if valid.sum() < 8:
        return False
    delta = np.median(new[valid].reshape(-1, 2) - old[valid].reshape(-1, 2), axis=0)
    return float(np.hypot(*delta)) / max(float(np.hypot(w, h)), 1.0) >= threshold_norm


def detect_exposure_change(previous_gray, current_gray, *, threshold: float = 0.15) -> bool:
    """IR 모드/노출 급변을 움직임과 별도 품질 신호로 찾는다."""
    previous_median = float(np.median(previous_gray))
    current_median = float(np.median(current_gray))
    return abs(current_median - previous_median) / 255.0 >= threshold


@dataclass(frozen=True, slots=True)
class StateSummary:
    intervals: tuple[StateInterval, ...]
    candidate_moving_sec_any_gecko: float
    moving_gecko_seconds: float
    visible_sec: float
    unknown_sec: float
    camera_motion_sec: float


def aggregate_states(frames, *, duration_sec: float) -> StateSummary:
    if not frames or duration_sec <= 0:
        return StateSummary((), 0.0, 0.0, 0.0, max(0.0, duration_sec), 0.0)
    ordered = sorted(frames, key=lambda v: v[0])
    raw: list[StateInterval] = []
    moving_gecko_seconds = 0.0
    for index, (start, track_states, camera_motion) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else duration_sec
        end = min(duration_sec, end)
        if end <= start:
            continue
        if camera_motion:
            state, track_ids = "camera_motion", ()
        else:
            moving = tuple(sorted(k for k, value in track_states.items() if value == "moving"))
            static = tuple(sorted(k for k, value in track_states.items() if value == "static"))
            if moving:
                state, track_ids = "moving", moving
                moving_gecko_seconds += (end - start) * len(moving)
            elif static:
                state, track_ids = "static", static
            else:
                state, track_ids = "unknown", ()
        if raw and raw[-1].state == state and raw[-1].track_ids == track_ids and abs(raw[-1].end_sec - start) < 1e-6:
            previous = raw.pop()
            raw.append(StateInterval(previous.start_sec, end, state, track_ids))
        else:
            raw.append(StateInterval(start, end, state, track_ids))
    moving = sum(v.end_sec - v.start_sec for v in raw if v.state == "moving")
    visible = sum(v.end_sec - v.start_sec for v in raw if v.state in {"moving", "static"})
    unknown = sum(v.end_sec - v.start_sec for v in raw if v.state == "unknown")
    camera = sum(v.end_sec - v.start_sec for v in raw if v.state == "camera_motion")
    return StateSummary(tuple(raw), moving, moving_gecko_seconds, visible, unknown, camera)
