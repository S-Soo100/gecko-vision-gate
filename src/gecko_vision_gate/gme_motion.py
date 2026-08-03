"""게코 track 변화와 카메라 변화를 분리해 candidate 상태 구간을 만든다."""

from __future__ import annotations

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
