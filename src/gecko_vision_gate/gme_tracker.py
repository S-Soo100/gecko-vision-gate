"""작은 개체 수에 맞춘 결정론적 multi-gecko tracker."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from .gme_contracts import Detection, TrackPoint


def _norm_bbox(bbox, shape) -> tuple[float, float, float, float]:
    h, w = shape[:2]
    x, y, bw, bh = bbox
    nx = min(1.0, max(0.0, x / w))
    ny = min(1.0, max(0.0, y / h))
    nw = min(max(0.0, bw / w), 1.0 - nx)
    nh = min(max(0.0, bh / h), 1.0 - ny)
    return (nx, ny, nw, nh)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a, b, shape) -> float:
    h, w = shape[:2]
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return float(np.hypot((ax + aw / 2 - bx - bw / 2) / w, (ay + ah / 2 - by - bh / 2) / h))


@dataclass(slots=True)
class _Track:
    track_id: str
    bbox: tuple[float, float, float, float]
    timestamp_sec: float
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    points: tuple[TrackPoint, ...]
    needs_redetection: bool


class MultiGeckoTracker:
    def __init__(self, *, min_flow_points: int = 4, max_match_distance: float = 0.25):
        self.min_flow_points = min_flow_points
        self.max_match_distance = max_match_distance
        self._next_id = 1
        self._tracks: dict[str, _Track] = {}
        self.fragmentation_count = 0
        self.possible_id_switch_count = 0
        self.detection_gap_count = 0
        self.position_jump_count = 0

    def reset(self) -> None:
        """IR/exposure 전환처럼 optical-flow 기준이 무효가 되면 active identity만 끊는다."""
        if self._tracks:
            self.detection_gap_count += len(self._tracks)
        self._tracks.clear()

    def update_anchor(self, detections: tuple[Detection, ...], frame_shape) -> tuple[TrackPoint, ...]:
        geckos = sorted((d for d in detections if d.class_name == "gecko"), key=lambda d: (d.bbox_xywh[0], d.bbox_xywh[1], -d.confidence))
        available = set(self._tracks)
        assigned: list[tuple[str, Detection]] = []
        for detection in geckos:
            candidates = []
            for track_id in available:
                track = self._tracks[track_id]
                distance = _center_distance(track.bbox, detection.bbox_xywh, frame_shape)
                if distance <= self.max_match_distance:
                    candidates.append((-_iou(track.bbox, detection.bbox_xywh), distance, track_id))
            if candidates:
                _, distance, track_id = min(candidates)
                available.remove(track_id)
                if distance > 0.15:
                    self.position_jump_count += 1
            else:
                track_id = f"g{self._next_id:04d}"
                self._next_id += 1
                if self._tracks:
                    self.fragmentation_count += 1
            self._tracks[track_id] = _Track(track_id, detection.bbox_xywh, detection.timestamp_sec, detection.confidence)
            assigned.append((track_id, detection))
        for track_id in available:
            del self._tracks[track_id]
            self.detection_gap_count += 1
        return tuple(
            TrackPoint(tid, d.timestamp_sec, _norm_bbox(d.bbox_xywh, frame_shape), d.confidence, "observed")
            for tid, d in sorted(assigned)
        )

    def update_tracked(self, previous_gray, current_gray, timestamp_sec: float, frame_shape) -> TrackingUpdate:
        points: list[TrackPoint] = []
        lost: list[str] = []
        for track_id, track in sorted(self._tracks.items()):
            x, y, w, h = (int(round(v)) for v in track.bbox)
            mask = np.zeros_like(previous_gray, dtype=np.uint8)
            mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 255
            old = cv2.goodFeaturesToTrack(previous_gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
            if old is None or len(old) < self.min_flow_points:
                lost.append(track_id)
                continue
            new, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, old, None)
            if new is None or status is None:
                lost.append(track_id)
                continue
            valid_old = old[status.reshape(-1) == 1].reshape(-1, 2)
            valid_new = new[status.reshape(-1) == 1].reshape(-1, 2)
            if len(valid_new) < self.min_flow_points:
                lost.append(track_id)
                continue
            delta = np.median(valid_new - valid_old, axis=0)
            moved = (float(x + delta[0]), float(y + delta[1]), float(w), float(h))
            confidence = min(1.0, len(valid_new) / max(len(old), 1))
            self._tracks[track_id] = _Track(track_id, moved, timestamp_sec, confidence)
            points.append(TrackPoint(track_id, timestamp_sec, _norm_bbox(moved, frame_shape), confidence, "tracked"))
        for track_id in lost:
            self._tracks.pop(track_id, None)
            self.detection_gap_count += 1
        return TrackingUpdate(tuple(points), bool(lost) or (not points and bool(lost)))


def interpolate_short_gaps(
    points: tuple[TrackPoint, ...], *, step_sec: float, max_gap_sec: float
) -> tuple[TrackPoint, ...]:
    by_track: dict[str, list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)
    out = list(points)
    for track_id, values in by_track.items():
        values.sort(key=lambda p: p.timestamp_sec)
        for left, right in zip(values, values[1:]):
            gap = right.timestamp_sec - left.timestamp_sec
            if gap <= step_sec + 1e-9 or gap > max_gap_sec + 1e-9:
                continue
            steps = int(round(gap / step_sec))
            for index in range(1, steps):
                ratio = index / steps
                bbox = tuple(a + (b - a) * ratio for a, b in zip(left.bbox_norm, right.bbox_norm))
                out.append(TrackPoint(track_id, left.timestamp_sec + index * step_sec, bbox, min(left.confidence, right.confidence), "interpolated"))
    return tuple(sorted(out, key=lambda p: (p.timestamp_sec, p.track_id)))
