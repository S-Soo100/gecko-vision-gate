"""동결된 Ultralytics YOLO checkpoint를 GME detector 계약에 연결한다."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .gme_contracts import Detection
from .provenance import SCHEMA_VERSION, checkpoint_sha256


YOLO_BBOX_COORDINATE_CONTRACT = "xywh-top-left-v1"


class YoloGMEAdapter:
    def __init__(
        self,
        *,
        model: object,
        checkpoint_sha: str,
        raw_confidence: float,
        score_threshold: float,
        image_size: int,
        nms_iou: float,
        max_detections: int,
        device: str,
        model_version: str = "v2.5-warm-start",
        post_nms_iou: float | None = None,
        analysis_fps: float | None = None,
        temporal_window_frames: int | None = None,
        temporal_min_positive_frames: int | None = None,
    ) -> None:
        if not 0 <= raw_confidence <= score_threshold <= 1:
            raise ValueError("invalid YOLO confidence contract")
        if image_size < 1 or max_detections < 1 or not 0 <= nms_iou <= 1:
            raise ValueError("invalid YOLO inference contract")
        if not model_version.strip():
            raise ValueError("blank YOLO model version")
        extended_values = (
            post_nms_iou,
            analysis_fps,
            temporal_window_frames,
            temporal_min_positive_frames,
        )
        uses_extended_contract = model_version != "v2.5-warm-start" or any(value is not None for value in extended_values)
        if uses_extended_contract and any(value is None for value in extended_values):
            raise ValueError("incomplete YOLO execution contract")
        if post_nms_iou is not None and not 0 <= post_nms_iou <= 1:
            raise ValueError("invalid post NMS IoU")
        if analysis_fps is not None and (not math.isfinite(analysis_fps) or analysis_fps <= 0):
            raise ValueError("invalid analysis fps")
        if temporal_window_frames is not None and (
            not isinstance(temporal_window_frames, int)
            or isinstance(temporal_window_frames, bool)
            or temporal_window_frames <= 0
        ):
            raise ValueError("invalid temporal window")
        if temporal_min_positive_frames is not None and (
            not isinstance(temporal_min_positive_frames, int)
            or isinstance(temporal_min_positive_frames, bool)
            or temporal_window_frames is None
            or not 1 <= temporal_min_positive_frames <= temporal_window_frames
        ):
            raise ValueError("invalid temporal minimum")
        self._model = model
        self.model_name = "yolo26n"
        self.model_version = model_version
        self.checkpoint_sha256 = checkpoint_sha
        self.schema_version = SCHEMA_VERSION
        self.threshold = score_threshold
        self.raw_confidence = raw_confidence
        self.image_size = image_size
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.device = device
        self.post_nms_iou = post_nms_iou
        self.bbox_coordinate_contract = YOLO_BBOX_COORDINATE_CONTRACT
        if uses_extended_contract:
            self.execution_contract = {
                "schema": "gme-yolo-execution-v2",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "detector_schema_version": self.schema_version,
                "raw_confidence": self.raw_confidence,
                "score_threshold": self.threshold,
                "image_size": self.image_size,
                "model_nms_iou": self.nms_iou,
                "post_nms_iou": self.post_nms_iou,
                "max_detections": self.max_detections,
                "analysis_fps": analysis_fps,
                "temporal_window_frames": temporal_window_frames,
                "temporal_min_positive_frames": temporal_min_positive_frames,
                "bbox_coordinate_contract": YOLO_BBOX_COORDINATE_CONTRACT,
            }
        else:
            # 좌표가 수정된 결과를 과거 center-xywh identity로 저장하지 않는다.
            self.execution_contract = {
                "schema": "gme-yolo-execution-v1",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "detector_schema_version": self.schema_version,
                "raw_confidence": self.raw_confidence,
                "score_threshold": self.threshold,
                "image_size": self.image_size,
                "model_nms_iou": self.nms_iou,
                "max_detections": self.max_detections,
                "bbox_coordinate_contract": YOLO_BBOX_COORDINATE_CONTRACT,
            }
        canonical = json.dumps(self.execution_contract, sort_keys=True, separators=(",", ":"))
        self.execution_identity = hashlib.sha256(canonical.encode()).hexdigest()

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float) -> tuple[Detection, ...]:
        raw_results = self._model.predict(
            source=frame_bgr,
            conf=self.raw_confidence,
            imgsz=self.image_size,
            iou=self.nms_iou,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
            stream=False,
            save=False,
        )
        if len(raw_results) != 1:
            raise ValueError("YOLO must return exactly one result per frame")
        boxes = raw_results[0].boxes
        if boxes is None:
            return ()
        xywh_rows = boxes.xywh.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        if not (len(xywh_rows) == len(confidences) == len(class_ids)):
            raise ValueError("YOLO box, confidence, and class counts differ")

        accepted: list[tuple[float, tuple[float, float, float, float]]] = []
        for xywh, confidence, class_id in zip(xywh_rows, confidences, class_ids, strict=True):
            score = float(confidence)
            numeric_class_id = float(class_id)
            if not math.isfinite(numeric_class_id) or numeric_class_id != 0.0:
                raise ValueError("unexpected YOLO class id for single-class gecko checkpoint")
            if score < self.threshold:
                continue
            if len(xywh) != 4:
                raise ValueError("YOLO xywh row must contain four values")
            accepted.append((score, tuple(float(value) for value in xywh)))
        ordered = sorted(accepted, key=lambda item: (-item[0], item[1]))
        if self.post_nms_iou is None:
            selected = ordered
        else:
            selected = []
            for candidate in ordered:
                if all(_center_xywh_iou(candidate[1], kept[1]) <= self.post_nms_iou for kept in selected):
                    selected.append(candidate)
        return tuple(
            Detection(
                timestamp_sec=timestamp_sec,
                bbox_xywh=_center_xywh_to_top_left(center_xywh),
                confidence=score,
                class_name="gecko",
            )
            for score, center_xywh in selected
        )


def _center_xywh_to_top_left(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Ultralytics 중심 xywh를 GME의 좌상단 xywh 계약으로 변환한다."""

    center_x, center_y, width, height = bbox
    return center_x - width / 2, center_y - height / 2, width, height


def _center_xywh_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Ultralytics ``boxes.xywh``(center x/y)를 기준으로 post NMS를 재현한다."""

    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_x1, left_y1 = lx - lw / 2, ly - lh / 2
    right_x1, right_y1 = rx - rw / 2, ry - rh / 2
    intersection_w = max(0.0, min(lx + lw / 2, rx + rw / 2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(ly + lh / 2, ry + rh / 2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def build_yolo_detector(
    *,
    checkpoint: str | Path,
    expected_sha256: str,
    raw_confidence: float = 0.001,
    score_threshold: float = 0.20,
    image_size: int = 960,
    nms_iou: float = 0.70,
    max_detections: int = 50,
    device: str = "mps",
    model_version: str = "v2.5-warm-start",
    post_nms_iou: float | None = None,
    analysis_fps: float | None = None,
    temporal_window_frames: int | None = None,
    temporal_min_positive_frames: int | None = None,
    model_factory: Callable[[str], object] | None = None,
) -> YoloGMEAdapter:
    path = Path(checkpoint)
    actual_sha256 = checkpoint_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(str(path))
    return YoloGMEAdapter(
        model=model,
        checkpoint_sha=actual_sha256,
        raw_confidence=raw_confidence,
        score_threshold=score_threshold,
        image_size=image_size,
        nms_iou=nms_iou,
        max_detections=max_detections,
        device=device,
        model_version=model_version,
        post_nms_iou=post_nms_iou,
        analysis_fps=analysis_fps,
        temporal_window_frames=temporal_window_frames,
        temporal_min_positive_frames=temporal_min_positive_frames,
    )
