"""동결된 Ultralytics YOLO checkpoint를 GME detector 계약에 연결한다."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from .gme_contracts import Detection
from .provenance import SCHEMA_VERSION, checkpoint_sha256


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
    ) -> None:
        if not 0 <= raw_confidence <= score_threshold <= 1:
            raise ValueError("invalid YOLO confidence contract")
        if image_size < 1 or max_detections < 1 or not 0 <= nms_iou <= 1:
            raise ValueError("invalid YOLO inference contract")
        self._model = model
        self.model_name = "yolo26n"
        self.model_version = "v2.5-warm-start"
        self.checkpoint_sha256 = checkpoint_sha
        self.schema_version = SCHEMA_VERSION
        self.threshold = score_threshold
        self.raw_confidence = raw_confidence
        self.image_size = image_size
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.device = device

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

        accepted: list[Detection] = []
        for xywh, confidence, class_id in zip(xywh_rows, confidences, class_ids, strict=True):
            score = float(confidence)
            numeric_class_id = float(class_id)
            if not math.isfinite(numeric_class_id) or numeric_class_id != 0.0:
                raise ValueError("unexpected YOLO class id for single-class gecko checkpoint")
            if score < self.threshold:
                continue
            if len(xywh) != 4:
                raise ValueError("YOLO xywh row must contain four values")
            accepted.append(
                Detection(
                    timestamp_sec=timestamp_sec,
                    bbox_xywh=tuple(float(value) for value in xywh),
                    confidence=score,
                    class_name="gecko",
                )
            )
        return tuple(sorted(accepted, key=lambda item: (-item.confidence, item.bbox_xywh)))


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
    )
