from __future__ import annotations

import hashlib

import numpy as np
import pytest


V25_BYTES = b"verified-yolo26n-v25"
V25_SHA = hashlib.sha256(V25_BYTES).hexdigest()
V26_SHA = "a00e5a7a1e1f9197accb036339a38a7c821f03c8ab79611ebce89e5cde59b513"
V26_IDENTITY = "89e4738a60ebb71900e05e96f5b7262e8b900f5c9bba9b9cb9e34fca36f789b7"
V25_PRODUCTION_SHA = "2b128f105e898bc472ed66861583ab80007dae6e94b291db497d7a2f8081f84a"
V25_PRODUCTION_IDENTITY = "d4654168af21d26697ab1bd9a5dc4a05bd92baf5c9328800915cc347803d05b6"


class _Tensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _Boxes:
    def __init__(self, *, xywh, confidence, class_ids):
        self.xywh = _Tensor(xywh)
        self.conf = _Tensor(confidence)
        self.cls = _Tensor(class_ids)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def _checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(V25_BYTES)
    return path


def test_build_rejects_checkpoint_sha_mismatch_before_model_load(tmp_path):
    from gecko_vision_gate.gme_yolo_detector import build_yolo_detector

    loaded = []
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        build_yolo_detector(
            checkpoint=_checkpoint(tmp_path),
            expected_sha256="0" * 64,
            model_factory=lambda path: loaded.append(path),
        )

    assert loaded == []


def test_detect_uses_frozen_inference_contract_and_filters_score_boundary(tmp_path):
    from gecko_vision_gate.gme_yolo_detector import build_yolo_detector

    model = _Model(
        _Result(
            _Boxes(
                xywh=[[10.0, 20.0, 4.0, 8.0], [30.0, 40.0, 10.0, 12.0], [50.0, 60.0, 14.0, 16.0]],
                confidence=[0.19, 0.20, 0.91],
                class_ids=[0.0, 0.0, 0.0],
            )
        )
    )
    detector = build_yolo_detector(
        checkpoint=_checkpoint(tmp_path),
        expected_sha256=V25_SHA,
        raw_confidence=0.001,
        score_threshold=0.20,
        image_size=960,
        nms_iou=0.70,
        max_detections=50,
        device="mps",
        model_factory=lambda _path: model,
    )

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = detector.detect(frame, 1.5)

    assert [(item.confidence, item.bbox_xywh, item.class_name) for item in detections] == [
        (0.91, (50.0, 60.0, 14.0, 16.0), "gecko"),
        (0.20, (30.0, 40.0, 10.0, 12.0), "gecko"),
    ]
    assert model.calls == [
        {
            "source": frame,
            "conf": 0.001,
            "imgsz": 960,
            "iou": 0.70,
            "max_det": 50,
            "device": "mps",
            "verbose": False,
            "stream": False,
            "save": False,
        }
    ]
    assert detector.checkpoint_sha256 == V25_SHA
    assert detector.model_name == "yolo26n"
    assert detector.model_version == "v2.5-warm-start"
    assert detector.threshold == 0.20


def test_detect_rejects_unexpected_class_in_single_class_checkpoint(tmp_path):
    from gecko_vision_gate.gme_yolo_detector import build_yolo_detector

    model = _Model(
        _Result(_Boxes(xywh=[[1.0, 1.0, 1.0, 1.0]], confidence=[0.9], class_ids=[1.0]))
    )
    detector = build_yolo_detector(
        checkpoint=_checkpoint(tmp_path),
        expected_sha256=V25_SHA,
        model_factory=lambda _path: model,
    )

    with pytest.raises(ValueError, match="unexpected YOLO class id"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), 0.0)


def test_detect_rejects_result_count_mismatch(tmp_path):
    from gecko_vision_gate.gme_yolo_detector import build_yolo_detector

    class EmptyModel:
        def predict(self, **_kwargs):
            return []

    detector = build_yolo_detector(
        checkpoint=_checkpoint(tmp_path),
        expected_sha256=V25_SHA,
        model_factory=lambda _path: EmptyModel(),
    )

    with pytest.raises(ValueError, match="exactly one result"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), 0.0)


def test_v26_execution_identity_uses_the_full_frozen_contract():
    from gecko_vision_gate.gme_engine import detector_identity
    from gecko_vision_gate.gme_yolo_detector import YoloGMEAdapter

    detector = YoloGMEAdapter(
        model=_Model(_Result(None)),
        checkpoint_sha=V26_SHA,
        model_version="v2.6-warm-start-s28",
        raw_confidence=0.001,
        score_threshold=0.15,
        image_size=960,
        nms_iou=0.70,
        post_nms_iou=0.55,
        max_detections=50,
        analysis_fps=10.0,
        temporal_window_frames=5,
        temporal_min_positive_frames=3,
        device="mps",
    )

    assert detector.execution_identity == V26_IDENTITY
    assert detector_identity(detector) == V26_IDENTITY


def test_v25_legacy_identity_remains_unchanged():
    from gecko_vision_gate.gme_engine import detector_identity
    from gecko_vision_gate.gme_yolo_detector import YoloGMEAdapter

    detector = YoloGMEAdapter(
        model=_Model(_Result(None)),
        checkpoint_sha=V25_PRODUCTION_SHA,
        raw_confidence=0.001,
        score_threshold=0.20,
        image_size=960,
        nms_iou=0.70,
        max_detections=50,
        device="mps",
    )

    assert detector_identity(detector) == V25_PRODUCTION_IDENTITY


def test_post_nms_suppresses_overlapping_lower_score_box(tmp_path):
    from gecko_vision_gate.gme_yolo_detector import build_yolo_detector

    model = _Model(
        _Result(
            _Boxes(
                xywh=[[50.0, 50.0, 20.0, 20.0], [51.0, 51.0, 20.0, 20.0], [90.0, 90.0, 10.0, 10.0]],
                confidence=[0.91, 0.80, 0.70],
                class_ids=[0.0, 0.0, 0.0],
            )
        )
    )
    detector = build_yolo_detector(
        checkpoint=_checkpoint(tmp_path),
        expected_sha256=V25_SHA,
        model_version="v2.6-warm-start-s28",
        score_threshold=0.15,
        post_nms_iou=0.55,
        analysis_fps=10.0,
        temporal_window_frames=5,
        temporal_min_positive_frames=3,
        model_factory=lambda _path: model,
    )

    detections = detector.detect(np.zeros((120, 120, 3), dtype=np.uint8), 0.0)

    assert [(row.confidence, row.bbox_xywh) for row in detections] == [
        (0.91, (50.0, 50.0, 20.0, 20.0)),
        (0.70, (90.0, 90.0, 10.0, 10.0)),
    ]
