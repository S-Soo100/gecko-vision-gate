"""기존 RF-DETR Gate를 GME detector protocol에 연결하는 얇은 adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import GeckoDetector
from .gme_contracts import Detection
from .provenance import SCHEMA_VERSION, checkpoint_sha256


class GateDetectorAdapter:
    def __init__(self, *, checkpoint: str, threshold: float = 0.5, model_size: str = "nano"):
        self._detector = GeckoDetector(model_size=model_size, threshold=threshold, checkpoint=checkpoint)
        path = Path(checkpoint)
        self.model_name = f"rf-detr-{model_size}"
        self.model_version = f"{path.parent.name} ({path.stem})"
        self.checkpoint_sha256 = checkpoint_sha256(path)
        self.schema_version = SCHEMA_VERSION
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float) -> tuple[Detection, ...]:
        return tuple(
            Detection(timestamp_sec, tuple(float(v) for v in raw.xywh), raw.confidence, raw.class_name)
            for raw in self._detector.detect(frame_bgr)
        )


def build_detector(*, checkpoint: str, threshold: float = 0.5, model_size: str = "nano") -> GateDetectorAdapter:
    return GateDetectorAdapter(checkpoint=checkpoint, threshold=threshold, model_size=model_size)
