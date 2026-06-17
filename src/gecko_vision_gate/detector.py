"""RF-DETR core detector 래퍼.

Phase 0: COCO pretrained (80 class). gecko 클래스가 COCO 에 없으므로 이 단계의
목적은 "추론 파이프라인이 도는지" sanity check 다. 사람 손이 나오는 hand_feeding
클립에선 person 이 잡혀 파이프라인 동작이 확인되고, 게코 영상에선 (예상대로)
아무것도 못 잡아 "fine-tune 이 필요하다"가 실증된다. 실제 gecko 검출은
Phase 1(seed 라벨 → fine-tune) 이후.

RF-DETR/torch import 가 무거워서 추론 시점까지 미루는 lazy import 를 쓴다.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import cv2
import numpy as np

# model_size → rfdetr 클래스명. Phase 0 검증은 가장 가벼운 Nano 로 충분.
_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
}


@dataclass(frozen=True, slots=True)
class RawDetection:
    """detector 가 한 프레임에서 뽑은 객체 1건 (frame_ts 는 호출부에서 붙인다)."""

    class_name: str
    confidence: float
    xywh: list[int]  # [x, y, w, h]


def _xyxy_to_xywh(xyxy) -> list[int]:
    """supervision 의 xyxy(x1,y1,x2,y2) → 계약 포맷 [x,y,w,h]."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
    return [x1, y1, x2 - x1, y2 - y1]


def _load_coco_classes() -> dict[int, str]:
    """COCO id→name 매핑. rfdetr 버전마다 모듈 경로가 달라 여러 후보를 시도한다."""
    candidates = [
        ("rfdetr.util.coco_classes", "COCO_CLASSES"),
        ("rfdetr.assets.coco_classes", "COCO_CLASSES"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cc = getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(cc, dict):
            return {int(k): str(v) for k, v in cc.items()}
        return {i: str(v) for i, v in enumerate(cc)}
    return {}


class GeckoDetector:
    """RF-DETR 래퍼. detect(frame_bgr) -> list[RawDetection].

    모델 1회 로드 후 재사용 (여러 프레임·여러 클립에 같은 인스턴스 사용).
    """

    def __init__(self, model_size: str = "nano", threshold: float = 0.5):
        self.model_size = model_size
        self.threshold = threshold
        self._model = None
        self._coco_classes: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        rfdetr = importlib.import_module("rfdetr")
        cls_name = _MODEL_CLASSES.get(self.model_size, "RFDETRNano")
        self._model = getattr(rfdetr, cls_name)()
        self._coco_classes = _load_coco_classes()

    def detect(self, frame_bgr: np.ndarray) -> list[RawDetection]:
        self._ensure_loaded()

        # cv2 는 BGR, RF-DETR(PIL 경유) 은 RGB 를 기대 → 변환 필수
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        det = self._model.predict(image, threshold=self.threshold)

        # supervision Detections: .xyxy (N,4) · .confidence (N,) · .class_id (N,)
        results: list[RawDetection] = []
        for xyxy, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            name = self._coco_classes.get(int(cid), f"class_{int(cid)}")
            results.append(RawDetection(name, float(conf), _xyxy_to_xywh(xyxy)))
        return results
