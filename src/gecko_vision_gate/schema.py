"""clip_prelabels JSON 계약 — gecko-vision-gate 의 출력 스키마.

frozen dataclass 로 모델링한 이유 (petcam donts/vlm.md 3):
prelabel 결과는 한 번 만들어지면 변하지 않는 "사실(fact)"이다. mutate 를 막아
"이 클립 결과 누가 언제 바꿨지"를 추적 불능으로 만드는 사고를 차단한다.
(TS 로 치면 readonly interface + Object.freeze.)

bbox 포맷은 [x, y, w, h] (좌상단 + 너비/높이) 로 통일 — architecture.md §5.2 가 SOT.
RF-DETR/supervision 은 xyxy(x1,y1,x2,y2) 를 쓰므로 detector 레이어에서 변환한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# [x, y, w, h] — 정수 픽셀 좌표
Bbox = list[int]


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """검출된 객체 1건. v0 은 COCO class name, v1+ 는 gecko/food_bowl/... 로 확장."""

    type: str
    confidence: float
    bbox: Bbox  # [x, y, w, h]
    frame_ts: float  # 이 객체가 검출된 프레임의 영상 내 초


@dataclass(frozen=True, slots=True)
class PrelabelResult:
    """한 클립의 prelabel 결과. PROJECT_PLAN §7 JSON 계약과 1:1 대응.

    dataclass 필드 순서는 "기본값 없는 것 먼저" 규칙 때문에 계약 순서와 다르다.
    출력 JSON 의 키 순서는 to_dict() 에서 계약 순서로 다시 맞춘다 (백엔드 가독성).
    """

    gecko_visible: bool
    visibility_confidence: float
    frames_sampled: int
    model_name: str
    model_version: str
    detected_objects: tuple[DetectedObject, ...] = ()
    best_frame_ts: float | None = None
    gecko_bbox: Bbox | None = None
    skip_recommendation: bool = False
    skip_reason: str | None = None
    clip_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """PROJECT_PLAN §7 계약 순서로 직렬화 가능한 dict 반환."""
        return {
            "clip_id": self.clip_id,
            "gecko_visible": self.gecko_visible,
            "visibility_confidence": self.visibility_confidence,
            "best_frame_ts": self.best_frame_ts,
            "gecko_bbox": self.gecko_bbox,
            "detected_objects": [asdict(o) for o in self.detected_objects],
            "skip_recommendation": self.skip_recommendation,
            "skip_reason": self.skip_reason,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "frames_sampled": self.frames_sampled,
        }
