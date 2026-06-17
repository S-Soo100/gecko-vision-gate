"""clip_prelabels JSON 계약 테스트 — 키/순서/중첩 직렬화/불변성."""

from dataclasses import FrozenInstanceError

import pytest

from gecko_vision_gate.schema import DetectedObject, PrelabelResult

# PROJECT_PLAN §7 계약 키 순서 (백엔드가 이 순서를 기대)
CONTRACT_KEYS = [
    "clip_id",
    "gecko_visible",
    "visibility_confidence",
    "best_frame_ts",
    "gecko_bbox",
    "detected_objects",
    "skip_recommendation",
    "skip_reason",
    "model_name",
    "model_version",
    "frames_sampled",
]


def _make() -> PrelabelResult:
    obj = DetectedObject(type="gecko", confidence=0.9, bbox=[1, 2, 3, 4], frame_ts=1.5)
    return PrelabelResult(
        gecko_visible=True,
        visibility_confidence=0.9,
        frames_sampled=12,
        model_name="rf-detr-nano",
        model_version="v0-coco",
        detected_objects=(obj,),
        best_frame_ts=1.5,
        gecko_bbox=[1, 2, 3, 4],
    )


def test_to_dict_has_contract_keys_in_order():
    assert list(_make().to_dict().keys()) == CONTRACT_KEYS


def test_nested_detected_objects_serialized_to_plain_dict():
    d = _make().to_dict()
    assert d["detected_objects"] == [
        {"type": "gecko", "confidence": 0.9, "bbox": [1, 2, 3, 4], "frame_ts": 1.5}
    ]


def test_result_is_frozen():
    r = _make()
    with pytest.raises(FrozenInstanceError):
        r.gecko_visible = False  # type: ignore[misc]
