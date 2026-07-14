"""activity_policy.decide — four-state 제품 판정 순수 로직 테스트.

Gate 는 evidence(gecko bbox/visibility)만 만들고, 여기 policy 가 제품 판정을 내린다
(책임 분리: petcam 지시문 §97). decide 는 PrelabelResult(evidence) + MotionMetrics(motion)
+ ActivityPolicy(versioned 임계값) 를 받아 active/exclude_absent/exclude_static/unknown 을 낸다.

fail-open 원칙: 애매하면 unknown (활동시간을 줄이지 않는 쪽).
"""

from dataclasses import FrozenInstanceError

import pytest

from gecko_vision_gate.activity_policy import ActivityAssessment, ActivityPolicy, decide
from gecko_vision_gate.motion_evidence import MotionMetrics
from gecko_vision_gate.schema import PrelabelResult

# 임계값은 Phase 3 사람 GT 로 확정 예정. 여기선 로직 검증용 고정값.
POLICY = ActivityPolicy(version="test-v0", gate_threshold=0.25)


def _result(*, gecko_visible: bool = True, frames_sampled: int = 12) -> PrelabelResult:
    return PrelabelResult(
        gecko_visible=gecko_visible,
        visibility_confidence=0.8 if gecko_visible else 0.0,
        frames_sampled=frames_sampled,
        model_name="rf-detr-nano",
        model_version="gecko_v2 (checkpoint_best_ema)",
    )


def _mm(**kw) -> MotionMetrics:
    """기본은 'gecko 안정적으로 보이고 안 움직임'(static). 케이스별로 override."""
    base = dict(
        visible_frame_count=10,
        visible_frame_ratio=0.9,
        max_bbox_center_disp=0.0,
        max_bbox_size_change=0.0,
        min_bbox_iou=1.0,
        roi_flow_mag=0.0,
        global_bg_change=0.0,
        bbox_edge_clipped=False,
    )
    base.update(kw)
    return MotionMetrics(**base)


# --- absent -------------------------------------------------------------
def test_absent_when_no_gecko_and_enough_frames():
    a = decide(_result(gecko_visible=False), _mm(visible_frame_count=0, visible_frame_ratio=0.0), POLICY)
    assert a.decision == "exclude_absent"
    assert a.reason_code == "no_gecko_detected"


# --- unknown (fail-open) ------------------------------------------------
def test_unknown_when_insufficient_frames():
    # decode/sample 실패로 프레임이 적으면 absent 로 몰지 않고 unknown
    a = decide(_result(gecko_visible=False, frames_sampled=3), _mm(), POLICY)
    assert a.decision == "unknown"
    assert a.reason_code == "insufficient_frames"


def test_unknown_on_sparse_detection():
    # gecko 가 한두 프레임만 잡히면 static 아님 → unknown
    a = decide(_result(), _mm(visible_frame_count=2, visible_frame_ratio=0.15), POLICY)
    assert a.decision == "unknown"
    assert a.reason_code == "sparse_detection"


def test_unknown_on_edge_clipped():
    a = decide(_result(), _mm(bbox_edge_clipped=True), POLICY)
    assert a.decision == "unknown"
    assert a.reason_code == "bbox_edge_clipped"


def test_unknown_on_ir_brightness_only():
    # 전체 밝기(IR)만 변하고 local motion 없음 → static 으로 오판 금지, unknown
    a = decide(_result(), _mm(global_bg_change=10.0), POLICY)
    assert a.decision == "unknown"
    assert a.reason_code == "global_motion_ambiguous"


def test_unknown_on_whole_camera_shake():
    # 카메라 전체 흔들림: bbox 도 크게 이동하지만 global 이 지배적 → active 아님, unknown
    a = decide(_result(), _mm(global_bg_change=10.0, max_bbox_center_disp=0.3), POLICY)
    assert a.decision == "unknown"


# --- active -------------------------------------------------------------
def test_active_when_bbox_moves():
    a = decide(_result(), _mm(max_bbox_center_disp=0.2), POLICY)
    assert a.decision == "active"
    assert a.reason_code == "motion_observed"


def test_active_on_micro_head_tongue_flow():
    # bbox 는 거의 안 움직여도 ROI 내부 flow 가 있으면 active
    # (미세 혀/머리 움직임을 빠르다는 이유로 무시하지 않는다: 지시문 §229)
    a = decide(_result(), _mm(roi_flow_mag=5.0), POLICY)
    assert a.decision == "active"


def test_active_on_bbox_iou_drop():
    a = decide(_result(), _mm(min_bbox_iou=0.4), POLICY)
    assert a.decision == "active"


# --- static -------------------------------------------------------------
def test_static_when_visible_and_stable():
    a = decide(_result(), _mm(), POLICY)  # 전부 0, visibility 0.9
    assert a.decision == "exclude_static"
    assert a.reason_code == "static_confirmed"


def test_borderline_low_visibility_is_unknown():
    # 안 움직이지만 visibility 도 낮으면 static 확신 못함 → unknown
    a = decide(_result(), _mm(visible_frame_count=4, visible_frame_ratio=0.35), POLICY)
    assert a.decision == "unknown"


# --- contract -----------------------------------------------------------
def test_assessment_is_frozen():
    a = decide(_result(), _mm(), POLICY)
    assert isinstance(a, ActivityAssessment)
    with pytest.raises(FrozenInstanceError):
        a.decision = "x"  # type: ignore[misc]


def test_measurements_snapshot_policy_version_and_metrics():
    a = decide(_result(), _mm(max_bbox_center_disp=0.2), POLICY)
    assert a.measurements["policy_version"] == POLICY.version
    assert a.measurements["max_bbox_center_disp"] == 0.2


def test_policy_threshold_injection_changes_outcome():
    # 같은 motion 도 policy 임계값에 따라 판정이 달라진다(임계값 주입 검증)
    strict = ActivityPolicy(version="strict", gate_threshold=0.25, move_center_disp=0.5)
    lenient = ActivityPolicy(version="lenient", gate_threshold=0.25, move_center_disp=0.01)
    mm = _mm(max_bbox_center_disp=0.1)
    assert decide(_result(), mm, strict).decision == "exclude_static"  # 0.1 < 0.5 → 안 움직인 걸로
    assert decide(_result(), mm, lenient).decision == "active"          # 0.1 > 0.01 → 움직인 걸로
