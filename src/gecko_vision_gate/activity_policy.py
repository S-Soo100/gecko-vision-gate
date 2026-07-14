"""four-state 활동 판정 — Gate evidence 를 소비하는 제품 정책 (순수 함수).

Gate(detector/prelabel) 는 evidence 만 만들고, 이 모듈이 제품 판정을 내린다.
DB 에서도 clip_prelabels(evidence) ↔ clip_activity_assessments(decision) 로 분리한다
(지시문 §97). 모든 임계값은 versioned ActivityPolicy 하나에서 주입한다 — 코드 상수로
숨기지 않는다(지시문 §231). 실제 수치는 Phase 3 dry-run + 사람 GT 로 확정한다.

fail-open 원칙: 애매하면 unknown → 활동시간을 줄이지 않는다(지시문 §89).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .motion_evidence import MotionMetrics
from .schema import PrelabelResult

DECISIONS = ("active", "exclude_absent", "exclude_static", "unknown")


@dataclass(frozen=True, slots=True)
class ActivityPolicy:
    """활동 판정 임계값 묶음 (versioned). Phase 3 에서 수치 확정."""

    version: str
    gate_threshold: float             # evidence 추출 시 detector threshold (기록/재현용)
    min_frames: int = 6               # 이보다 적으면 decode/sample 부족 → unknown
    sparse_min_visible: int = 3       # gecko visible 프레임이 이보다 적으면 sparse → unknown
    move_center_disp: float = 0.03    # bbox 중심 이동(대각선 정규화) 이상이면 활동 후보
    move_iou_max: float = 0.85        # 연속 IoU 가 이 이하로 떨어지면 활동 후보
    move_size_change: float = 0.15    # bbox 면적 변화율 이상이면 활동 후보
    roi_flow_active: float = 2.0      # ROI 내부 변화(global 보정 후) 이상이면 활동(혀/머리)
    global_motion_max: float = 3.0    # global 변화 이상이면 local/global 구분 불가 → unknown
    static_min_visible_ratio: float = 0.6   # static 확정 최소 visibility ratio
    absent_max_global_motion: float = 3.0   # no_gecko 여도 global 이 이 이상이면 absent 단정 말고 unknown


@dataclass(frozen=True, slots=True)
class ActivityAssessment:
    """clip 1건의 제품 판정 (frozen). measurements = 판정 근거 스냅샷."""

    decision: str
    reason_code: str
    measurements: dict[str, Any]


def _assess(decision: str, reason: str, result: PrelabelResult,
            motion: MotionMetrics, policy: ActivityPolicy) -> ActivityAssessment:
    return ActivityAssessment(
        decision=decision,
        reason_code=reason,
        measurements={
            "policy_version": policy.version,
            "gate_threshold": policy.gate_threshold,
            "gecko_visible": result.gecko_visible,
            "frames_sampled": result.frames_sampled,
            "visible_frame_count": motion.visible_frame_count,
            "visible_frame_ratio": motion.visible_frame_ratio,
            "max_bbox_center_disp": motion.max_bbox_center_disp,
            "max_bbox_size_change": motion.max_bbox_size_change,
            "min_bbox_iou": motion.min_bbox_iou,
            "roi_flow_mag": motion.roi_flow_mag,
            "global_bg_change": motion.global_bg_change,
            "bbox_edge_clipped": motion.bbox_edge_clipped,
        },
    )


def decide(result: PrelabelResult, motion: MotionMetrics, policy: ActivityPolicy) -> ActivityAssessment:
    """evidence + motion + policy → four-state 판정. 순수 함수."""
    # 1. 프레임 부족(decode/sample 실패 신호) → unknown (absent 로 몰지 않는다)
    if result.frames_sampled < policy.min_frames:
        return _assess("unknown", "insufficient_frames", result, motion, policy)

    # 2. gecko detection 0건 + 충분한 프레임 → absent 후보
    if not result.gecko_visible:
        # 하드닝 5: 게코 미검출이어도 global motion 이 크면 threshold 아래 게코/움직임이 있을 수 있어
        # absent 단정은 위험(audit 실증: 0.25 no_gecko 가 실제 active) → unknown fail-open.
        if motion.global_bg_change >= policy.absent_max_global_motion:
            return _assess("unknown", "no_gecko_but_global_motion", result, motion, policy)
        return _assess("exclude_absent", "no_gecko_detected", result, motion, policy)

    # --- 여기부터 gecko visible ---
    # 3. 한두 프레임만 잡힘 → static 아님, unknown
    if motion.visible_frame_count < policy.sparse_min_visible:
        return _assess("unknown", "sparse_detection", result, motion, policy)

    # 4. bbox 가 프레임 가장자리에 심하게 잘림 → 이동/정지 신뢰 불가, unknown
    if motion.bbox_edge_clipped:
        return _assess("unknown", "bbox_edge_clipped", result, motion, policy)

    global_dominant = motion.global_bg_change >= policy.global_motion_max
    moved = (
        motion.max_bbox_center_disp >= policy.move_center_disp
        or motion.min_bbox_iou <= policy.move_iou_max
        or motion.max_bbox_size_change >= policy.move_size_change
        or motion.roi_flow_mag >= policy.roi_flow_active
    )

    # 5. 활동 신호가 global 흔들림/밝기와 분리돼 신뢰 가능 → active (미세 움직임도 존중)
    if moved and not global_dominant:
        return _assess("active", "motion_observed", result, motion, policy)

    # 6. global 이 지배적이면 local motion 판단 불가(IR·흔들림) → unknown
    if global_dominant:
        return _assess("unknown", "global_motion_ambiguous", result, motion, policy)

    # 7. 안정적으로 보이고 활동 없음 → static
    if motion.visible_frame_ratio >= policy.static_min_visible_ratio:
        return _assess("exclude_static", "static_confirmed", result, motion, policy)

    # 8. 안 움직이지만 visibility 낮음 → 확신 못함, unknown
    return _assess("unknown", "borderline_low_visibility", result, motion, policy)
