"""compute_motion_metrics — 합성 프레임으로 motion 수치 계산 검증.

판정(active/static/...)은 activity_policy 몫이고, 여기선 수치가 물리적으로 맞는지만 본다:
정지 gecko=이동 0, 이동 gecko=이동>0, global 밝기변화 분리, ROI 내부변화 분리, edge 감지.
"""

from __future__ import annotations

import numpy as np

from gecko_vision_gate.motion_evidence import compute_motion_metrics
from gecko_vision_gate.schema import DetectedObject, PrelabelResult

H = W = 100


def _frame(val: int) -> np.ndarray:
    return np.full((H, W, 3), val, np.uint8)


def _gecko(ts: float, bbox: list[int], conf: float = 0.8) -> DetectedObject:
    return DetectedObject(type="gecko", confidence=conf, bbox=bbox, frame_ts=ts)


def _res(objs, frames_sampled: int) -> PrelabelResult:
    return PrelabelResult(
        gecko_visible=any(o.type == "gecko" for o in objs),
        visibility_confidence=0.8,
        frames_sampled=frames_sampled,
        model_name="rf-detr-nano",
        model_version="gecko_v2",
        detected_objects=tuple(objs),
    )


def test_empty_frames_returns_zeroed():
    m = compute_motion_metrics([], _res([], 0))
    assert m.visible_frame_count == 0
    assert m.max_bbox_center_disp == 0.0
    assert m.bbox_edge_clipped is False


def test_no_gecko_visible_count_zero():
    frames = [(0.0, _frame(50)), (1.0, _frame(50))]
    m = compute_motion_metrics(frames, _res([], frames_sampled=2))
    assert m.visible_frame_count == 0
    assert m.visible_frame_ratio == 0.0


def test_static_gecko_zero_displacement():
    frames = [(0.0, _frame(50)), (1.0, _frame(50))]  # 동일 프레임 → global 0
    objs = [_gecko(0.0, [30, 30, 20, 20]), _gecko(1.0, [30, 30, 20, 20])]  # 동일 bbox
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.visible_frame_count == 2
    assert m.max_bbox_center_disp == 0.0
    assert m.min_bbox_iou == 1.0
    assert m.global_bg_change == 0.0
    assert m.roi_flow_mag == 0.0


def test_moving_gecko_positive_displacement():
    frames = [(0.0, _frame(50)), (1.0, _frame(50))]
    objs = [_gecko(0.0, [10, 10, 20, 20]), _gecko(1.0, [60, 60, 20, 20])]  # 크게 이동, 안 겹침
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.max_bbox_center_disp > 0.1
    assert m.min_bbox_iou == 0.0


def test_global_brightness_change_captured_not_as_roi():
    # 전체 프레임 밝기가 확 변함(IR) → global_bg 큼, roi_flow 는 global 보정으로 0 근처
    frames = [(0.0, _frame(0)), (1.0, _frame(200))]
    objs = [_gecko(0.0, [30, 30, 20, 20]), _gecko(1.0, [30, 30, 20, 20])]
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.global_bg_change > 100
    assert m.roi_flow_mag == 0.0  # ROI 변화 == global 변화 → 분리 후 0


def test_local_roi_change_above_global():
    # 배경 고정, gecko ROI 안에서만 픽셀 변화 → roi_flow > 0, global 은 작음
    f0 = _frame(0)
    f1 = _frame(0).copy()
    f1[30:50, 30:50] = 200  # bbox 영역만 변화
    frames = [(0.0, f0), (1.0, f1)]
    objs = [_gecko(0.0, [30, 30, 20, 20]), _gecko(1.0, [30, 30, 20, 20])]
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.roi_flow_mag > 0.0
    assert m.roi_flow_mag > m.global_bg_change  # local 이 global 보다 큼


def test_edge_clipped_when_two_edges_touched():
    # 좌+상 두 변 동시 접촉(코너로 이탈) = 심하게 잘림
    frames = [(0.0, _frame(50)), (1.0, _frame(50))]
    objs = [_gecko(0.0, [0, 0, 20, 20]), _gecko(1.0, [0, 0, 20, 20])]
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.bbox_edge_clipped is True


def test_single_edge_touch_not_clipped():
    # 게코가 바닥(한 변)에만 붙음 = 온전히 보임 → 잘림 아님 (펫캠 흔한 정상 상태)
    frames = [(0.0, _frame(50)), (1.0, _frame(50))]
    objs = [_gecko(0.0, [40, 80, 20, 20]), _gecko(1.0, [40, 80, 20, 20])]  # y+h=100=H 하단만
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=2))
    assert m.bbox_edge_clipped is False


def test_visible_ratio_uses_frames_sampled():
    frames = [(float(i), _frame(50)) for i in range(3)]
    objs = [_gecko(0.0, [30, 30, 20, 20]), _gecko(1.0, [30, 30, 20, 20]), _gecko(2.0, [30, 30, 20, 20])]
    m = compute_motion_metrics(frames, _res(objs, frames_sampled=10))
    assert m.visible_frame_count == 3
    assert m.visible_frame_ratio == 0.3
