"""프레임 + prelabel evidence → 정지/활동 판정용 motion 수치 (순수 OpenCV/numpy).

**이 모듈은 판정하지 않는다.** gecko bbox trajectory·ROI 내부 변화·global 배경 변화를
수치로만 뽑는다. four-state 제품 판정은 activity_policy.decide 가 이 수치를 소비해서 한다
(Gate evidence 생성 ↔ 제품 정책 decision 분리: 지시문 §97·§119).

bbox 중심 이동 하나로 static 을 확정하지 않기 위해(지시문 §213·§436) 여러 신호를 조합한다:
  - bbox center 이동 / 면적 변화 / 연속 IoU  (게코 몸통 이동)
  - gecko ROI 내부 grayscale 변화에서 global 배경 변화를 뺀 값  (혀·머리 미세 움직임)
  - global 배경 변화  (IR auto-exposure·카메라 흔들림을 local motion 과 구분)
  - visibility frame count/ratio  (sparse detection 을 static 으로 오판 방지)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import DetectedObject, PrelabelResult

TARGET_CLASS = "gecko"


@dataclass(frozen=True, slots=True)
class MotionMetrics:
    """한 클립의 motion 수치 스냅샷 (frozen = 사실, mutate 금지)."""

    visible_frame_count: int      # gecko 가 검출된 프레임 수
    visible_frame_ratio: float    # visible / frames_sampled
    max_bbox_center_disp: float   # 연속 visible 프레임 bbox 중심 이동 최대 (프레임 대각선으로 정규화)
    max_bbox_size_change: float   # 연속 visible 프레임 bbox 면적 변화율 최대
    min_bbox_iou: float           # 연속 visible 프레임 bbox IoU 최소 (낮을수록 이동)
    roi_flow_mag: float           # gecko ROI 내부 grayscale 변화 − global (0 clamp)
    global_bg_change: float       # 전체 프레임 연속 grayscale 절대차 평균
    bbox_edge_clipped: bool       # gecko bbox 가 프레임 가장자리에 심하게 붙음


def _center(b: list[int]) -> tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def _area(b: list[int]) -> float:
    return float(max(0, b[2]) * max(0, b[3]))


def _iou(a: list[int], b: list[int]) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _edge_clipped(b: list[int], w: int, h: int, margin_frac: float) -> bool:
    """여러 변(2+)에 동시 접촉 = 게코가 코너/화면 밖으로 잘려나감 → 판단 신뢰 불가.

    한 변 접촉(바닥/벽에 붙은 게코)은 온전히 보이는 정상 상태라 잘림으로 보지 않는다.
    펫캠은 게코가 바닥 경계(y+h≈H)에 흔히 붙어 있어 한 변 접촉을 잘림으로 보면 전부 unknown 된다.
    """
    mx, my = w * margin_frac, h * margin_frac
    touches = (
        int(b[0] <= mx)
        + int(b[1] <= my)
        + int((b[0] + b[2]) >= (w - mx))
        + int((b[1] + b[3]) >= (h - my))
    )
    return touches >= 2


def _expand_roi(b: list[int], w: int, h: int, scale: float = 1.5) -> tuple[int, int, int, int]:
    cx, cy = _center(b)
    nw, nh = b[2] * scale, b[3] * scale
    x1 = int(max(0, cx - nw / 2))
    y1 = int(max(0, cy - nh / 2))
    x2 = int(min(w, cx + nw / 2))
    y2 = int(min(h, cy + nh / 2))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def compute_motion_metrics(
    frames: list[tuple[float, np.ndarray]],
    result: PrelabelResult,
    *,
    edge_margin_frac: float = 0.02,
    roi_scale: float = 1.5,
) -> MotionMetrics:
    """frames(=[(ts, bgr)]) + prelabel evidence → MotionMetrics. 판정 안 함, 수치만."""
    if not frames:
        return MotionMetrics(0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, False)

    h, w = frames[0][1].shape[:2]
    diag = float(np.hypot(w, h)) or 1.0

    # 프레임별 최고 conf gecko bbox
    gecko_by_ts: dict[float, DetectedObject] = {}
    for o in result.detected_objects:
        if o.type != TARGET_CLASS:
            continue
        prev = gecko_by_ts.get(o.frame_ts)
        if prev is None or o.confidence > prev.confidence:
            gecko_by_ts[o.frame_ts] = o

    visible_count = len(gecko_by_ts)
    ratio = visible_count / max(1, result.frames_sampled)

    # bbox trajectory (ts 순 연속쌍)
    boxes = [gecko_by_ts[ts].bbox for ts in sorted(gecko_by_ts)]
    max_disp = max_size = 0.0
    min_iou = 1.0
    for a, b in zip(boxes, boxes[1:]):
        ca, cb = _center(a), _center(b)
        max_disp = max(max_disp, float(np.hypot(cb[0] - ca[0], cb[1] - ca[1])) / diag)
        aa, ab = _area(a), _area(b)
        if max(aa, ab) > 0:
            max_size = max(max_size, abs(aa - ab) / max(aa, ab))
        min_iou = min(min_iou, _iou(a, b))

    edge = any(_edge_clipped(o.bbox, w, h, edge_margin_frac) for o in gecko_by_ts.values())

    # global 배경 변화 + gecko ROI 내부 변화 (grayscale 절대차; optical flow 는 v0 과잉)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames]
    ts_list = [ts for ts, _ in frames]
    global_diffs: list[float] = []
    roi_diffs: list[float] = []
    for i in range(1, len(grays)):
        d = cv2.absdiff(grays[i], grays[i - 1])
        global_diffs.append(float(d.mean()))
        o = gecko_by_ts.get(ts_list[i]) or gecko_by_ts.get(ts_list[i - 1])
        if o is not None:
            rx, ry, rw, rh = _expand_roi(o.bbox, w, h, roi_scale)
            patch = d[ry:ry + rh, rx:rx + rw]
            if patch.size:
                roi_diffs.append(float(patch.mean()))

    global_bg = float(np.mean(global_diffs)) if global_diffs else 0.0
    roi_flow = max(0.0, (float(np.mean(roi_diffs)) - global_bg)) if roi_diffs else 0.0

    return MotionMetrics(
        visible_frame_count=visible_count,
        visible_frame_ratio=round(ratio, 3),
        max_bbox_center_disp=round(max_disp, 4),
        max_bbox_size_change=round(max_size, 4),
        min_bbox_iou=round(min_iou, 4),
        roi_flow_mag=round(roi_flow, 3),
        global_bg_change=round(global_bg, 3),
        bbox_edge_clipped=edge,
    )
