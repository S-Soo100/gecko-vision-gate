"""전 영상 bounded Level 0/1 raw temporal evidence — 순차 디코딩 · 수치만.

**이 모듈은 판정하지 않는다.** 모든 `motion_clips` 영상을 순차 디코딩하면서 global/ROI grayscale
MAD(mean absolute difference) 시계열, 4×4 공간 dwell, numeric periodicity, motion excursion 을
숫자로만 낸다(설계 §7). 행동명(drinking/basking...)·selector·VLM 은 여기서 다루지 않는다.

핵심 계약:
  - Level 0 (모든 영상): 순차 디코딩, **전체 프레임 배열 보유 금지**(prev 축소 grayscale 1장 + bounded point 만).
  - Level 1 (bbox 있는 영상): sparse detected gecko box 들의 union ROI 내부 MAD 를 같은 timestamp 로 저장.
  - point cap 256: 시계열이 cap 을 넘으면 decimation(간격 stride 2배)으로 메모리 O(cap) 유지.
  - VideoCapture 는 예외 경로 포함 항상 release (donts/python#7).

TS 로 치면: 영상 스트림을 chunk 로 흘리며 이전 프레임 1장만 들고 diff 를 누적하는 reducer.
전체를 배열에 담지 않아 60초 4K 여도 메모리가 터지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .schema import PrelabelResult

# ── freeze shared names (Task 1 Step 3): nightly 가 이 상수를 import 해 literal 중복을 피한다 ──
EVIDENCE_SCHEMA_VERSION = "python-evidence-raw-v1"
ALGORITHM_VERSION = "croi-temporal-v1"
POINT_CAP = 256

TARGET_CLASS = "gecko"
# 분석용 축소 grayscale 의 긴 변 상한. 원본 해상도 무관하게 메모리/CPU 를 bound (설계 §7.1 "분석용 축소").
# no-upscale — 작은 소스는 그대로 둔다.
_ANALYSIS_LONG_EDGE = 256
_GRID = 4  # 4×4 spatial dwell
_MAX_DWELL_ATTR_SEC = 10.0  # sparse 관찰 1건이 dwell 로 배분받을 수 있는 절대 상한(초)


@dataclass(frozen=True, slots=True)
class TemporalPoint:
    """시계열 한 점 (t=영상 내 초, value=grayscale MAD). frozen = 사실."""

    t: float
    value: float


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    """한 clip 의 raw temporal evidence 묶음 (frozen, mutate 금지). append-only 원장으로 저장."""

    evidence_schema_version: str
    algorithm_version: str
    level0_status: str  # ok | no_decodable_frames | insufficient_decodable_frames | invalid_metadata
    level1_status: str  # ok | no_bbox | skipped
    decoded_frame_count: int
    point_stride: int
    global_motion_series: tuple[TemporalPoint, ...]
    roi_motion_series: tuple[TemporalPoint, ...]
    motion_summary: dict = field(default_factory=dict)
    spatial_dwell: dict = field(default_factory=dict)
    periodicity_summary: dict = field(default_factory=dict)
    motion_excursions: tuple[dict, ...] = ()


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _analysis_scale(w: int, h: int) -> float:
    """긴 변을 _ANALYSIS_LONG_EDGE 로 맞추는 축소 배율(no-upscale)."""
    long_edge = max(w, h)
    if long_edge <= _ANALYSIS_LONG_EDGE:
        return 1.0
    return _ANALYSIS_LONG_EDGE / float(long_edge)


def _union_gecko_bbox(result: PrelabelResult | None) -> list[int] | None:
    """sparse detected gecko box 들 + best gecko_bbox 의 union [x,y,w,h]. 없으면 None."""
    if result is None:
        return None
    boxes: list[list[int]] = [
        list(o.bbox) for o in result.detected_objects if o.type == TARGET_CLASS
    ]
    if result.gecko_bbox is not None:
        boxes.append(list(result.gecko_bbox))
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def _roi_in_analysis(bbox: list[int], scale: float, aw: int, ah: int) -> tuple[int, int, int, int] | None:
    """원본 bbox → 분석 grayscale 좌표계 ROI(clamp). 유효 영역 없으면 None(→ no_bbox)."""
    x = int(round(bbox[0] * scale))
    y = int(round(bbox[1] * scale))
    w = int(round(bbox[2] * scale))
    h = int(round(bbox[3] * scale))
    x1 = max(0, min(x, aw))
    y1 = max(0, min(y, ah))
    x2 = max(0, min(x + w, aw))
    y2 = max(0, min(y + h, ah))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


class _Decimator:
    """point cap 유지 스트리밍 저장기. cap 초과 시 stride 2배 + 기존 점 절반 솎기(O(cap) 메모리).

    consecutive diff 를 모두 계산하되(수치 CPU 만) 저장은 stride 간격으로만 → 균등 간격 downsample.
    """

    def __init__(self, cap: int):
        self.cap = cap
        self.stride = 1
        self._k = -1  # 0-based diff 카운터
        self.points: list[TemporalPoint] = []

    def offer(self, t: float, value: float) -> None:
        self._k += 1
        if self._k % self.stride != 0:
            return
        self.points.append(TemporalPoint(round(t, 3), round(value, 4)))
        if len(self.points) > self.cap:
            # stride 2배로 올리고 기존 저장점을 절반으로 → 새 stride 배수에 정렬(균등 유지)
            self.points = self.points[::2]
            self.stride *= 2


def _finite(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _series_stats(points: tuple[TemporalPoint, ...]) -> tuple[float, float, float]:
    """(mean, max, std) — 빈 시계열은 0."""
    if not points:
        return 0.0, 0.0, 0.0
    vals = np.array([p.value for p in points], dtype=float)
    return _finite(vals.mean()), _finite(vals.max()), _finite(vals.std())


def _spatial_dwell(result: PrelabelResult | None, w: int, h: int) -> dict:
    """sparse gecko 관찰의 4×4 정규화 dwell. 관찰 간극이 크면 unobserved_sec 로 뺀다(설계 §7.2).

    관찰 i(t_i, center_i) 와 다음 관찰의 시간 간격을 cell(center_i) 에 배분하되, 간극이
    `max_attribute_gap` 을 넘으면 dwell 이 아니라 unobserved 로 본다(게코가 안 잡힌 구간).
    """
    grid = [[0.0] * _GRID for _ in range(_GRID)]
    base = {"grid_size": _GRID, "observed_sec": 0.0, "unobserved_sec": 0.0,
            "cells": grid, "n_observations": 0}
    if result is None or w <= 0 or h <= 0:
        return base

    obs: list[tuple[float, int, int]] = []  # (ts, row, col)
    seen: dict[float, tuple[float, float]] = {}  # ts → (conf, center 를 위해) 최고 conf 만
    for o in result.detected_objects:
        if o.type != TARGET_CLASS:
            continue
        prev = seen.get(o.frame_ts)
        if prev is None or o.confidence > prev[0]:
            cx = (o.bbox[0] + o.bbox[2] / 2.0) / w
            cy = (o.bbox[1] + o.bbox[3] / 2.0) / h
            col = min(_GRID - 1, max(0, int(cx * _GRID)))
            row = min(_GRID - 1, max(0, int(cy * _GRID)))
            seen[o.frame_ts] = (o.confidence, 0.0)
            obs.append((o.frame_ts, row, col))
    if not obs:
        return base
    obs.sort(key=lambda x: x[0])
    base["n_observations"] = len(obs)

    # 관찰 간 시간 배분. 간극 상한 = 관찰 간격 중앙값의 3배(최소 2초)이되, **절대 상한 10초**로 clamp.
    # 절대 상한이 없으면 단일 관찰의 큰 간극이 스스로 median 이 돼 unobserved 를 못 낸다(sparse 관찰
    # 하나가 55초 dwell 을 주장하는 건 비현실적 — 그만큼은 미관찰로 본다).
    gaps = [obs[i + 1][0] - obs[i][0] for i in range(len(obs) - 1)]
    median_gap = float(np.median(gaps)) if gaps else 2.0
    max_attribute_gap = min(max(2.0, median_gap * 3.0), _MAX_DWELL_ATTR_SEC)

    observed = 0.0
    unobserved = 0.0
    for i, (ts, row, col) in enumerate(obs):
        if i < len(obs) - 1:
            dt = obs[i + 1][0] - ts
        else:
            dt = min(max_attribute_gap, median_gap)  # 마지막 관찰의 짧은 tail
        if dt <= max_attribute_gap:
            grid[row][col] += dt
            observed += dt
        else:
            grid[row][col] += min(dt, max_attribute_gap)
            observed += min(dt, max_attribute_gap)
            unobserved += dt - max_attribute_gap
    # 정규화 (관찰 시간 대비 셀 비율)
    if observed > 0:
        for r in range(_GRID):
            for c in range(_GRID):
                grid[r][c] = round(grid[r][c] / observed, 4)
    base["observed_sec"] = round(observed, 3)
    base["unobserved_sec"] = round(unobserved, 3)
    base["cells"] = grid
    return base


def _periodicity(points: tuple[TemporalPoint, ...], fps: float, stride: int) -> dict:
    """시계열 numeric autocorrelation 요약(행동명 아님). 지배 lag/peak 만.  n<4 면 null."""
    n = len(points)
    out = {"n_points": n, "dominant_lag_points": None,
           "dominant_lag_sec": None, "peak_autocorr": None}
    if n < 4:
        return out
    x = np.array([p.value for p in points], dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return out  # 상수 시계열 → 주기성 없음
    ac = np.correlate(x, x, mode="full")[n - 1:]  # lag 0..n-1
    ac = ac / denom
    lags = range(1, n // 2 + 1)
    best_lag = max(lags, key=lambda k: ac[k])
    out["dominant_lag_points"] = int(best_lag)
    out["peak_autocorr"] = round(_finite(ac[best_lag]), 4)
    if fps > 0:
        out["dominant_lag_sec"] = round(best_lag * stride / fps, 3)
    return out


def _excursions(points: tuple[TemporalPoint, ...], cap: int) -> tuple[dict, ...]:
    """mean+std 를 넘는 연속 구간을 raw excursion 으로 분절(수치, 행동명 아님). 최대 cap 개."""
    if len(points) < 2:
        return ()
    vals = np.array([p.value for p in points], dtype=float)
    thr = float(vals.mean() + vals.std())
    if not np.isfinite(thr) or vals.max() <= thr:
        return ()  # 변동 없음/전부 임계 이하
    segs: list[dict] = []
    cur: list[TemporalPoint] = []
    for p in points:
        if p.value > thr:
            cur.append(p)
        elif cur:
            segs.append(_seg(cur))
            cur = []
        if len(segs) >= cap:
            return tuple(segs[:cap])
    if cur:
        segs.append(_seg(cur))
    return tuple(segs[:cap])


def _seg(pts: list[TemporalPoint]) -> dict:
    vs = [p.value for p in pts]
    return {
        "start_t": pts[0].t,
        "end_t": pts[-1].t,
        "n_points": len(pts),
        "peak": round(max(vs), 4),
        "mean": round(sum(vs) / len(vs), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def compute_temporal_evidence(
    video_path: str | Path,
    result: PrelabelResult | None,
    *,
    point_cap: int = POINT_CAP,
    grid_size: int = _GRID,
) -> TemporalEvidence:
    """mp4 → Level 0 global + (bbox 있으면) Level 1 ROI raw temporal evidence.

    순차 디코딩 1회로 global/ROI MAD 를 같은 timestamp 로 뽑는다. 전체 프레임을 배열로 들지 않고
    이전 축소 grayscale 1장 + decimator(≤cap point) 만 유지한다.
    """
    path = Path(video_path).resolve()
    union_bbox = _union_gecko_bbox(result)

    g_dec = _Decimator(point_cap)
    r_dec = _Decimator(point_cap)
    local_diff_max = 0.0
    local_diff_sum = 0.0
    local_diff_n = 0
    roi_active = False  # 유효 ROI 로 실제 diff 를 저장했나

    cap = cv2.VideoCapture(str(path))
    decoded = 0
    orig_w = orig_h = 0
    fps = 30.0
    opened = False
    try:
        opened = cap.isOpened()
        if opened:
            f = cap.get(cv2.CAP_PROP_FPS)
            fps = float(f) if f and f > 0 else 30.0  # 깨진 메타 폴백
            prev_gray = None
            scale = 1.0
            roi = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                if orig_w == 0:
                    orig_h, orig_w = frame.shape[:2]
                    scale = _analysis_scale(orig_w, orig_h)
                    aw = max(1, int(round(orig_w * scale)))
                    ah = max(1, int(round(orig_h * scale)))
                    if union_bbox is not None:
                        roi = _roi_in_analysis(union_bbox, scale, aw, ah)
                # 분석용 축소 grayscale (BGR 원본은 보관 안 함)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if scale != 1.0:
                    gray = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)
                ts = (decoded - 1) / fps
                if prev_gray is not None and gray.shape == prev_gray.shape:
                    d = cv2.absdiff(gray, prev_gray)
                    g_val = float(d.mean())
                    g_dec.offer(ts, g_val)
                    if roi is not None:
                        rx, ry, rw, rh = roi
                        patch = d[ry:ry + rh, rx:rx + rw]
                        if patch.size:
                            roi_active = True
                            r_val = float(patch.mean())
                            r_dec.offer(ts, r_val)
                            ld = max(0.0, r_val - g_val)
                            local_diff_max = max(local_diff_max, ld)
                            local_diff_sum += ld
                            local_diff_n += 1
                prev_gray = gray
    finally:
        cap.release()

    # ── decode status ──
    if not opened:
        level0 = "invalid_metadata"
    elif decoded == 0:
        level0 = "no_decodable_frames"
    elif decoded == 1:
        level0 = "insufficient_decodable_frames"
    else:
        level0 = "ok"

    # ── level1 status ──
    if level0 != "ok":
        level1 = "skipped"
    elif union_bbox is None or not roi_active:
        level1 = "no_bbox"
    else:
        level1 = "ok"

    global_series = tuple(g_dec.points)
    roi_series = tuple(r_dec.points) if level1 == "ok" else ()
    point_stride = max(g_dec.stride, r_dec.stride)

    g_mean, g_max, g_std = _series_stats(global_series)
    r_mean, r_max, _ = _series_stats(roi_series)
    duration = round(decoded / fps, 3) if (decoded and fps > 0) else None

    motion_summary = {
        "duration_sec": duration,
        "fps": round(fps, 3),
        "decoded_frame_count": decoded,
        "width": orig_w or None,
        "height": orig_h or None,
        "global_mean": round(g_mean, 4),
        "global_max": round(g_max, 4),
        "global_std": round(g_std, 4),
        "roi_mean": round(r_mean, 4) if roi_series else None,
        "roi_max": round(r_max, 4) if roi_series else None,
        "local_diff_mean": round(local_diff_sum / local_diff_n, 4) if local_diff_n else None,
        "local_diff_max": round(local_diff_max, 4) if roi_series else None,
        "point_stride": point_stride,
    }

    # dwell/periodicity/excursion 은 ROI 시계열 우선, 없으면 global
    series_for_analysis = roi_series if roi_series else global_series
    dwell = _spatial_dwell(result, orig_w, orig_h)
    periodicity = _periodicity(series_for_analysis, fps, point_stride)
    excursions = _excursions(series_for_analysis, point_cap)

    return TemporalEvidence(
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        level0_status=level0,
        level1_status=level1,
        decoded_frame_count=decoded,
        point_stride=point_stride,
        global_motion_series=global_series,
        roi_motion_series=roi_series,
        motion_summary=motion_summary,
        spatial_dwell=dwell,
        periodicity_summary=periodicity,
        motion_excursions=excursions,
    )
