"""compute_temporal_evidence — bounded sequential Level 0/1 raw temporal evidence.

frame_sampling 테스트와 같은 방식으로 fake VideoCapture 를 monkeypatch 해 실제 mp4/codec 없이
디코딩 시퀀스를 재현한다(바이너리 fixture 를 git 에 안 넣음). 이 모듈은 **판정하지 않는다** —
global/ROI grayscale MAD 시계열·dwell·periodicity·excursion 을 수치로만 낸다(설계 §7).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gecko_vision_gate.temporal_evidence import (
    ALGORITHM_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    POINT_CAP,
    compute_temporal_evidence,
)
from gecko_vision_gate.schema import DetectedObject, PrelabelResult

H = W = 120


def _frame(val: int) -> np.ndarray:
    return np.full((H, W, 3), val, np.uint8)


class FakeCap:
    """순차 read 로 정한 프레임 시퀀스를 돌려주는 fake. None 프레임 = 디코딩 종료(끊김)."""

    def __init__(self, frames, *, fps=10.0, opened=True, meta_count=None):
        self.frames = frames
        self.fps = fps
        self._opened = opened
        self.meta_count = meta_count if meta_count is not None else len(frames)
        self._i = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.meta_count)
        return 0.0

    def read(self):
        if self._i >= len(self.frames):
            return False, None
        f = self.frames[self._i]
        self._i += 1
        if f is None:
            return False, None
        return True, f

    def set(self, *_a):
        return True

    def release(self):
        self.released = True


class _Factory:
    def __init__(self, cap):
        self.cap = cap
        self.made = []

    def __call__(self, _path):
        self.made.append(self.cap)
        return self.cap


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00")  # exists() 통과용
    return p


def _patch(monkeypatch, cap):
    f = _Factory(cap)
    monkeypatch.setattr(cv2, "VideoCapture", f)
    return f


def _res(objs, frames_sampled=12, bbox=None):
    return PrelabelResult(
        gecko_visible=bool(objs),
        visibility_confidence=0.8,
        frames_sampled=frames_sampled,
        model_name="rf-detr-nano",
        model_version="gecko_v2",
        detected_objects=tuple(objs),
        gecko_bbox=bbox,
    )


def _gecko(ts, bbox, conf=0.8):
    return DetectedObject(type="gecko", confidence=conf, bbox=bbox, frame_ts=ts)


def _all_finite(seq):
    return all(math.isfinite(v) for v in seq)


# --- versioning / contract names (Task 1 Step 3 freeze) ---

def test_frozen_shared_constants():
    assert EVIDENCE_SCHEMA_VERSION == "python-evidence-raw-v1"
    assert ALGORITHM_VERSION == "croi-temporal-v1"
    assert POINT_CAP == 256


# --- Level 0 decode statuses ---

def test_no_decodable_frames(monkeypatch, video):
    _patch(monkeypatch, FakeCap([]))
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "no_decodable_frames"
    assert ev.level1_status == "skipped"
    assert ev.global_motion_series == ()
    assert ev.roi_motion_series == ()
    assert ev.decoded_frame_count == 0


def test_one_frame_insufficient(monkeypatch, video):
    _patch(monkeypatch, FakeCap([_frame(50)]))
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "insufficient_decodable_frames"
    assert ev.decoded_frame_count == 1
    assert ev.global_motion_series == ()
    assert ev.level1_status == "skipped"


def test_cannot_open_invalid_metadata(monkeypatch, video):
    cap = FakeCap([], opened=False)
    f = _patch(monkeypatch, cap)
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "invalid_metadata"
    assert cap.released is True  # 열기 실패해도 release
    assert f.made  # capture 는 만들어졌다


def test_no_bbox_returns_complete_global_evidence(monkeypatch, video):
    # gecko 없음(result None) 이어도 Level 0 global evidence 는 완전
    frames = [_frame(0), _frame(40), _frame(0), _frame(40)]
    _patch(monkeypatch, FakeCap(frames))
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "ok"
    assert len(ev.global_motion_series) == 3  # 4프레임 → 3 diff
    assert ev.level1_status == "no_bbox"
    assert ev.roi_motion_series == ()
    assert all(p.value > 0 for p in ev.global_motion_series)


def test_invalid_bbox_falls_back_to_no_bbox(monkeypatch, video):
    frames = [_frame(0), _frame(30), _frame(0)]
    _patch(monkeypatch, FakeCap(frames))
    # bbox 가 프레임 밖 + 0 크기 → 유효 ROI 없음
    res = _res([_gecko(0.0, [999, 999, 0, 0])])
    ev = compute_temporal_evidence(video, res)
    assert ev.level1_status == "no_bbox"
    assert ev.roi_motion_series == ()


def test_global_lighting_change_captured(monkeypatch, video):
    frames = [_frame(0), _frame(200)]  # 전체 밝기 급변
    _patch(monkeypatch, FakeCap(frames))
    ev = compute_temporal_evidence(video, None)
    assert len(ev.global_motion_series) == 1
    assert ev.global_motion_series[0].value > 100
    assert ev.motion_summary["global_max"] > 100


def test_roi_local_change_above_global(monkeypatch, video):
    # 배경 고정, bbox 내부만 변화 → roi 시계열 > 0, local_diff_max > 0
    f0 = _frame(0)
    f1 = _frame(0).copy()
    f1[30:60, 30:60] = 220  # bbox 영역만
    frames = [f0, f1]
    _patch(monkeypatch, FakeCap(frames))
    res = _res([_gecko(0.0, [30, 30, 30, 30]), _gecko(0.1, [30, 30, 30, 30])])
    ev = compute_temporal_evidence(video, res)
    assert ev.level1_status == "ok"
    assert len(ev.roi_motion_series) == 1
    assert ev.roi_motion_series[0].value > 0
    assert ev.motion_summary["local_diff_max"] > 0


def test_constant_series_zero_motion_no_excursions(monkeypatch, video):
    frames = [_frame(50)] * 5
    _patch(monkeypatch, FakeCap(frames))
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "ok"
    assert all(p.value == 0 for p in ev.global_motion_series)
    assert ev.motion_excursions == ()


def test_frame_shape_change_does_not_crash(monkeypatch, video):
    # 중간에 해상도 다른 프레임 → 그 diff 는 건너뛰되 크래시 없이 완주
    frames = [_frame(0), np.full((60, 60, 3), 40, np.uint8), _frame(0), _frame(40)]
    _patch(monkeypatch, FakeCap(frames))
    ev = compute_temporal_evidence(video, None)
    assert ev.level0_status == "ok"
    assert ev.decoded_frame_count == 4
    # 최소한 shape 일치 쌍(마지막 0→40)에서 motion 이 잡힌다
    assert any(p.value > 0 for p in ev.global_motion_series)


def test_point_cap_enforced_with_stride(monkeypatch, video):
    # cap 초과 프레임 → 저장 point ≤ cap, stride > 1 (bounded memory decimation)
    rng = list(range(POINT_CAP * 2 + 50))
    frames = [_frame(v % 256) for v in rng]
    _patch(monkeypatch, FakeCap(frames))
    ev = compute_temporal_evidence(video, None)
    assert len(ev.global_motion_series) <= POINT_CAP
    assert ev.point_stride >= 2


def test_metadata_frame_count_lie_uses_actual(monkeypatch, video):
    # 메타는 100000장이라 우기지만 실제 디코딩은 3장 → decoded_frame_count = 실제값
    frames = [_frame(0), _frame(30), _frame(0)]
    _patch(monkeypatch, FakeCap(frames, meta_count=100000))
    ev = compute_temporal_evidence(video, None)
    assert ev.decoded_frame_count == 3
    assert ev.level0_status == "ok"


def test_dwell_unobserved_time_for_large_gaps(monkeypatch, video):
    # sparse 관찰 사이 큰 시간 간극 → unobserved_sec > 0
    frames = [_frame(0), _frame(20)]
    _patch(monkeypatch, FakeCap(frames))
    res = _res([_gecko(0.0, [10, 10, 20, 20]), _gecko(55.0, [10, 10, 20, 20])])
    ev = compute_temporal_evidence(video, res)
    assert ev.spatial_dwell["grid_size"] == 4
    assert ev.spatial_dwell["unobserved_sec"] > 0


def test_all_outputs_finite(monkeypatch, video):
    f1 = _frame(0).copy()
    f1[30:60, 30:60] = 200
    frames = [_frame(0), f1, _frame(10), _frame(60)]
    _patch(monkeypatch, FakeCap(frames))
    res = _res([_gecko(0.0, [30, 30, 30, 30]), _gecko(0.3, [30, 30, 30, 30])])
    ev = compute_temporal_evidence(video, res)
    assert _all_finite(p.value for p in ev.global_motion_series)
    assert _all_finite(p.value for p in ev.roi_motion_series)
    assert _all_finite(p.t for p in ev.global_motion_series)
    assert all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in ev.motion_summary.values() if isinstance(v, (int, float)))


def test_capture_released_on_success(monkeypatch, video):
    cap = FakeCap([_frame(0), _frame(30)])
    _patch(monkeypatch, cap)
    compute_temporal_evidence(video, None)
    assert cap.released is True


def test_capture_released_on_exception(monkeypatch, video):
    cap = FakeCap([_frame(0), _frame(30), _frame(0)])
    _patch(monkeypatch, cap)

    # 내부 처리 중 예외 강제 → try/finally 로 반드시 release (donts/python#7)
    def boom(*_a, **_k):
        raise RuntimeError("cvt boom")

    monkeypatch.setattr(cv2, "cvtColor", boom)
    with pytest.raises(RuntimeError):
        compute_temporal_evidence(video, None)
    assert cap.released is True


def test_evidence_carries_version_fields(monkeypatch, video):
    _patch(monkeypatch, FakeCap([_frame(0), _frame(30)]))
    ev = compute_temporal_evidence(video, None)
    assert ev.evidence_schema_version == EVIDENCE_SCHEMA_VERSION
    assert ev.algorithm_version == ALGORITHM_VERSION


# ── H1: CROI 해상도 계약 — ROI 는 원본 crop 후 별도 bounded resize (전체 축소로 소실 금지) ──

def test_croi_micro_roi_change_survives_on_1080p(monkeypatch, video):
    """1920×1080 화면의 24×24 ROI 미세 변화가 전체 화면 256px 축소로 소실되지 않는다.

    전체 프레임만 256px 로 줄이면 24px→3px 로 뭉개져 ROI 변화가 사라진다(기존 회귀). ROI 는 원본에서
    bbox 를 먼저 crop 한 뒤 별도 resize 하므로 crop(24×24, no-upscale)에서 변화가 온전히 보존된다.
    """
    f0 = np.zeros((1080, 1920, 3), np.uint8)
    f1 = f0.copy()
    f1[500:524, 900:924] = 220  # 정확히 24×24 영역만 급변
    _patch(monkeypatch, FakeCap([f0, f1]))
    res = _res([_gecko(0.0, [900, 500, 24, 24]), _gecko(0.1, [900, 500, 24, 24])])
    ev = compute_temporal_evidence(video, res)
    assert ev.level1_status == "ok"
    assert len(ev.roi_motion_series) == 1
    # crop 24×24 전체가 0→220 이므로 ROI MAD 는 크다(≈220). 전체 축소였다면 ~0.06 로 소실됐을 값.
    assert ev.roi_motion_series[0].value > 150
    # global 은 24×24/(1920×1080) 로 극히 희석 → ROI 가 global 보다 압도적으로 큼
    assert ev.roi_motion_series[0].value > ev.global_motion_series[0].value * 50
    assert ev.motion_summary["local_diff_max"] > 150


def test_croi_tiny_bbox_not_lost_to_subpixel(monkeypatch, video):
    """1080p 화면의 작은 bbox(3×3)가 전체 축소 sub-pixel 반올림으로 ROI 자체를 잃지 않는다.

    전체 프레임을 256px 로 줄이면 3×3 bbox 는 analysis 좌표계에서 0px 로 반올림돼 ROI 가 사라진다
    (level1=no_bbox). crop-first 는 원본에서 3×3 을 그대로 crop 하므로 ROI evidence 를 보존한다.
    """
    f0 = np.zeros((1080, 1920, 3), np.uint8)
    f1 = f0.copy()
    f1[500:503, 900:903] = 220  # 3×3 영역만 변화
    _patch(monkeypatch, FakeCap([f0, f1]))
    res = _res([_gecko(0.0, [900, 500, 3, 3]), _gecko(0.1, [900, 500, 3, 3])])
    ev = compute_temporal_evidence(video, res)
    assert ev.level1_status == "ok"  # 구(전체 축소) 구현은 no_bbox 로 소실
    assert len(ev.roi_motion_series) == 1
    assert ev.roi_motion_series[0].value > 150  # 3×3 crop 전체 변화 보존


def test_croi_roi_resize_no_upscale(monkeypatch, video):
    # 작은 bbox(20×20)는 upscale 하지 않는다 — resize 로 인한 가짜 신호 없음. 상수 crop → ROI MAD 0.
    frames = [np.full((300, 300, 3), 40, np.uint8), np.full((300, 300, 3), 40, np.uint8)]
    _patch(monkeypatch, FakeCap(frames))
    res = _res([_gecko(0.0, [10, 10, 20, 20]), _gecko(0.1, [10, 10, 20, 20])])
    ev = compute_temporal_evidence(video, res)
    assert ev.level1_status == "ok"
    assert ev.roi_motion_series[0].value == 0.0  # 변화 없음 → upscale 노이즈도 없음


# ── H6: dwell dedup(같은 ts 최고 conf 1건) + fps fallback 명시 ──

def test_dwell_dedups_same_timestamp_to_highest_conf(monkeypatch, video):
    frames = [_frame(0), _frame(20)]
    _patch(monkeypatch, FakeCap(frames))
    # ts=0.0 에 detection 2건(conf 0.5, 0.9) + ts=1.0 1건 → 관찰은 ts 당 1개 = 2개
    objs = [_gecko(0.0, [10, 10, 20, 20], conf=0.5), _gecko(0.0, [80, 80, 20, 20], conf=0.9),
            _gecko(1.0, [10, 10, 20, 20], conf=0.7)]
    res = _res(objs)
    ev = compute_temporal_evidence(video, res)
    assert ev.spatial_dwell["n_observations"] == 2  # 3건 아님(같은 ts dedup)


def test_fps_fallback_flag(monkeypatch, video):
    # fps 메타가 깨짐(<=0) → 30 폴백 + fps_fallback True
    _patch(monkeypatch, FakeCap([_frame(0), _frame(30)], fps=0.0))
    ev = compute_temporal_evidence(video, None)
    assert ev.motion_summary["fps"] == 30.0
    assert ev.motion_summary["fps_fallback"] is True
    # 정상 fps → False
    _patch(monkeypatch, FakeCap([_frame(0), _frame(30)], fps=10.0))
    ev2 = compute_temporal_evidence(video, None)
    assert ev2.motion_summary["fps_fallback"] is False
