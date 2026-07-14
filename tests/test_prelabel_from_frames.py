"""prelabel_from_frames — 이미 샘플한 프레임으로 evidence 생성 (frames 재사용 경로).

worker 는 frames 를 한 번 샘플해서 prelabel(evidence)과 compute_motion_metrics(motion)에
같이 쓴다(중복 디코딩 방지). 그래서 sample_frames 를 안 거치는 진입점이 필요하다.
기존 prelabel_clip 은 이 함수에 sample_frames 를 앞에 붙인 얇은 wrapper 로 유지(하위호환).
"""

from __future__ import annotations

import numpy as np

from gecko_vision_gate.detector import RawDetection
from gecko_vision_gate.prelabel import prelabel_from_frames


class _FakeDetector:
    """detect() 를 프레임 순서대로 미리 정한 RawDetection 리스트로 반환 (모델 없이 로직 테스트)."""

    def __init__(self, per_frame):
        self._seq = list(per_frame)
        self._i = 0

    def detect(self, frame):
        out = self._seq[self._i] if self._i < len(self._seq) else []
        self._i += 1
        return out


def _f() -> np.ndarray:
    return np.zeros((10, 10, 3), np.uint8)


def test_gecko_visible_takes_max_confidence_frame():
    frames = [(0.0, _f()), (1.0, _f())]
    det = _FakeDetector([[RawDetection("gecko", 0.9, [1, 2, 3, 4])],
                         [RawDetection("gecko", 0.7, [5, 6, 7, 8])]])
    r = prelabel_from_frames(frames, detector=det, clip_id="c1")
    assert r.gecko_visible is True
    assert r.visibility_confidence == 0.9
    assert r.best_frame_ts == 0.0
    assert r.gecko_bbox == [1, 2, 3, 4]
    assert r.frames_sampled == 2
    assert len(r.detected_objects) == 2
    assert r.clip_id == "c1"


def test_absent_when_no_gecko_class():
    frames = [(0.0, _f()), (1.0, _f())]
    det = _FakeDetector([[], [RawDetection("cat", 0.9, [1, 2, 3, 4])]])
    r = prelabel_from_frames(frames, detector=det)
    assert r.gecko_visible is False
    assert r.skip_reason == "no_gecko_detected"
    assert r.best_frame_ts is None
    assert len(r.detected_objects) == 1  # non-gecko 도 evidence 로는 보존


def test_frames_sampled_reflects_input_length():
    frames = [(float(i), _f()) for i in range(5)]
    det = _FakeDetector([[] for _ in range(5)])
    r = prelabel_from_frames(frames, detector=det)
    assert r.frames_sampled == 5
    assert r.gecko_visible is False


def test_model_version_derived_from_checkpoint_path():
    frames = [(0.0, _f())]
    det = _FakeDetector([[RawDetection("gecko", 0.8, [0, 0, 2, 2])]])
    r = prelabel_from_frames(frames, detector=det, checkpoint="runs/gecko_v2/checkpoint_best_ema.pth")
    assert r.model_version == "gecko_v2 (checkpoint_best_ema)"
