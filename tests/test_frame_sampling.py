"""frame_sampling 순수 로직 + fallback 테스트.

`evenly_spaced_indices` 는 영상 I/O 없는 인덱스 계산만.
`sample_frames` fallback 은 fake VideoCapture 로 메타데이터/seek 이상을 재현한다
(바이너리 fixture 를 git 에 넣지 않기 위함).
"""

import cv2
import numpy as np
import pytest

from gecko_vision_gate.frame_sampling import evenly_spaced_indices, sample_frames


def _frame():
    # 작은 더미 BGR 프레임 (실제 픽셀값은 fallback 로직과 무관)
    return np.zeros((4, 4, 3), dtype=np.uint8)


class FakeCapture:
    """indexed seek 실패 vs 순차 디코딩 성공을 분리 재현하는 fake.

    - `seekable`: `set(POS_FRAMES, idx)` 직후 read 가 성공하는 idx 집합 (indexed 경로)
    - `sequential_count`: set 없이 순차 read 로 디코딩 가능한 프레임 수 (fallback 경로)
    """

    def __init__(self, meta_count, seekable, sequential_count, opened=True, fps=7.0):
        self.meta_count = meta_count
        self.seekable = set(seekable)
        self.sequential_count = sequential_count
        self._opened = opened
        self._fps = fps
        self.released = False
        self._pos = 0
        self._seek_pending = False
        self._seq_cursor = 0

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.meta_count)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(val)
            self._seek_pending = True
        return True

    def read(self):
        if not self._opened:
            return False, None
        if self._seek_pending:  # indexed 경로
            self._seek_pending = False
            return (True, _frame()) if self._pos in self.seekable else (False, None)
        idx = self._seq_cursor  # 순차 경로
        if idx < self.sequential_count:
            self._seq_cursor += 1
            return True, _frame()
        return False, None

    def release(self):
        self.released = True


class CaptureFactory:
    """같은 영상(path)을 재오픈할 때마다 동일 설정의 fake 를 새로 발급 + 인스턴스 추적."""

    def __init__(self, **cfg):
        self.cfg = cfg
        self.instances = []

    def __call__(self, path):
        cap = FakeCapture(**self.cfg)
        self.instances.append(cap)
        return cap


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00")  # exists() 통과용 (fake 는 내용 무시)
    return p


def _patch(monkeypatch, **cfg):
    factory = CaptureFactory(**cfg)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return factory


def test_zero_metadata_count_falls_back_to_sequential_decode(monkeypatch, video):
    # CAP_PROP_FRAME_COUNT=0 → indexed 경로는 0장 → 순차 fallback 이 20장 중 12장 복구
    _patch(monkeypatch, meta_count=0, seekable=set(), sequential_count=20)
    out = sample_frames(video, num_frames=12)
    assert len(out) == 12


def test_indexed_read_shortfall_falls_back_to_sequential_decode(monkeypatch, video):
    # 메타데이터는 39지만 seek 은 3장만 성공 → 순차 fallback 이 30장 중 12장 복구
    _patch(monkeypatch, meta_count=39, seekable={0, 1, 2}, sequential_count=30)
    out = sample_frames(video, num_frames=12)
    assert len(out) == 12
    assert len(out) > 3


def test_normal_metadata_path_does_not_open_fallback_passes(monkeypatch, video):
    # 정상 영상: indexed 12장 성공 → fallback 미발동(캡처 1회만 오픈)
    factory = _patch(monkeypatch, meta_count=100, seekable=set(range(100)), sequential_count=100)
    out = sample_frames(video, num_frames=12)
    assert len(out) == 12
    assert len(factory.instances) == 1  # 추가 pass 없음


def test_fallback_keeps_at_most_requested_frames(monkeypatch, video):
    # 500장 순차 디코딩 가능해도 출력은 num_frames 로 bounded
    _patch(monkeypatch, meta_count=0, seekable=set(), sequential_count=500)
    out = sample_frames(video, num_frames=12)
    assert len(out) == 12  # 500 이 아니라 12 (bounded output)


def test_every_video_capture_is_released_on_success_and_failure(monkeypatch, video):
    # 성공(fallback 3 capture) 전부 release
    factory = _patch(monkeypatch, meta_count=0, seekable=set(), sequential_count=20)
    sample_frames(video, num_frames=12)
    assert factory.instances and all(c.released for c in factory.instances)
    # 실패(open 실패) 시에도 release
    fail = _patch(monkeypatch, meta_count=0, seekable=set(), sequential_count=0, opened=False)
    with pytest.raises(RuntimeError):
        sample_frames(video, num_frames=12)
    assert fail.instances and all(c.released for c in fail.instances)


def test_basic_count_and_range():
    idxs = evenly_spaced_indices(total_frames=100, num_samples=12)
    assert len(idxs) == 12
    assert idxs == sorted(idxs)
    assert idxs[0] == 0 and idxs[-1] == 99  # 양끝 포함
    assert all(0 <= i < 100 for i in idxs)


def test_short_video_dedups_to_total():
    # 요청(12)보다 프레임(5)이 적으면 중복 제거되어 total 개
    assert evenly_spaced_indices(total_frames=5, num_samples=12) == [0, 1, 2, 3, 4]


def test_empty_or_invalid():
    assert evenly_spaced_indices(0, 12) == []
    assert evenly_spaced_indices(-1, 12) == []


def test_single_frame():
    assert evenly_spaced_indices(1, 12) == [0]
