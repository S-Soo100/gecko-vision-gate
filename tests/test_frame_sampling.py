"""frame_sampling 순수 로직 테스트 (영상 I/O 없는 인덱스 계산만).

실제 mp4 추출은 모델 의존 E2E 로 검증하고, 여기선 균등분할 경계를 못 박는다.
"""

from gecko_vision_gate.frame_sampling import evenly_spaced_indices


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
