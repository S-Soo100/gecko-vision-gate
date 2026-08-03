from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gecko_vision_gate.gme_contracts import Detection, GMEConfig
from gecko_vision_gate.gme_engine import analyze_clip


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float):
        self.frames = list(frames)
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 5:  # cv2.CAP_PROP_FPS
            return self.fps
        if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
            return len(self.frames)
        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


@dataclass
class FakeDetector:
    calls: list[float]
    model_name: str = "fake"
    model_version: str = "v1"
    checkpoint_sha256: str = "a" * 64
    schema_version: str = "detector-v1"
    threshold: float = 0.5

    def detect(self, frame_bgr, timestamp_sec):
        self.calls.append(timestamp_sec)
        return (Detection(timestamp_sec, (10, 10, 20, 20), 0.9, "gecko"),)


def test_engine_decodes_every_frame_but_detector_uses_half_second_anchors(monkeypatch):
    # tracker가 신뢰도를 유지하는 정적 texture. feature가 없는 검은 화면이면 설계대로 즉시 재검출한다.
    rng = np.random.default_rng(7)
    textured = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
    frames = [textured.copy() for _ in range(31)]
    cap = FakeCapture(frames, 30.0)
    detector = FakeDetector([])
    monkeypatch.setattr("gecko_vision_gate.gme_engine.cv2.VideoCapture", lambda _: cap)
    result = analyze_clip("clip.mp4", detector=detector, config=GMEConfig(anchor_interval_sec=0.5))
    assert result.decoded_frame_count == 31
    assert detector.calls == [0.0, 0.5, 1.0]
    assert cap.released is True


def test_engine_limits_analysis_clock_to_30fps(monkeypatch):
    frames = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(61)]
    cap = FakeCapture(frames, 60.0)
    detector = FakeDetector([])
    monkeypatch.setattr("gecko_vision_gate.gme_engine.cv2.VideoCapture", lambda _: cap)
    result = analyze_clip("clip.mp4", detector=detector)
    assert result.decoded_frame_count == 61
    assert result.analyzed_frame_count == 31


def test_engine_releases_capture_and_marks_empty_video_terminal(monkeypatch):
    cap = FakeCapture([], 30.0)
    detector = FakeDetector([])
    monkeypatch.setattr("gecko_vision_gate.gme_engine.cv2.VideoCapture", lambda _: cap)
    result = analyze_clip("empty.mp4", detector=detector)
    assert result.status == "no_decodable_frames"
    assert cap.released is True


def test_exposure_jump_is_unknown_quality_region_not_gecko_movement(monkeypatch):
    rng = np.random.default_rng(9)
    dark = rng.integers(0, 80, (80, 100, 3), dtype=np.uint8)
    bright = np.clip(dark.astype(np.int16) + 170, 0, 255).astype(np.uint8)
    cap = FakeCapture([dark, bright, bright.copy()], 2.0)
    detector = FakeDetector([])
    monkeypatch.setattr("gecko_vision_gate.gme_engine.cv2.VideoCapture", lambda _: cap)
    result = analyze_clip("clip.mp4", detector=detector, config=GMEConfig(anchor_interval_sec=0.5))
    assert result.unknown_sec >= 0.5
    assert any(row.get("exposure_change") is True for row in result.frame_debug)
    assert detector.calls == [0.0, 1.0]
