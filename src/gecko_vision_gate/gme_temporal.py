"""GME의 최대 분석 FPS와 짧은 탐지 지속성 계약."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .gme_contracts import Detection


@dataclass(slots=True)
class AnalysisClock:
    """원본 FPS와 무관하게 절대 시간축에서 최대 분석 FPS를 지킨다."""

    max_analysis_fps: float
    _next_deadline_number: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_analysis_fps) or self.max_analysis_fps <= 0:
            raise ValueError("max_analysis_fps must be finite and positive")

    def accept(self, frame_index: int, source_fps: float) -> bool:
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("source_fps must be finite and positive")
        timestamp_sec = frame_index / source_fps
        deadline_sec = self._next_deadline_number / self.max_analysis_fps
        if timestamp_sec + 1e-12 < deadline_sec:
            return False
        self._next_deadline_number += 1
        return True


class TemporalDetectionGate:
    """최근 N개 분석 frame에서 충분히 반복된 현재 탐지만 통과시킨다."""

    def __init__(self, *, window_frames: int, min_positive_frames: int) -> None:
        if not isinstance(window_frames, int) or isinstance(window_frames, bool) or window_frames <= 0:
            raise ValueError("window_frames must be a positive integer")
        if (
            not isinstance(min_positive_frames, int)
            or isinstance(min_positive_frames, bool)
            or not 1 <= min_positive_frames <= window_frames
        ):
            raise ValueError("min_positive_frames must be inside the window")
        self._positive_history: deque[bool] = deque(maxlen=window_frames)
        self._minimum = min_positive_frames

    def push(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        self._positive_history.append(bool(detections))
        return detections if sum(self._positive_history) >= self._minimum else ()

    def reset(self) -> None:
        self._positive_history.clear()
