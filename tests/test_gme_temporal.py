from __future__ import annotations

import pytest

from gecko_vision_gate.gme_contracts import Detection
from gecko_vision_gate.gme_temporal import AnalysisClock, TemporalDetectionGate


def _detection(timestamp_sec: float) -> Detection:
    return Detection(timestamp_sec, (10.0, 10.0, 20.0, 20.0), 0.9, "gecko")


def test_25fps_uses_absolute_10fps_deadline_grid():
    clock = AnalysisClock(max_analysis_fps=10.0)

    selected = [index for index in range(26) if clock.accept(index, 25.0)]

    assert selected == [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25]


def test_two_of_five_is_suppressed_and_third_positive_is_accepted():
    gate = TemporalDetectionGate(window_frames=5, min_positive_frames=3)

    assert gate.push((_detection(0.0),)) == ()
    assert gate.push(()) == ()
    assert gate.push((_detection(0.2),)) == ()
    assert gate.push((_detection(0.3),)) == (_detection(0.3),)


@pytest.mark.parametrize(
    ("window_frames", "min_positive_frames"),
    [(0, 0), (5, 0), (5, 6)],
)
def test_temporal_gate_rejects_invalid_window_contract(window_frames, min_positive_frames):
    with pytest.raises(ValueError):
        TemporalDetectionGate(
            window_frames=window_frames,
            min_positive_frames=min_positive_frames,
        )


def test_reset_discards_pre_exposure_positive_history():
    gate = TemporalDetectionGate(window_frames=5, min_positive_frames=3)
    gate.push((_detection(0.0),))
    gate.push((_detection(0.1),))

    gate.reset()

    assert gate.push((_detection(0.2),)) == ()
