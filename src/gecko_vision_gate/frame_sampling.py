"""mp4 → 균등 간격 프레임 샘플링.

Gate v0 는 적응형이 아니라 단순 균등 N장 (MODEL_AND_TRAINING_PLAN §6 Step 6).
적응형 시간밀도 레시피는 lab/nightly 의 VLM 입력용이고, Gate detector 의 목적은
"게코가 한 프레임이라도 잡히나"라서 균등 커버리지로 충분하다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def evenly_spaced_indices(total_frames: int, num_samples: int) -> list[int]:
    """0..total-1 을 num_samples 개로 균등 분할한 프레임 인덱스.

    np.linspace 로 양끝 포함 균등 분할. 짧은 영상에서 반올림 충돌로 같은 인덱스가
    나올 수 있어 set 으로 중복 제거한다. (순수 함수라 단위 테스트 대상)
    """
    if total_frames <= 0:
        return []
    n = min(num_samples, total_frames)
    idxs = np.linspace(0, total_frames - 1, n).round().astype(int)
    return sorted({int(i) for i in idxs})


def sample_frames(video_path: str | Path, num_frames: int = 12) -> list[tuple[float, np.ndarray]]:
    """영상에서 균등 간격 프레임을 (timestamp_sec, frame_bgr) 리스트로 반환.

    petcam donts/python.md 반영:
      7. cap.release() 를 try/finally 로 보장 (스레드 누수 방지)
      10. fps/총프레임을 런타임 확인 — 소스마다 다르므로 가정하지 않는다
      11. 절대경로 사용
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video (codec?): {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 메타데이터가 깨진 영상 폴백

        out: list[tuple[float, np.ndarray]] = []
        for idx in evenly_spaced_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue  # 깨진 프레임은 스킵 (로컬 파일이라 재시도 불필요)
            ts = round(idx / fps, 2)
            out.append((ts, frame))
        return out
    finally:
        cap.release()
