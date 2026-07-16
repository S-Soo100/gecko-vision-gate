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


def _count_sequential_frames(path: Path) -> int:
    """set(POS_FRAMES) 없이 순차 read 로 디코딩 가능한 프레임 수만 센다.

    frame 배열은 보관하지 않으므로 메모리는 O(1) (영상 길이 무관). indexed seek 이
    실패해도(sparse keyframe H.264 등) 순차 디코딩은 성공하는 경우가 있어, 실제
    복구 가능한 프레임 수를 먼저 확정한다.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0
        count = 0
        while True:
            ok, _frame = cap.read()  # 배열 미보관(bounded memory)
            if not ok:
                break
            count += 1
        return count
    finally:
        cap.release()


def _sample_sequential(
    path: Path, total: int, num_frames: int, fps: float
) -> list[tuple[float, np.ndarray]]:
    """순차 read 로 목표 인덱스의 프레임만 최대 num_frames 개 보관한다.

    `evenly_spaced_indices(total, num_frames)` 로 목표를 정하고, 순차 커서가 그 인덱스에
    닿을 때만 배열을 보관한다. 목표 수를 채우면 즉시 중단하므로 메모리는 O(num_frames).
    """
    targets = set(evenly_spaced_indices(total, num_frames))
    if not targets:
        return []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return []
        out: list[tuple[float, np.ndarray]] = []
        idx = 0
        while len(out) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in targets:
                out.append((round(idx / fps, 2), frame))
            idx += 1
        return out
    finally:
        cap.release()


def sample_frames(video_path: str | Path, num_frames: int = 12) -> list[tuple[float, np.ndarray]]:
    """영상에서 균등 간격 프레임을 (timestamp_sec, frame_bgr) 리스트로 반환.

    기본 경로는 메타데이터 기반 균등 index sampling(정상 영상은 이 경로만 사용하며 프레임
    index·timestamp·성능 불변). 메타데이터 frame count 가 0이거나 indexed read 가 요청 수보다
    적으면 bounded-memory 순차 디코딩 fallback 으로 실제 프레임을 복구한다(설계 §4).

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
    finally:
        cap.release()

    # 정상 경로가 요청 수를 채웠으면 추가 pass 없이 반환(기존 동작 불변).
    if len(out) >= num_frames:
        return out

    # fallback: metadata count 0 또는 indexed 부족. 순차 디코딩으로 복구 시도.
    seq_total = _count_sequential_frames(path)
    if seq_total <= 0:
        return out  # 순차로도 디코딩 불가 → 있는 만큼(빈 list 가능) 반환
    fallback = _sample_sequential(path, seq_total, num_frames, fps)
    return fallback if len(fallback) > len(out) else out
