# R0003 — Frame Sampling bounded-memory 순차 fallback (indexed seek 실패 복구)

| | |
|---|---|
| **번호** | R0003 |
| **날짜** | 2026-07-16 |
| **상태** | ✅확정 |
| **분류** | **인프라 복구** (Gate v3 행동/모델 작업 아님) |
| **관련** | 커밋 `<이 커밋>` · petcam-lab handoff `s0-frame-sampling-self-healing` · 설계 `petcam-lab/docs/superpowers/specs/2026-07-16-s0-frame-sampling-self-healing-design.md` |

## 1. 배경·동기 (Context)

petcam-nightly-reporter `activity-worker` 가 만든 evidence 중 `frames_sampled < 6` 인 불완전
prelabel 이 production 에 누적됐다(진단 시점 24건, 매 cycle 증가). S0 coverage 감사가 이
`frames_sampled=0` 계약 위반으로 `S0_HOLD_DATA_CONTRACT` 판정을 냈고 S1 이 보류됐다.

**이 리포트는 detector·정확도·행동 판정을 바꾸지 않는다.** Gate 의 역할(프레임 복구)만
고친다. "게코가 잡히나"를 판정하려면 프레임이 실제로 디코딩돼야 하는데, 현재 sampler 가
프레임을 놓치고도 빈 list 를 정상 반환하던 인프라 결함을 메운다.

## 2. 근본 원인 (실측)

petcam-lab 에서 불완전 clip 24건을 read-only 로 다운로드해 decode 진단:

| 지표 | 값 |
|---|---|
| `CAP_PROP_FRAME_COUNT` | 32~58 (0 아님) |
| indexed read (`set(POS_FRAMES)`+read, 12 target) | **0~5장만 성공** |
| 순차 디코딩 가능 프레임 | **12~46장 (전부 ≥ 6)** |
| root cause | **24/24 `indexed_seek_failure`** |
| 순차 fallback 복구 가능 | **24/24 (permanent 0)** |

원인: ~7fps sparse-keyframe H.264(≈5.5s) 에서 OpenCV `POS_FRAMES` 인덱스 seek 이 비-keyframe
에 착지해 `read()` 가 실패. 순차 디코딩은 정상. 설계가 상정한 `CAP_PROP_FRAME_COUNT=0` 은
실제 원인이 아니었지만, 동일 fallback(설계 §4.2 "indexed 결과 < 요청")이 두 경우를 모두 커버.

## 3. 변경 (Method)

`src/gecko_vision_gate/frame_sampling.py`:

- 기본 index sampling 경로는 **byte-for-byte 불변** — 정상 영상은 fallback 미발동(프레임
  index·timestamp·성능 그대로).
- `len(out) < num_frames` 이면 2-pass 순차 fallback:
  1. `_count_sequential_frames(path)` — 배열 미보관, 디코딩 가능 프레임 수만 카운트 (O(1) 메모리).
  2. `_sample_sequential(path, total, num_frames, fps)` — `evenly_spaced_indices` 목표 인덱스만
     최대 `num_frames` 개 보관 (O(num_frames) 메모리). 절대 전체 프레임을 적재하지 않는다.
- 모든 `VideoCapture` 는 `finally` 에서 release. fps 무효 시 기존대로 30fps.
- fallback 이 기존보다 많이 복구할 때만 교체.

## 4. 검증 (Results)

fake `VideoCapture`(seek 실패 vs 순차 성공 분리 재현)로 단위 테스트. 바이너리 fixture 없음.

| 테스트 | 목적 | 결과 |
|---|---|---|
| `test_zero_metadata_count_falls_back_to_sequential_decode` | count=0 → 순차 12장 | ✅ |
| `test_indexed_read_shortfall_falls_back_to_sequential_decode` | seek 3장 → 순차 12장 | ✅ |
| `test_normal_metadata_path_does_not_open_fallback_passes` | 정상 영상 fallback 미발동(캡처 1회) | ✅ |
| `test_fallback_keeps_at_most_requested_frames` | 500 디코딩 가능해도 출력 12 (bounded) | ✅ |
| `test_every_video_capture_is_released_on_success_and_failure` | 성공·실패 모두 release | ✅ |

- `uv run pytest -q tests/test_frame_sampling.py` → 9 passed
- `uv run pytest -q` (full) → 65 passed
- `git diff --check` → clean

## 5. 한계·다음 (Limits / Next)

- fallback 이후에도 6프레임 미만이면 Gate 는 있는 만큼만 반환한다. 이 clip 을 production
  완료로 인정할지는 **nightly `assess_clip` 의 최소 프레임 barrier** 책임(설계 §4.3, 별도 레포).
- 순차 count pass 는 read 횟수가 O(전체 프레임)이지만 메모리는 O(1). 5초 clip 기준 비용 무시 가능.
- 영구 손상 파일의 무한 재시도 제어(failure ledger)는 현재 범위 밖(설계 대안 C, 필요 시 별도 spec).
