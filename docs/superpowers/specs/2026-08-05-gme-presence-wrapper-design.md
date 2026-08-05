# GME Presence Wrapper 설계

## 목표

기존 `analyze_clip()` 결과를 촬영 서버가 안전하게 소비할 수 있는 3상태 계약으로 변환한다.
이 wrapper는 게코 부재를 확정하거나 원본을 삭제하지 않는다.

## 공개 계약

- `detected_candidate`: 정상 분석에서 `provenance="observed"`인 직접 관측점이 하나 이상 있다.
- `not_observed`: 정상 분석·분석 프레임 존재·직접 관측 없음·unknown 없음·camera motion 없음이다.
- `unresolved`: 분석 실패, 분석 프레임 없음, 직접 관측이 없는 상태에서 unknown 또는 camera motion이 있다.

`not_observed`는 “게코 없음”이 아니라 “현재 모델에서 직접 관측되지 않음”이다.

## 구조

- `gme_presence.py`: 순수 판정 함수, 영상 분석 wrapper, Gate detector 편의 함수, 직렬화 가능한 결과 계약을 담당한다.
- `test_gme_presence.py`: 상태 우선순위, fail-closed 처리, detector metadata, 경로 비노출을 검증한다.
- `README.md`: 서버 개발자가 복사할 수 있는 최소 Python 예제를 제공한다.

## 데이터 흐름

`video path → GateDetectorAdapter → analyze_clip → decide_presence → PresenceResult`

결과에는 schema/version, 이유 코드, detector/checkpoint identity, threshold, frame 수, 관측점 수,
moving/unknown/camera-motion 시간만 담는다. 입력 경로·credential·R2 key는 담지 않는다.

## 비목표

- 행동명·GT·하이라이트 결정
- VLM 호출 skip/route
- A/B 원본 삭제 또는 영구 격리
- threshold 자동 튜닝

