# R0004 — Gecko Motion Engine production shadow v0 코어

> 상태: ✅확정
> 날짜: 2026-08-03

## 1. 목적

기존 sparse Python Evidence가 하지 못한 프레임별 게코 검출·추적·카메라 변화 분리와 실제 움직인
시간 후보 계산을 하나의 순차 처리 엔진으로 만든다. 결과는 사용자 정답이 아니라 `candidate` shadow다.

## 2. 결정

- 원본 30fps 이하는 모든 프레임을 분석하고, 그 이상은 분석 시계를 최대 30fps로 제한한다.
- detector anchor는 기본 0.5초, tracker 신뢰도 하락 시 즉시 재검출한다.
- 여러 게코는 별도 track으로 유지하고 identity를 억지로 연결하지 않는다.
- 점 provenance를 `observed/tracked/interpolated/unknown`으로 분리한다.
- 양쪽 anchor가 있는 1.0초 이하 gap만 보간한다.
- 상태는 `moving/static/not_visible/unknown/camera_motion` 계약을 쓰며 검출 실패는 `unknown`이다.
- 사용자 시간은 any-gecko interval union, 내부 연구값은 gecko-seconds로 분리한다.

## 3. 구현

`gme_contracts`, `gme_tracker`, `gme_motion`, `gme_engine`, `gme_serialization`, detector adapter와
`gecko-gme` CLI를 추가했다. 전체 프레임 배열은 보관하지 않고 직전 프레임과 bounded track state만
유지한다. artifact는 canonical JSON+gzip+SHA-256으로 재현한다.

## 4. 검증

- 전체 Gate 회귀: `104 passed`
- 합성 프레임: 다중 개체 association, optical-flow failure 재검출, 짧은 gap 보간, 카메라 이동,
  노출 급변, any-gecko/gecko-seconds, 최대 30fps 분석, capture release를 검증했다.
- 실제 영상 정확도 성적은 이 라운드에서 측정하지 않았다.

## 5. 한계와 위험

- v2 detector는 사람 blind future holdout을 아직 통과하지 않았다.
- bbox 몸길이는 실제 몸길이의 proxy라 자세·가림에 취약하다.
- sparse LK tracker는 빠른 자세변화·IR 노이즈·긴 가림에서 fragmentation이 생길 수 있다.
- `not_visible`을 확정할 독립 absent calibration이 없으므로 현 v0의 무검출은 `unknown`이다.

## 6. 채택 경계

운영 queue/R2/DB에 저장해도 사용자 활동시간, 행동 GT, VLM route, 자동 skip, 원본 삭제에는 사용하지
않는다. 사람 time-interval+bbox/mask GT와 camera/animal/enclosure/video 분리 future holdout을 통과한
뒤에만 verified 지표 승격을 논의한다.
