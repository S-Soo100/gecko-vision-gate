# Gecko Vision Gate v3 — 데이터 다양성·사람 GT·안전한 활용 계획

**상태:** v3 설계 확정 / v3 학습·배포 전 / v2 `activity-v1` shadow evidence 축적 중
**작성일:** 2026-07-12
**목적:** 여러 카메라·개체·사육환경에서 게코의 존재와 위치를 안정적으로 기록하는 evidence sensor를 만든다.

## 0. 2026-07-14 운영 연결과 신뢰 경계

`petcam-nightly-reporter`가 v2 best-EMA와 threshold 0.10을 사용한 `activity-v1` worker를 테스트 카메라 3대에서 1시간마다 실행한다. 이 단계는 **v3 배포가 아니라 v3를 위한 운영 evidence·hardcase 축적**이다. Gate 결과와 motion evidence는 `clip_prelabels`와 `clip_activity_assessments`에 분리 저장된다. Flutter `v0.20.1+35`는 effective activity view를 읽고, 카메라 A 한 대의 visible-static만 가역 canary로 차감한다. 전 카메라 absent와 다른 두 카메라 static은 비활성이다.

- **운영 파이프라인은 신뢰:** 버전 guard, checkpoint/threshold/sampler provenance, 첫 200건 무실패, 임시 파일 정리, fail-open을 확인했다.
- **자동 제외 정확도는 미신뢰:** no-gecko는 active 2/12를 놓쳐 REJECT, visible-static은 독립 에피소드가 약 2개뿐이라 정식 판정은 HOLD다. 카메라 A canary는 채택 승격이 아니라 사용자 위험 수용 실험이다.
- **v3에 환류:** 카메라 B recall hardcase 2건과 카메라 A visible-static calibration 사례를 분리 보존한다. product `exclude` 라벨을 presence detector GT로 사용하지 않는다.
- **승격 전제:** 최소 3개 날짜·30분 episode dedup·독립 static 20개 이상 사람 blind GT에서 false exclusion 0을 확인하기 전까지 정식 채택·VLM skip·미검증 카메라 확장은 금지한다. false exclusion 1건이면 카메라 A canary도 즉시 rollback한다.

## 1. 한 줄 결정

v3는 행동 분류기가 아니며 VLM을 즉시 차단하는 gate도 아니다. 모든 clip에 `gecko visible / bbox / best frame / trajectory` evidence를 붙이는 상시 prelabeler로 먼저 shadow 운영하고, 독립 future holdout에서 false negative 위험이 충분히 낮아진 뒤에만 별도 router가 제한적으로 비용 절감에 사용한다.

## 2. v2에서 확인한 사실과 불일치

### gate 레포 자체 평가

R0002의 동일 test 300(frame 230 positive/70 negative, clip 45 positive/23 negative)에서 v2 `checkpoint_best_ema.pth`는 다음 결과를 냈다.

- threshold 0.25: clip recall 97.8%, FP 5/23
- threshold 0.5: clip recall 93.3%, FP 3/23
- 데이터는 여전히 같은 카메라·사육장·릴리화이트 개체 중심이다.

### petcam backlog 평가

petcam-lab의 2026-07-07 backlog 300 평가는 Claude 판정을 proxy GT로 사용했고 recall 90.9%, specificity 40.0%, score=0 FN 후보 20건을 기록했다. 그러나 이 평가는 R0002 권장 `checkpoint_best_ema.pth`가 아니라 `checkpoint_best_regular.pth`를 사용했다.

따라서 현재 수치만으로 v2 일반화 실패나 v3 재학습 효과를 확정하지 않는다. **첫 단계는 artifact와 사람 GT를 통일한 v2 최종 감사**다.

근거:

- `reports/R0002-evening-recall-v2.md`
- petcam-lab `experiments/gate-recall/TEST-SHEET.md`, `experiments/gate-recall/REPORT.md`

## 3. v3 목표와 비목표

### 목표

- 카메라·개체·모프·사육장·시간대가 달라도 게코 존재를 놓치지 않는다.
- 부분 노출·원거리·IR·유리 반사 환경에서 bbox와 best frame을 공급한다.
- bbox trajectory와 camera ROI를 결합할 수 있는 안정적인 좌표 evidence를 만든다.
- 사람이 빠르게 교정할 수 있는 prelabel을 라벨링 웹에 제공한다.

### 비목표

- drinking, feeding, shedding 등 행동을 확정하지 않는다.
- v3 자체가 `cloud_now`, `activity_only`, VLM skip을 결정하지 않는다.
- gecko presence가 일반화되기 전에 bowl/prey/shed/hand-tool 다중 클래스로 확장하지 않는다.
- 같은 backlog를 보고 학습한 뒤 다시 독립 test라고 부르지 않는다.

## 4. Phase 0 — v2 최종 감사

1. `checkpoint_best_ema.pth`, frame sampler, frames=12를 고정하고 checkpoint SHA-256을 기록한다.
2. petcam backlog 300 전체를 재추론하고 threshold 0.10~0.70 curve를 저장한다.
3. 라벨링 웹에서 Claude·detector 결과를 숨긴 상태로 300 clip 전체를 사람이 `visible / absent / uncertain`으로 검수한다.
4. detector가 놓치거나 잘못 잡은 프레임에는 bbox를 추가·삭제·교정한다.
5. camera, individual/morph, enclosure, color/IR, occlusion, edge, distance, glare metadata를 기록한다.
6. 사람 GT 기준 recall, specificity, FP/FN 유형과 bbox 품질을 다시 계산한다.

68개 불일치만 검수하면 두 모델이 함께 틀린 232개를 발견할 수 없으므로 300개 전체를 검수한다. 이 감사 결과는 새 report로 남긴다.

## 5. Phase 1 — v3 학습 데이터

### 다양성 축

- 여러 카메라와 렌즈, 설치 높이·거리·각도
- 여러 개체와 밝은/어두운/패턴 모프
- 서로 다른 사육장·식물·코르크·은신처 배경
- 주간 컬러·저녁·야간 IR
- 머리/꼬리만 보임, 사물 뒤 가림, 프레임 가장자리, 원거리 소형 개체
- 유리 반사·물방울·IR 글레어·그림자·흰 장식물

### paired hard negative

같은 카메라·같은 구도에서 게코가 있는 장면과 없는 장면을 함께 수집한다. positive만 늘려 recall을 올리면 식물·은신처·글레어 오발동이 다시 증가하므로 hard positive와 hard negative를 한 라운드에서 같이 검수한다.

### 데이터 역할

- 외부/크롤 데이터: train 또는 보조 val만 사용한다.
- 운영 영상: camera-night 단위로 train/val/test를 분리한다.
- backlog 300의 오류를 학습에 사용하면 해당 300은 EDA/training으로 강등한다.
- 최종 adoption은 모델·threshold 동결 이후 촬영된 미래 camera-night holdout으로만 평가한다.
- 정적 장면 near-duplicate는 clip별 cap을 적용해 특정 배경이 학습을 지배하지 않게 한다.

## 6. Phase 2 — 학습 순서

1. RF-DETR Nano v3를 v2와 같은 학습 설정으로 먼저 학습한다.
2. 같은 frozen regression set에서 v2와 v3를 비교해 데이터 개선 효과만 측정한다.
3. Nano가 데이터 다양성을 충분히 학습한 뒤에도 recall 병목이 남을 때만 RF-DETR Small을 A/B한다.
4. 모델을 고른 뒤 future holdout inference 전에 threshold를 고정한다.

모델 크기 변경과 데이터 변경을 동시에 하지 않는다. v3의 1차 레버는 모델 크기가 아니라 사람 검수된 운영 다양성이다.

## 7. Phase 3 — 출력 evidence 확장

기존 JSON 계약을 유지하면서 다음 파생값을 추가 후보로 둔다.

- sampled frame별 gecko detection과 timestamp
- visible frame ratio
- max/median confidence
- bbox center trajectory와 이동 거리
- edge-touch / partial visibility / temporal stability
- camera ROI별 dwell time 계산에 필요한 bbox history
- checkpoint hash, dataset version, sampler version, threshold

파생값 추가는 schema version을 올리고 하위 호환을 유지한다. Gate는 좌표와 품질 metadata만 공급하며 ROI 의미 해석과 route 결정은 소비자가 담당한다.

## 8. 활용 단계

| 단계 | 허용 용도 | 금지 |
|---|---|---|
| L0 지금 | bbox/best frame/visibility metadata 저장, 라벨링 초안, hard-case 발굴 | VLM 자동 skip |
| L1 v3 shadow | bbox trajectory × camera ROI로 체류·활동 evidence 생성, VLM frame 우선순위 | 행동 자동 확정 |
| L2 독립 평가 후 | frozen router의 입력 evidence로 사용 | Gate가 route 직접 결정 |
| L3 안전성·비용 gate 통과 후 | 검증된 strata에서만 absent 후보를 제한적으로 비용 절감에 사용 | 미검증 카메라·모프까지 일반화 |

ROI crop 자체가 VLM 행동 정확도를 높인다고 가정하지 않는다. bbox의 1차 가치는 좌표·체류·프레임 선택·검수 보조다.

## 9. 평가와 승격 조건

### Prelabel shadow

- 모든 clip을 처리하고 `gecko_visible=false`도 삭제하지 않는다.
- camera/morph/IR/occlusion strata별 recall·FP·bbox 품질을 보고한다.
- FN과 uncertain은 라벨링 큐에 자동 적재한다.

### 자동 skip 후보 승격

- 동결 이후 future holdout에서 positive clip 최소 300건을 확보한다.
- 전체 positive에서 FN 0건을 기본 목표로 하고, strata별 FN을 숨기지 않는다.
- 특정 camera/morph/IR strata에서 FN이 1건이라도 나오면 해당 strata는 skip 금지다.
- specificity와 실제 비용 절감은 petcam-lab의 별도 router 비용 시험에서 검증한다.
- Gate 단독 성능이 좋아도 P0 recall·사람 검수 부담·eventual VLM KRW gate를 통과하기 전에는 production skip을 켜지 않는다.

## 10. 라벨링 웹 계약

라벨링 웹 Gate 검수 모드는 다음을 제공해야 한다.

- detector/Claude 결과를 처음 숨기는 blind labeling
- 원본 영상, sampled frame, bbox overlay on/off
- `visible / absent / uncertain`
- bbox 추가·삭제·교정
- occlusion/edge/distance/IR/glare 등 hard-case tag
- camera·개체·모프·사육장 metadata
- dataset role(train/val/test/future-holdout), 검수자, 수정 이력
- camera-night split 충돌과 class/strata 부족 경고

## 11. 즉시 다음 순서

1. petcam 라벨링 웹 Gate 검수 스펙 작성
2. v2 artifact hash와 backlog 300 sample list 동결
3. 300건 human-first blind GT
4. v2 best-EMA 재평가 report
5. 새 카메라·개체·사육장 데이터 수집
6. v3 Nano 학습
7. shadow 배포와 future holdout
8. petcam router 비용 시험으로 제한적 활용 검토
