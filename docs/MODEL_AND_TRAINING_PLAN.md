# 모델 선택과 학습 계획

> 목적: `gecko-vision-gate`에서 어떤 detector를 쓸지, 어떤 학습자료가 필요한지, 어떻게 학습/평가할지 정리한다.

## 1. 현재 목표

이 프로젝트의 첫 목표는 행동 분석이 아니다.

```text
목표:
- event clip 안에 도마뱀이 보이는지 확인
- 가장 잘 보이는 프레임을 고르기
- 도마뱀 bbox와 confidence를 JSON으로 출력
- Claude RBA Worker가 분석할지 결정하는 gate 역할
```

하지 않는 것:

```text
- drinking 확정
- feeding 확정
- defecating 확정
- 건강 상태 판단
- 아침 리포트 생성
```

## 2. 모델 후보

### 2.1 RF-DETR core

장점:

```text
- Roboflow가 공개한 최신 계열 object detector
- detection + segmentation 학습 지원
- core 모델은 Apache 2.0 계열로 안내됨
- 상용 전환을 고려할 때 라이선스 부담이 낮은 편
- COCO류 detection workflow와 잘 맞음
```

주의:

```text
- YOLO 계열보다 생태계 경험치가 적을 수 있음
- 로컬 학습/배포 자료는 직접 검증 필요
- RF-DETR+ 같은 확장 모델은 별도 라이선스일 수 있으니 core 모델 범위 확인 필요
```

### 2.2 YOLOX

장점:

```text
- Apache-2.0 license
- YOLO 계열이라 detector 개념이 익숙함
- custom dataset 학습 문서가 있음
- one-stage real-time detector라 gecko visibility gate에 잘 맞음
```

주의:

```text
- Ultralytics YOLO보다 설치/사용 경험이 덜 매끈할 수 있음
- 레포/의존성 스택이 상대적으로 오래된 편이라 환경 세팅 검증 필요
- 최신 모델 대비 성능/편의성은 실측 필요
```

### 2.3 Ultralytics YOLO11 / YOLOv8

장점:

```text
- 개발 편의성 좋음
- 튜토리얼과 생태계가 풍부함
- 빠르게 baseline을 만들기 쉬움
```

주의:

```text
- 공식 문서 기준 AGPL-3.0 또는 Enterprise License
- 상용/비공개/internal automated workflow 전제에서는 라이선스 검토 필수
- 이 프로젝트의 기본 상용 후보로는 신중
```

## 3. 추천

결정:

```text
RF-DETR core
```

이유:

```text
- 상용 가능성을 고려한 라이선스 부담이 낮다.
- 최신 detector 계열이다.
- 지금 목표인 gecko visibility + bbox 추출에 잘 맞는다.
```

백업 후보:

```text
YOLOX
```

이유:

```text
- Apache-2.0이라 상용 친화적이다.
- YOLO 계열이라 detector로 이해하기 쉽다.
- RF-DETR 로컬 학습/배포가 불편하면 현실적인 대체 후보가 된다.
```

비교 실험용:

```text
Ultralytics YOLO11
```

이유:

```text
- 편의성 baseline으로는 좋다.
- 단, 상용 전제에서는 AGPL/Enterprise 이슈 때문에 기본 선택으로 두지 않는다.
```

정리:

```text
v0 기본 detector = RF-DETR core
YOLOX = RF-DETR이 설치/학습/배포에서 막힐 때의 대체 후보
Ultralytics = 빠른 편의성 baseline 또는 참고용, 상용 기본 선택 아님
```

## 4. 필요한 학습자료

학습에는 영상 자체보다 **프레임 이미지 + bbox 라벨**이 필요하다.

클래스는 v0에서 하나만 둔다.

```text
class:
- gecko
```

필요한 데이터 종류:

```text
A. gecko-present
도마뱀이 보이는 프레임

B. gecko-absent
움직임/조명/그림자는 있지만 도마뱀이 없는 프레임

C. ambiguous
가림, 흐림, 위장색, 원거리, 일부만 보이는 프레임
```

최소 PoC:

```text
gecko bbox 이미지: 100~200장
no-gecko negative 이미지: 100장
ambiguous 이미지: 30~50장
```

괜찮은 v0:

```text
gecko bbox 이미지: 500~1,000장
no-gecko negative 이미지: 300~500장
test 전용 영상: 50~100개
```

더 좋은 v1:

```text
gecko bbox 이미지: 2,000장+
다양한 카메라 각도/조명/거리/가림 포함
```

반드시 포함할 운영 분포:

```text
- 야간 IR
- 상단 대각선 원거리
- 바닥/벽과 색이 비슷한 게코
- 머리/꼬리/몸 일부만 보이는 장면
- 흐림
- 물그릇/은신처/나무 뒤 가림
- 게코 없이 조명 변화나 그림자만 있는 장면
```

## 5. 라벨 형식

처음부터 COCO 포맷을 추천한다.

```text
COCO format
- images
- annotations
- categories
```

이유:

```text
- object detection 표준에 가깝다.
- 모델 교체가 쉽다.
- RF-DETR / YOLOX / 다른 detector로 이동하기 좋다.
```

negative 이미지는 annotation 없이 둔다.

```text
image 있음
annotation 없음
→ 도마뱀 없는 프레임
```

## 5.5 그냥 영상/사진을 보내면 학습될까

결론:

```text
object detection 학습에는 bbox 라벨이 필요하다.
영상/사진만 모아서 "일단 보내기"로는 gecko detector가 제대로 학습되지 않는다.
```

왜냐하면 detector는 이미지 전체 라벨만 배우는 게 아니라, 이미지 안에서 **어디가 도마뱀인지**를 배운다.

필요한 정답:

```text
image: frame_001.jpg
class: gecko
bbox: x, y, width, height
```

즉 최소한 도마뱀이 보이는 프레임에는 박스를 쳐야 한다.

다만 처음부터 모든 프레임을 하나하나 라벨링할 필요는 없다. 단계적으로 간다.

### 추천 라벨링 전략

1단계: 작은 수동 라벨셋

```text
운영 영상에서 다양한 프레임 100~200장 추출
도마뱀 bbox를 직접 라벨링
no-gecko negative 이미지도 100장 수집
```

목적:

```text
RF-DETR이 우리 환경에서 학습/추론 가능한지 확인
```

2단계: 초기 모델 학습

```text
수동 라벨 100~200장으로 작은 detector fine-tuning
```

목적:

```text
완벽한 모델이 아니라, 다음 라벨링을 도와주는 초안 모델 만들기
```

3단계: auto-label + 사람 검수

```text
초안 모델로 새 프레임에 bbox 자동 예측
사람이 틀린 박스 수정
빠진 도마뱀 추가
잘못 잡은 박스 삭제
```

목적:

```text
처음부터 전부 손으로 그리는 부담 줄이기
```

4단계: hard case 집중 라벨링

```text
모델이 자주 놓치는 장면만 추가 라벨링
- 야간 IR
- 원거리
- 가림
- 흐림
- 위장색
- 벽/나무 뒤 일부만 보임
```

목적:

```text
false negative 줄이기
```

### 하면 안 좋은 방식

```text
영상/사진만 잔뜩 모아서 라벨 없이 학습시키기
→ detection 학습에는 부적합

예쁜 게코 사진 위주로 라벨링하기
→ 운영 환경에서 실패

연속 프레임 1,000장을 거의 똑같이 라벨링하기
→ 데이터 수만 부풀고 다양성 부족

train/test를 프레임 단위로 섞기
→ 같은 영상 장면이 양쪽에 들어가 점수 부풀어 오름
```

### 현실적인 최소 루프

```text
1. 영상 20~30개 고르기
2. 각 영상에서 대표 프레임 5~10장 추출
3. 총 150~300장 라벨링
4. RF-DETR core fine-tuning
5. test 영상에서 false negative 확인
6. 놓친 유형만 추가 라벨링
7. 재학습
```

핵심:

```text
처음에는 하나하나 라벨링이 필요하다.
하지만 작은 seed 라벨셋을 만든 뒤에는 auto-label + 사람 검수로 속도를 올린다.
```

## 6. 학습 진행 순서

### Step 1. 영상에서 프레임 추출

mp4 전체를 바로 학습시키는 게 아니라 프레임 이미지를 뽑는다.

```text
clip_001.mp4
→ frame_000.jpg
→ frame_060.jpg
→ frame_120.jpg
```

처음에는 1초마다 1장 또는 일정 간격 샘플링으로 시작한다.

주의:

```text
너무 비슷한 프레임만 많으면 학습 데이터가 부풀기만 한다.
다양한 자세/거리/조명/가림을 우선한다.
```

### Step 2. bbox 라벨링

라벨링 툴에서 도마뱀을 박스로 친다.

```text
class: gecko
bbox: x, y, width, height
```

도마뱀이 안 보이는 이미지는 박스를 치지 않는다.

### Step 3. train / val / test 분리

프레임 단위로 섞으면 안 된다.

같은 영상에서 나온 프레임이 train과 test에 동시에 들어가면 점수가 부풀어 오른다.

추천:

```text
train videos: 70%
val videos: 15%
test videos: 15%
```

즉 영상 단위로 먼저 나누고, 그 안에서 프레임을 배정한다.

### Step 4. fine-tuning

pretrained detector에서 gecko 한 클래스 detector로 fine-tuning한다.

```text
input: image
output: gecko bbox + confidence
```

### Step 5. 이미지 단위 평가

기본 지표:

```text
precision
recall
mAP
false negative count
```

이 프로젝트에서 가장 중요한 지표:

```text
recall
```

이유:

```text
도마뱀이 있는데 없다고 판단하면 Claude 분석 전에 중요한 클립을 버릴 수 있다.
```

### Step 6. 영상 단위 평가

최종 제품은 이미지가 아니라 mp4 clip을 입력으로 받는다.

그래서 여러 프레임을 샘플링한 뒤 영상 단위 JSON을 만든다.

```text
mp4
→ 12장 샘플링
→ 각 프레임 detector 실행
→ 최고 confidence frame 선택
→ threshold 이상이면 gecko_visible=true
```

영상 단위 출력:

```json
{
  "gecko_visible": true,
  "visibility_confidence": 0.91,
  "best_frame_ts": 18.2,
  "gecko_bbox": [120, 80, 260, 190]
}
```

### Step 7. threshold 결정

초기 기준:

```text
confidence >= 0.7
→ Claude 분석

0.3 <= confidence < 0.7
→ Claude 분석 + review 후보

confidence < 0.3
→ skip 후보
→ 단, 하루 N개는 샘플 검수
```

threshold는 감으로 정하지 않고 test set에서 false negative를 보고 조정한다.

### Step 8. 백엔드 전달용 스크립트

최종적으로 백엔드 개발자가 이렇게 쓸 수 있어야 한다.

```bash
uv run python -m gecko_vision_gate.prelabel \
  --input /tmp/clip.mp4 \
  --output /tmp/prelabel.json
```

백엔드 통합 흐름:

```text
R2에서 mp4 다운로드
→ prelabel 스크립트 실행
→ JSON 읽기
→ clip_prelabels 저장
→ Claude job 생성 여부 결정
```

## 7. 성공 조건

좋은 모델을 만드는 것보다 중요한 성공 조건:

```text
도마뱀이 있는 영상을 Claude 전에 놓치지 않는 것.
```

우선순위:

```text
1. false negative 최소화
2. 안정적인 JSON 출력
3. 백엔드가 쉽게 호출 가능한 CLI
4. 나중에 모델 교체 가능한 구조
5. 속도/비용 최적화
```

## 8. 다음 액션

> **진행 현황 (2026-06-17)** — 데이터/스크립트 상세는 `datasets/README.md`.

```text
1. RF-DETR core 설치/예제 inference 검증        ✓ (Phase 0)
2. 운영 영상에서 프레임 추출                     ✓ 32 clip / 185 프레임
3. gecko bbox 라벨링 시작                        ✓ 운영 185 라벨(Label Studio) + 외부 Roboflow PD 1,430 base
4. no-gecko negative 프레임 수집                 ◐ 운영 negative 25 (부분 — 확대 여지)
5. 작은 PoC fine-tuning                          ✓ RFDETRNano MPS, best val mAP 0.82
6. false negative 중심 평가                      ✓ test(28) recall@0.25=1.00 · mAP@50 0.90 (단 negative 0 → FP 미측정)
7. JSON contract 확정                            — (PROJECT_PLAN.md)
```

후속: 베이스라인 → auto-label 루프(초안 모델로 미라벨 프레임 pre-label → 사람 검수 → 재학습),
hard-case(야간/가림) 집중 라벨, fine-tune 가중치를 `prelabel` 에 연결.

## 9. 다음 세션 로드맵 (2026-06-18)

> **v0 상태**: test(28) recall@0.25=1.00 · mAP@50 0.90 · val-neg FP 0/7 → 게이트 conf≈0.25.
> **약점**: test/negative 표본 작음(neg 25), 주간 편중, 같은 펫캠 환경. → 개선 핵심은 모델이 아니라 **데이터**.

### 방법 — auto-label 루프 (v0 모델을 라벨링 조수로)
1. `uv run python scripts/autolabel.py --checkpoint runs/gecko_v0/checkpoint_best_total.pth --source <폴더> --conf 0.25` → bbox 초안 COCO
2. Label Studio 로 불러 사람이 교정(빠른 수정·놓친 것 추가·틀린 박스 삭제)
3. `scripts/import_label_studio_operational.py`(운영) / `import_roboflow_coco.py`(외부) 로 재인입
4. `scripts/train_gecko_detector.py --model small --epochs 30 --accelerator mps` 재학습
5. recall·FP 재측정(`runs/.../gate_metrics.json`) → 놓친 유형만 다음 라운드 집중(§5.5 step4)

### 데이터 우선순위 (영향 큰 순)
- [ ] **1. negative 확대 (최우선)** — ⚠ 파일럿(2026-06-18): `dataset-203` 균등샘플링 yield **0.6%**(이벤트 클립이라 게코 상존) → 균등추출 **부적합**. **야간 연속녹화 신규 확보**(권장) 또는 **야간 클립 v0 1fps 정밀스캔**(보조)으로 전환. FP 6장 전부 야간 IR. 큐레이션 기준 **`NEGATIVE_DATA_GUIDELINE.md`**. 목표 100~300장(야간 비중↑).
- [ ] **2. 야간·가림 hard-case** — 운영 프레임 더 + `staging/` 크롤 후보 선별·라벨. false negative ↓.
- [ ] **3. 환경/시간대 다양성** — 새 카메라·사육장·시간대 클립 추가(학습 + test 대표성).

### 모델 손잡이 (보조 — 데이터 다음)
- [ ] operational 비중 ↑ (현재 주간 1430 : 운영 127 편중)
- [ ] Nano → `--model small`
- [ ] 작은 데이터 과튜닝 금지(§5.5)

### 완료 기준
더 크고 다양한 test(negative 포함)에서 recall 높음 + FP 낮음 → 게이트 conf 확정 → mp4 클립단위 평가(§6).

### ▶ 세션 재개점 (2026-06-20 — negative 야간 IR 영상 대기 중)

**현재 상태**: negative 확대 1라운드. 파일럿으로 "이벤트 클립(dataset-203) 균등샘플링은 negative
yield 0.6% 로 부적합" 판명 → 가이드라인 확립(`NEGATIVE_DATA_GUIDELINE.md`). **다음 진행은 사용자가
야간 IR 펫캠 영상을 찍어와야 가능**(폰 사진 ❌ — 도메인 갭·IR 글레어는 후처리로 못 만듦). 가이드 핵심:
**같은 펫캠으로 · 게코 유무 같은 구도 대조 · 야간 IR 비중↑ · 100~300장**.

**사용자가 "영상 갖고왔어" 하면** (영상은 보통 `~/petcam-lab/storage/<폴더>`):
1. `extract_operational_frames.py --source-dir <폴더> --frames N` — 프레임 추출 (멱등: 기존 clip skip)
2. `autolabel.py --checkpoint runs/gecko_v0/checkpoint_best_total.pth --source operational --conf 0.25` — 박스0=negative 후보 자동선별
3. [사람] Label Studio 빈제출로 negative 확정 (+놓친 야간 게코는 bbox). ⚠ v0가 놓친 가림/위장 게코가 박스0에 섞이니 **검수 필수**
4. `import_label_studio_operational.py` — COCO 재인입
5. `train_gecko_detector.py --model small --epochs 30 --accelerator mps` — 재학습
6. recall·FP 재측정(`runs/.../gate_metrics.json`) — negative 확대 효과 확인

**사용자가 "뭐부터 해야해?" 하면**: 영상 있으면 위 1~6, 없으면 `NEGATIVE_DATA_GUIDELINE.md` 기준으로 야간 IR 영상 확보부터 안내.

**이번 세션 미결/대기**:
- **미커밋**: 문서(README·LEARNINGS·이 파일) + `extract_operational_frames.py` 멱등화 + `NEGATIVE_DATA_GUIDELINE.md`(신규). main 직접금지 → **새 브랜치 커밋 필요**.
- **파일럿 부산물**: `raw/operational` 18 clip·174프레임(positive 173 + v0 박스초안, **manifest 미반영**) — 검수 후 positive 인입 or 삭제.
- **새 영상 없이 당장** negative 일부 확보하려면: 기존 야간 클립 v0 1fps 정밀스캔(보조 경로).

### 추가 후속 (2026-06-18 wrap 발굴)
- [x] **(P0) PR `feat/hardcase-image-pipeline` → main 머지** — ✅ 완료 (PR #1, merge `b9fc731`).
- [ ] **(P1) staging hard-case 후보 선별 → `promote_staging.py`** — 이미 수백 장(ir_night ~414 등) 수집돼 선별 대기.
- [ ] **(P1) mp4 클립단위 평가(§6)** — v0 체크포인트로 지금 가능. test 클립에 `prelabel --checkpoint` → clip-level recall/FP (이미지단위 recall 1.0이 클립단위에서도 유지되는지 조기 검증).
- [ ] **(P1) 운영 positive 160장 domain 태깅** — 현재 빈칸 → recall-by-domain 분석용.
- [ ] **(P2) crawl_breeder 7,830장 활용 결정** — 주간이라 무작정 추가 시 도메인 갭 심화(보류/일부/제외 판단).
- [ ] **(P2) JSON contract 확정(§8.7)** — `specs/architecture.md §5.1`을 SOT로 선언.
- [ ] **파일럿 부산물 처리** — `raw/operational` 18 clip·174 프레임(positive 173 + v0 박스초안, **manifest 미반영**) → 검수 후 운영 positive 인입 or 삭제 판단.
- [x] **`extract_operational_frames.py` 멱등화** — 이미 추출된 clip skip(+`--force`). 기존 라벨 원본 덮어쓰기 방지. (2026-06-18)
- ⚠️ **negative 100장 확보 전 `--model small` 금지** (작은 데이터 과튜닝 위험 — 데이터 먼저).
