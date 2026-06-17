# gecko-vision-gate 아키텍처 설계

> **상위 SOT**: `tera-ai-product-master/docs/specs/petcam-ai-pipeline.md §11` (Gate 레이어).
> **기획 초안**: `PROJECT_PLAN.md` · `docs/MODEL_AND_TRAINING_PLAN.md`.
> 본 문서 = 초안 위에 올리는 **확정 아키텍처** (2026-06-17). PROJECT_PLAN §9 "게이팅"을 "상시 prelabel"로 수정 반영.

---

## 1. 한 줄 정의 + 포지셔닝

`gecko-vision-gate` = R2 업로드마다 자동으로 도는 **상시 사전라벨 인프라(auto-prelabeler)**.
모든 영상에 "게코 보이나 / 어디(bbox) / 무슨 단서"를 붙여 **Supabase 메타데이터를 채우고, 그게 VLM·nightly·검수에 주는 힌트**가 된다.

- **detector지 행동 분석기가 아니다** (drinking/feeding 확정 안 함 — 그건 lab Claude / evidence layer).
- **게이트키퍼가 아니다** — VLM을 막거나 라우팅하지 않는다 (§2).
- petcam 파이프라인에서 **R&D/운영 분리의 "부품"** 역할 (lab=연구소가 쓸 detector를 만들어 운영에 상시 공급, SOT §11.2).

---

## 2. 원안에서 바뀐 것 (PROJECT_PLAN 대비) — ⚠️ 재논의 금지

| | PROJECT_PLAN 원안 | **확정 (2026-06-17)** |
|---|---|---|
| 역할 | VLM 게이트키퍼 (`gecko_visible=true`만 VLM 호출) | **상시 prelabeler** (VLM과 무관하게 다 처리) |
| 게이팅 | `gecko_visible`로 VLM 라우팅 / `skip_recommendation` | **뗌** — VLM 호출 여부는 Gate 책임 아님. 소비자가 알아서 |
| 목적 | 비용 절감 (VLM skip) | **evidence baseline(북극성)** + 메타 강화 |
| 트리거 | (암묵) 업로드 후 호출 | **폴링** (mac-mini NAT, §4.2) |

**왜 게이팅을 뗐나**: VLM 호출은 비용·시간이 들지만 prelabel은 영상 올 때마다 싸고 빠르게 처리 가능. 게이팅으로 VLM을 막기보다 **모든 영상의 메타를 미리 깔아두는 게(=VLM에 힌트 더 주기)** 더 크다. `skip_recommendation`/`skip_reason` 필드는 출력에 남기되 **권고 신호일 뿐**, Gate가 직접 차단하지 않는다.

---

## 3. 시스템 흐름

```
Camera/HW → R2 업로드 → Supabase camera_clips INSERT
                              │
        ┌─────────────────────┘  (폴링: ~1분마다 "prelabel 없는 클립?" SELECT)
        ▼
  gecko-vision-gate worker  (mac-mini 24h)
        │  1. camera_clips 폴링 (clip_prelabels 없는 행)
        │  2. R2에서 mp4 다운로드 (temp)
        │  3. 프레임 샘플링 (일정 간격, 12장 기본)
        │  4. RF-DETR 추론 → gecko bbox/conf, (v1+) 객체 단서
        │  5. best_frame 선택 + visibility_confidence 산출
        ▼
  Supabase clip_prelabels INSERT  ← Gate의 유일한 쓰기 영역
        │
        └──▶ 소비자 (단방향, Gate는 이들을 모름):
             • petcam-lab VLM    — 메타를 힌트로 (어느 ROI? best_frame?)
             • nightly-reporter  — "게코 활동 클립"만 추리기 + 디코딩 재활용
             • 라벨링 웹          — 검수 사전표시
             • evidence layer    — gecko bbox × camera_rois = ROI 체류 evidence
```

---

## 4. 핵심 설계 결정

### 4.1 상시 prelabel (게이팅 아님)

모든 업로드 영상을 무조건 prelabel한다. 출력은 "판단"이 아니라 **메타데이터**다 — gecko 좌표, best frame, 시각 단서. 이걸로 Supabase를 빵빵하게 채워두면 다운스트림(VLM/nightly/검수)이 각자 필요한 만큼 가져다 쓴다. Gate는 누가 무엇을 쓰는지 신경 쓰지 않는다 (느슨한 결합).

### 4.2 폴링 트리거 (webhook 아님)

mac-mini가 집 NAT 뒤라 외부 push를 못 받는다.
- ❌ **webhook(push)**: Supabase→mac-mini POST는 mac-mini 외부 노출(공개IP/터널) 필요 — 복잡·보안부담.
- ✅ **폴링(pull)**: worker가 ~1분마다 *"clip_prelabels 없는 camera_clips 있나?"* SELECT (LEFT JOIN). **아웃바운드라 NAT 무관, endpoint 노출 0.**
- petcam-lab VLM worker의 검증된 DB-as-message-bus 패턴 재사용 (학습곡선 0).
- 폴링 단점(실시간X·worker상주·DB부하)이 이 맥락(prelabel 비실시간 + mac-mini 24h + SELECT 1개)에선 다 무력.
- webhook은 MVP-1 클라우드 자동화(실시간 필요) 때 `webhook→Queue→클라우드 worker`로 격상.

### 4.3 evidence baseline = gecko bbox × camera_rois (북극성)

```
gecko bbox(어디)  ×  camera_rois(거기 뭐가 있나)  ×  시간(얼마나)
   = "water_bowl ROI에 18초 체류"   ← VLM 호출 0번으로 생성
```
이게 evidence spec §4.1 `visit` 레벨을 **detector만으로** 달성 → drinking/feeding/prey 비-VLM 레이어(§4.5)의 원재료를 모든 영상에 공짜로 공급. **단 Gate는 "좌표 공급자"까지** — "그 좌표가 water_bowl"이라는 의미는 `camera_rois`(evidence spec §5.1, 사람이 카메라별 정의)가 붙어야 완성. Gate(좌표) + camera_rois(의미) = evidence baseline.

### 4.4 false negative 최소화 (성공 조건)

가장 중요한 지표는 recall. **게코가 있는데 "없음"으로 prelabel하면** 다운스트림이 그 클립을 놓친다. precision보다 recall 우선. 초기엔 비용절감보다 안정성 — `gecko_visible=false`도 바로 폐기하지 않고 메타로 남겨 샘플 검수 가능하게.

### 4.5 worker 공존 — nightly랑 안 겹치게 (SOT §11.4)

| | gecko-vision-gate | nightly-reporter |
|---|---|---|
| Supabase | `clip_prelabels` **쓰기** | 읽기만 |
| R2 | raw clip **읽기만** | raw 읽기 + `reports/` 쓰기 |
| 로컬 상태 | 0 (clip_prelabels 존재로 갈음) | SQLite |
| 주기 | 상시 폴링(~1분) | 야간 분할(3~4회) + 06 종합 |

쓰기 영역 교집합 0 → 충돌 원천 차단. 멱등(이미 prelabel된 클립 skip) + flock(자기 중복 실행 방지) + temp 디렉토리 분리.

---

## 5. 출력 계약 (clip_prelabels)

### 5.1 JSON (worker 출력 — PROJECT_PLAN §7 기반, 게이팅 필드 의미 조정)

```json
{
  "clip_id": "uuid",
  "gecko_visible": true,
  "visibility_confidence": 0.91,
  "best_frame_ts": 18.2,
  "gecko_bbox": [120, 80, 260, 190],
  "detected_objects": [
    {"type": "gecko", "confidence": 0.91, "bbox": [120,80,260,190], "frame_ts": 18.2}
  ],
  "skip_recommendation": false,
  "skip_reason": null,
  "model_name": "rf-detr-core",
  "model_version": "v0",
  "frames_sampled": 12
}
```
> `skip_recommendation`은 **권고만** — 소비자가 참고할 수 있는 신호이지 Gate가 차단하는 게 아님 (§2).
> `detected_objects`의 `type`은 v0=`gecko` 1종, v1+=`food_bowl`/`water_bowl`/`prey`/`shed_skin`/`hand_tool` 확장 (§6).

### 5.2 Supabase 테이블 (PROJECT_PLAN §9 기반)

```sql
CREATE TABLE clip_prelabels (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_id               UUID NOT NULL REFERENCES camera_clips(id) ON DELETE CASCADE,
  gecko_visible         BOOLEAN NOT NULL,
  visibility_confidence REAL NOT NULL,
  best_frame_ts         REAL,
  gecko_bbox            JSONB,          -- [x,y,w,h] or null
  detected_objects      JSONB NOT NULL DEFAULT '[]',
  skip_recommendation   BOOLEAN NOT NULL DEFAULT FALSE,
  skip_reason           TEXT,
  model_name            TEXT NOT NULL,
  model_version         TEXT NOT NULL,
  frames_sampled        INT  NOT NULL,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (clip_id, model_version)       -- 멱등: 같은 클립·버전 재처리 방지
);
CREATE INDEX idx_prelabels_clip ON clip_prelabels(clip_id);
-- 폴링용: prelabel 없는 클립 = camera_clips LEFT JOIN clip_prelabels WHERE prelabels.id IS NULL
```

---

## 6. detector 로드맵 (SOT §11.5)

| 단계 | 클래스 | 출력 | 비고 |
|---|---|---|---|
| **v0** | gecko 1클래스 | gecko_visible / bbox / best_frame | "분석 가치 있나" + 좌표 |
| **v1+** | + food_bowl / water_bowl / prey / shed_skin / hand_tool | ROI 체류 evidence / 객체 단서 | evidence spec §5와 같은 모델 확장 |

- **모델**: RF-DETR core (1순위) / YOLOX (백업). 둘 다 **Apache 2.0** — Ultralytics AGPL 회피 (2026-06-09 결정).
- **"YOLO bbox crop → VLM 재판정" 경로 기각** (evidence spec §4.5, roi-crop 실험 순0). detector는 **좌표·evidence 생성용**이지 VLM 정확도 보강용 아님.
- **학습 전략** (MODEL_AND_TRAINING_PLAN 요약): seed 라벨 100~200장(bbox) → fine-tune → auto-label + 사람 검수 → hard case(IR/원거리/가림/위장색) 집중. train/val/test는 **영상 단위 분리**(프레임 단위 섞으면 누수). petcam-lab `storage/dataset-203/`(197클립)이 seed 후보.

---

## 7. 구현 단계

| Phase | 내용 | 완료 조건 |
|---|---|---|
| **0** ✅ | RF-DETR core 설치 + 로컬 mp4 1개 inference 검증 | **완료 (2026-06-17)** — 아래 결과 |
| **1** | seed 라벨(100~200 bbox) + fine-tune PoC | gecko detector 추론 가능 |
| **2** | 영상단위 JSON + 평가(recall/FN) | test셋 recall 측정 + threshold 도출 |
| **3** | **폴링 worker** (Supabase/R2 연동) | mac-mini에서 camera_clips 폴링 → clip_prelabels 적재 |
| **4** | `camera_rois` 교차 → evidence baseline | ROI 체류 evidence 생성 (lab evidence spec과 페어) |

> Phase 0~2 = detector 만들기 (로컬). Phase 3 = 상시 운영화. Phase 4 = evidence 본격화.

### Phase 0 결과 (2026-06-17) ✅

- **골격**: uv(`pyproject.toml`, Python 3.12) + src layout `src/gecko_vision_gate/{schema,frame_sampling,detector,prelabel}.py`. 의존성 `rfdetr 1.8.0` + `opencv-python-headless`. 유닛테스트 7 passed (`tests/`, 순수 로직 — 인덱스 균등분할 + JSON 계약).
- **검증**: 로컬 mp4 2개 prelabel 동작 확인 (seed = petcam-lab `storage/dataset-203/`).
  - hand_feeding 클립 → 게코를 COCO 모델이 **`bird`로 오탐** (conf 0.5~0.88, 8/12 프레임).
  - 게코 클립 → 물그릇을 **`cup`** 1건. 둘 다 `gecko_visible=false` (계약대로).
- **결론**: 파이프라인(프레임 균등샘플 → RF-DETR 추론 → best frame → JSON 계약)은 동작. COCO pretrained(80 class, gecko 없음)는 게코를 못 잡음 → **Phase 1 fine-tune 필요성 실증.** `model_version="v0-coco"`로 fine-tune 전임을 명시.
- **메모**: rfdetr 1.8.0 Nano 가중치 로드 시 `_kp_active_mask not in checkpoint`(keypoint 헤드 random init) 경고 — detection 출력은 정상(bird/cup 합리적 오탐)이라 무관. Phase 1 fine-tune 시 재확인.
- **CLI**: `uv run python -m gecko_vision_gate.prelabel --input clip.mp4 --output out.json [--frames 12 --threshold 0.5 --model-size nano]`

---

## 8. 평가 기준

- **1순위 recall** (false negative = 게코 있는데 버림). PROJECT_PLAN §8 표(TP/FP/TN/FN/recall/precision) 유지.
- 평가셋: gecko-present 30~50 / absent 30~50 / ambiguous 10~20. 운영 분포(IR·원거리·가림·위장색·흐림) 포함.
- threshold는 감으로 X, test셋 FN 보고 조정 (conf≥0.7 분석 / 0.3~0.7 review / <0.3 skip 후보지만 샘플 검수).

---

## 9. 스코프

**In**: detector(gecko 1클래스) · 폴링 worker · clip_prelabels 적재 · 영상단위 JSON · recall 평가.
**Out (당장 X)**: 행동 확정(drinking/feeding) · v1 멀티클래스(나중) · 자체 R2/DB 인프라 신규(petcam-lab 것 재사용) · MVP-1 클라우드 자동화(webhook/Queue).

---

## 10. 리스크

| 리스크 | 대응 |
|---|---|
| RF-DETR 로컬 학습/배포 난이도 | YOLOX 백업 후보 (Apache) |
| seed 라벨 부족 | dataset-203(197클립) 활용 + auto-label 루프 |
| 위장색/IR/원거리 false negative | hard case 집중 라벨링 (Phase 학습 4단계) |
| 폴링 worker 중복/누락 | UNIQUE(clip_id, model_version) 멱등 + flock |
| mac-mini 리소스(nightly와 공존) | 쓰기영역 분리 + 추론 간헐적이라 무해 (§4.5) |
