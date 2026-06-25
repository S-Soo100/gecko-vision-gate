# Learnings — gecko-vision-gate

> 세션별 **비반복 교훈(TIL)**. 함정·발견을 모아 다음에 같은 시간 안 쓰게.

## 2026-06-17~18 — v0 수집 → 학습 → 작동 게이트

### 일반 (다른 프로젝트에도)
- **`.gitignore` 인라인 주석 금지**: `datasets/staging/ # 주석` 은 `#` 이후가 주석이 아니라 **패턴의 일부** → 무효화돼 staging 이미지가 커밋에 샜다. 주석은 **별도 행**. 커밋 전 `git diff --cached --name-only` 로 이미지/바이너리 혼입 확인.
- **numpy truthiness 함정**: `arr or []`, `if arr:`, `not arr` 는 다원소 배열에서 `ValueError`. → `if arr is not None and len(arr)`.
- **Google Custom Search JSON API**: 2025 신규 마감 · 2027 종료. 이미지검색 신규 프로젝트는 CSE 빼고 **SerpApi**(무료 250검색/월) 1순위. 출처는 image URL + page URL **양쪽** 기록, 라이선스는 `license_hint` 로 따로.
- **외부 이미지 수집 재개로직은 1차 구현부터**: 매 다운로드 append + 페이지별 체크포인트 + `--resume`. "끝나고 한 번에 저장" 은 인터넷 끊기면 전량 소실(feedle 교훈).
- **순수로직 / I/O 분리**: 파서·변환은 `src/`(오프라인 pytest), HTTP·파일쓰기·재시도는 `scripts/`. 네트워크 없이 로직 검증 가능.

### RF-DETR (rfdetr 1.8)
- 기본 설치는 **추론 전용**. 학습엔 `rfdetr[train,loggers]` extra 필요(pytorch_lightning 등). → `train` 의존성 그룹 + `[tool.uv] default-groups`.
- 데이터셋: `dir/{train,valid,test}/_annotations.coco.json` + 이미지 같은 폴더(Roboflow 레이아웃). 폴더명 `valid`(≠`val`).
- Config 는 pydantic `extra='forbid'` — 모르는 kwarg 즉시 에러. 필드 먼저 확인(introspection).
- 체크포인트: `RFDETR.from_checkpoint(path)`, 클래스명은 `model.class_names`(property). categories 는 id0 더미 + id1 실클래스(Roboflow 관례).
- **MPS(Apple GPU) 로 학습·추론 정상**(GPU 필수 아님). 소규모 데이터는 오버헤드 지배(CPU 와 체감 비슷), 대규모에서 MPS 우위.

### Label Studio
- COCO export 가 `width/height` 를 **null** 로 줄 수 있음 → raw 이미지에서 `cv2.imread().shape` 로 보강.
- export `file_name` 에 원본 경로(clip_id) 보존 → `datasets/raw/` 이후를 잘라 manifest 와 조인. `/`·`\\` 정규화 필수.
- "게코 없음" 은 skip 이 아니라 **빈 제출(박스 0개) = negative**. skip 하면 export 에서 빠진다.

### 평가 설계 (이 프로젝트 핵심)
- recall 우선 게이트 → 단일 conf 가 아니라 **임계값 sweep**(0.1/0.25/0.5/0.7)로 recall·FN·FP 를 함께. test 에 negative 0 이면 **FP 측정 불가**.
- **test = 운영만**, split 은 **clip 단위**(프레임 단위로 섞으면 같은 장면 누수 → 점수 부풀림).
- 외부 주간 데이터(Roboflow) = "게코 형태" prior base. 진짜 recall(야간/가림/원거리)은 **운영 라벨**이 좌우. → **데이터 > 모델 손잡이.**
- recall 1.00 이어도 표본(test 28장·neg 25)이 작으면 "어려운 케이스를 아직 못 본 것"일 수 있다 — 지표보다 **데이터 분포 커버리지**부터 확인.
- **이벤트 트리거 펫캠 클립은 negative 원천 부적합**: 게코 활동 감지로 녹화돼 게코가 상존(파일럿 균등샘플 yield 0.6%, 174장 중 박스0 1장). negative 는 게코 **은신/부재 시간대 연속녹화**에서. v0 FP 6장은 **전부 야간 IR** 의 은신처 실루엣·유리볼 글레어 오인 → 야간 negative 우선. 기준 `NEGATIVE_DATA_GUIDELINE.md`.

## 2026-06-24~25 — v1 (negative 확대 · 야간 IR)

### 평가/지표 함정 (가장 중요)
- **"recall 1.00·FP 0"은 test에 negative 0이라 생긴 착시**: v0 를 새 test(neg 56)에 돌리니 **클립 FP 19/21(90%)** — 빈 클립 대부분 오발동하던 무용지물 게이트였음. 지표는 test 셋만큼만 정직. **버전 비교는 반드시 같은 test 에서**(v0/v1 test 가 달라 각자 gate_metrics 직접비교 불가 → v0 를 새 test 에 재평가해 A/B).
- **클립단위 FP > 프레임단위 FP 가능**: 음성 클립은 프레임 하나라도 오발동하면 "오발동 클립" → 프레임 많을수록 기회↑. v0 프레임 FP 41/56(73%) → 클립 19/21(90%). 게이트는 mp4 단위로 동작하니 **클립단위로 평가**(test 프레임을 clip_id 로 묶어 "어떤 프레임이든 conf≥t 면 클립 fire" — mp4 없이 계산 가능).

### FP 근본원인 (이 사육장)
- 게코가 **릴리화이트(전신 흰색)** → 야간 IR 에서 흰 몸이 **흰 인조넝쿨·관엽식물 흰무늬·유리/물그릇 IR 글레어**와 같은 밝기. v0 가 "창백한 덩어리=게코"로 과일반화 = FP 주범. **밝기·형태 아닌 움직임+게코 고유부위(눈·머리·사지)로 판별.**

### 데이터/검수
- **over-FP 모델의 negative 는 box0 가 아니라 "오검출 박스 삭제"에서 나온다**: v0 가 야간 IR 프레임 97%에 박스 → box0 자동선별로는 IR negative 11장뿐. 진짜 야간 negative 는 검수자가 **글레어 가짜박스를 지워서** 확보. autolabel 은 "그릴 것"뿐 아니라 "지울 것"을 주는 도구.
- **near-dupe 솎기**: 정적 장면 6프레임이 거의 동일 → neg 1168 중 1114 가 near-dupe. **클립당 2장 cap(→420)**으로 다양성 유지 + 과중복 방지(recall 보호).
- 결과: neg 25→445 · test 28→170 · **클립 FP 19→2 · recall@0.25 0.96 유지**(같은 conf 에서 −89% FP).

### Label Studio 로컬파일 적재
- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` + `DOCUMENT_ROOT=datasets/raw` 면 소스 스토리지 없이 `/data/local-files/?d=<rel>` 서빙. **Local Storage 경로 == DOCUMENT_ROOT 면 거부**(하위폴더 요구) — 보통 스토리지 단계 불필요. COCO→LS tasks JSON 변환 시 bbox 는 **원본 dim 대비 %**.
