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
