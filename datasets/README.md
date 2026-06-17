# gecko-vision-gate 학습 데이터셋

> RF-DETR gecko detector (Phase 1+) 학습/평가용.
> 이미지(`raw/`, `coco/images/`, `staging/`, `rfdetr_build/`)는 무거워서 **git 추적 안 함** — `manifest.csv` · `source_metadata.csv` · `crawl_queries.json` · `coco/annotations/*.json` · 이 README 만 추적. 백업은 추후 R2.

## 구조

```
datasets/
├── raw/                  # 원본 (라벨 전, 소스별 격리 — 절대 안 섞음)
│   ├── operational/      #   운영 펫캠 영상 → 프레임 (영상단위 폴더 <clip_id>/)
│   ├── crawl_breeder/    #   분양샵 크롤링 7,830 (미라벨, 주간)
│   ├── roboflow_*/       #   외부 라벨 COCO (예: roboflow_gecko_teddychiu, PD) — import 로 생성
│   ├── crawl_google/     #   구글 등 다양 (앞으로)
│   ├── crawl_nighttime/  #   야간 게코 (앞으로)
│   └── negative/         #   게코 없는 펫캠 프레임
├── coco/                 # 라벨링 결과 (COCO format) — 라벨링 단계에서 채움
│   ├── images/
│   └── annotations/{train,val,test}.json
├── manifest.csv          # 출처·split·라벨상태 SOT (추적)
├── source_metadata.csv   # 크롤 출처 기록 — query/URL/날짜 (설계서 §6, 추적)
├── crawl_queries.json    # hard-case 검색어 큐 (설계서 §8, 추적)
├── staging/              # 크롤 후보 작업장 (선별 전, git 미추적)
└── rfdetr_build/         # 학습용 COCO materialize (symlink, git 미추적)
```

## manifest.csv 컬럼

| 컬럼 | 값 | 용도 |
|---|---|---|
| `filename` | raw 상대경로 | 식별 |
| `source` | operational / crawl_breeder / crawl_google / crawl_nighttime / negative / `roboflow_<name>`(외부 라벨 COCO) | 소스 격리 (폴더명=source) |
| `clip_id` | 운영 영상 출처 (크롤링은 빈칸) | 영상단위 split |
| `split` | train / val / test | 학습 분배 (**test = 운영만**) |
| `labeled` | yes / no | 라벨링 추적 |
| `domain` | ir_night / occluded / ir_night_occluded / negative / day_other | hard case 추적 (설계서 §7 로 통일 — 옛 day/closeup/distant 폐기) |

## 데이터 인입 — 새 이미지 추가 시 (SOP)

1. **소스에 맞는 폴더에 넣기**
   - 구글·다양 → `raw/crawl_google/`
   - 야간 게코 → `raw/crawl_nighttime/`
   - 운영 영상 프레임 → `raw/operational/<clip_id>/` (영상단위!)
   - 새 소스면 `raw/crawl_<name>/` 폴더를 새로 만들어도 됨 — manifest 가 폴더명으로 source 자동 인식.
2. **manifest 갱신**: `uv run python scripts/build_manifest.py`
   - 신규 파일만 추가, 기존에 사람이 채운 split/labeled/domain 은 보존. raw 에서 지운 파일은 manifest 에서도 제거.
3. (라벨링 후) `labeled=yes` + `split` 배정.

**규칙:**
- 운영 프레임은 반드시 `operational/<clip_id>/` 영상단위 (train/test split 누수 방지).
- **test split 은 운영 영상만** — 크롤링(주간 클로즈업)으로 평가하면 운영 성능이 깜깜이.
- 중복 이미지 주의 (같은 걸 여러 번 크롤링). dedup 은 추후 해시 기반.

## Google hard-case 수집 (SerpApi) — 반자동 SOP

> 설계: `docs/GOOGLE_HARDCASE_CRAWLING_DESIGN.md`. 예쁜 사진이 아니라 **운영 false
> negative 를 닮은 hard case**(야간 IR · 가림 · 야간+가림)를 사람이 골라 넣는다.
> 크롤러는 후보만 떠오고 채택은 사람이 한다(§9.1·§15 — 자동 대량적재 아님).
>
> 메커니즘: **SerpApi google_images**. (Google Custom Search JSON API 는 2025 신규
> 마감 · 2027.1.1 종료라 신규 발급 불가 → SerpApi 무료티어 250검색/월로 대체.)

```
1. 검색어 큐 확인/수정      datasets/crawl_queries.json (§8.1~8.3, negative 제외)
2. (계획만) dry-run         uv run python scripts/fetch_hardcase_images.py --dry-run
3. SerpApi 키 설정          .env 에 SERPAPI_KEY=...  (gitignore 됨 / serpapi.com 무료가입)
4. 후보 수집 → staging/     uv run python scripts/fetch_hardcase_images.py --pages 1 --resume
5. [사람] 육안 선별         staging/<domain>/ 에서 나쁜 후보(잘못된 동물·그림·상품·중복) 삭제
6. raw 로 승격              uv run python scripts/promote_staging.py   (→ source_metadata.csv 기록)
7. manifest 갱신            uv run python scripts/build_manifest.py
8. domain/split/labeled     manifest.csv 에서 채움 (domain 은 source_metadata.notes 의 domain_hint 참고)
9. 무결성 가드              uv run python scripts/check_dataset.py   (test=운영만 · domain · 출처기록률)
```

**쿼터 산수:** 큐 23개 × `--pages 1` = SerpApi 23회/실행 → 무료 250/월 = 약 10회 실행분.
`--pages` 를 올리면 비례 증가. 스모크는 `--limit 5` 로 소량만.

**negative(게코 없음)** 는 SerpApi 가 아니라 **운영 프레임에서** 뽑는다(§4.1·§8.4):
`extract_operational_frames.py` 로 추출 → `raw/negative/` 로 분류. (부득이할 때만
`crawl_queries.json` 의 `_negative_reference` 를 수동 참고.)

**재개:** 인터넷이 끊겨도 매 다운로드가 `staging_metadata.csv` 에, 매 페이지가
`.fetch_progress.json` 에 저장된다. `--resume` 면 쿼리별 `last_page+1` 부터 이어받음.

## 스크립트 & 워크플로우

| 스크립트 | 역할 |
|---|---|
| `build_manifest.py` | raw/ 스캔 → manifest.csv 갱신 (멱등, 사람값 보존) |
| `extract_operational_frames.py` | 운영 영상 → `raw/operational/<clip>/` 프레임 |
| `fetch_hardcase_images.py` | SerpApi → `staging/` hard-case 후보 (↑ SerpApi SOP) |
| `promote_staging.py` | `staging/`(선별 후) → `raw/` + source_metadata.csv |
| `import_roboflow_coco.py` | 외부 Roboflow COCO → raw/ + manifest + coco/train.json (gecko 단일클래스 remap) |
| `import_label_studio_operational.py` | Label Studio COCO(운영) → manifest + coco/{train,val,test} (clip 단위 split, negatives 포함) |
| `check_dataset.py` | 무결성 가드 (test=운영만 · domain · 출처기록률) |
| `train_gecko_detector.py` | COCO → RF-DETR materialize → fine-tune → test recall@threshold |

**외부 라벨 데이터셋 인입 (Roboflow COCO)** — 외부 데이터는 train 만(§4.2):
```
Roboflow Universe → Download(format COCO) → unzip
uv run python scripts/import_roboflow_coco.py --src <폴더> --source-name roboflow_<name> --license "<라이선스>" --dry-run
# 숫자 확인 후 --dry-run 빼고 실행
```

**운영 프레임 라벨 인입 (Label Studio)** — clip 단위 split, test=운영만:
```
Label Studio: gecko bbox 라벨(게코 없음=빈 제출=negative) → Export COCO → unzip
uv run python scripts/import_label_studio_operational.py --src <폴더> --dry-run
# 이미지는 raw 의 것을 참조(복사 X), file_name 의 clip_id 로 split 배정
```

**학습 (fine-tune)** — RF-DETR 은 `dataset_dir/{train,valid,test}/_annotations.coco.json` 레이아웃 기대 → 스크립트가 우리 COCO 를 symlink 로 materialize:
```
uv run python scripts/train_gecko_detector.py --build-only                                   # 데이터 준비만 검증
uv run python scripts/train_gecko_detector.py --smoke --accelerator cpu                      # 파이프라인 스모크
uv run python scripts/train_gecko_detector.py --epochs 20 --batch-size 4 --accelerator mps   # 로컬 Apple GPU 실학습
```
디바이스: CPU·MPS·CUDA 모두 가능(속도 CUDA>MPS>CPU). recall 우선 게이트라 결과는 **임계값별 recall sweep**(`runs/<out>/gate_metrics.json`).

## 현재 통계 (2026-06-17)

| source | 장수 | 라벨 | split | 비고 |
|---|---|---|---|---|
| `crawl_breeder` | 7,830 | ✗ 미라벨 | — | 분양샵/브리더, 밝은 주간 클로즈업 312~396px |
| `roboflow_gecko_teddychiu` | 1,430 | ✓ | train | Roboflow Universe, **Public Domain**, 주간. 외부 base |
| `operational` | 185 | ✓ | train 127 / val 30 / test 28 | 펫캠 32 clip, positives 160 + **negatives 25** |
| `crawl_google`/`crawl_nighttime`/`negative` | 0 | — | — | raw 비어있음 — 야간 hard-case 후보는 `staging/` 에 수집됨(선별·승격 대기) |

- **COCO 라벨 완성**: `coco/annotations/` → train **1,557**(roboflow 1430 + 운영 127) · val **30** · test **28**(운영 전용).
- **domain**: day_other 1,430 · negative 25 · (운영 positive 160 미태깅 — recall-by-domain 분석용 추후 태깅).
- **모델**: fine-tune 스크립트 검증 완료(스모크), 실학습은 진행/대기 — 아직 확정 baseline 없음.

⚠️ **도메인 갭 (여전히)**: 라벨된 1,615장 중 1,430(roboflow)+breeder 가 **주간**. 야간 IR·가림 hard-case 신호는 운영 127 + 미선별 staging 뿐 → 운영/hard-case 라벨 확대가 recall 의 핵심. **test 는 운영만**.

## 출처 / 라이선스

- `crawl_breeder`: 분양샵/브리더 공개 이미지 크롤링. 내부 fine-tune 학습용. **상용 확대 시 라이선스 재검토.**
- `roboflow_gecko_teddychiu`: Roboflow Universe (`teddychiu/gecko-qxjbq`), **Public Domain**. 1,430장 주간. 출처는 `source_metadata.csv`.
- `operational`: 자체 운영 펫캠 프레임. 진짜 도메인 — test split 전용.
- SerpApi hard-case 크롤: 이미지별 출처 URL 을 `source_metadata.csv` 에 기록. **상용/공개 배포 시 이미지별 라이선스 재검토**(설계서 §15).
