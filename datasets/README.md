# gecko-vision-gate 학습 데이터셋

> RF-DETR gecko detector (Phase 1+) 학습/평가용.
> 이미지(`raw/`, `coco/images/`)는 무거워서 **git 추적 안 함** — `manifest.csv` 와 이 README 만 추적. 백업은 추후 R2.

## 구조

```
datasets/
├── raw/                  # 원본 (라벨 전, 소스별 격리 — 절대 안 섞음)
│   ├── operational/      #   운영 펫캠 영상 → 프레임 (영상단위 폴더 <clip_id>/)
│   ├── crawl_breeder/    #   분양샵 크롤링
│   ├── crawl_google/     #   구글 등 다양 (앞으로)
│   ├── crawl_nighttime/  #   야간 게코 (앞으로)
│   └── negative/         #   게코 없는 펫캠 프레임
├── coco/                 # 라벨링 결과 (COCO format) — 라벨링 단계에서 채움
│   ├── images/
│   └── annotations/{train,val,test}.json
├── manifest.csv          # 출처·split·라벨상태 SOT (추적)
├── source_metadata.csv   # 크롤 출처 기록 — query/URL/날짜 (설계서 §6, 추적)
├── crawl_queries.json    # hard-case 검색어 큐 (설계서 §8, 추적)
└── staging/              # 크롤 후보 작업장 (선별 전, git 미추적)
```

## manifest.csv 컬럼

| 컬럼 | 값 | 용도 |
|---|---|---|
| `filename` | raw 상대경로 | 식별 |
| `source` | operational / crawl_breeder / crawl_google / crawl_nighttime / negative | 소스 격리 |
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

## 현재 통계 (2026-06-17)

- **crawl_breeder 7830** — 분양샵/브리더, crested gecko, 밝은 주간 클로즈업, 312~396px.
- operational / crawl_google / crawl_nighttime / negative: 비어있음 (수집 예정).

⚠️ **도메인 갭**: 크롤링 = 주간 클로즈업 / 운영 = 야간 IR·원거리. 크롤링만으로 학습하면 운영에서 실패 → **운영 프레임 필수**(특히 hard case). test 는 운영만.

## 출처 / 라이선스

- `crawl_breeder`: 분양샵/브리더 공개 이미지 크롤링. 내부 fine-tune 학습용. **상용 확대 시 라이선스 재검토.**
