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
└── manifest.csv          # 출처·split·라벨상태 SOT
```

## manifest.csv 컬럼

| 컬럼 | 값 | 용도 |
|---|---|---|
| `filename` | raw 상대경로 | 식별 |
| `source` | operational / crawl_breeder / crawl_google / crawl_nighttime / negative | 소스 격리 |
| `clip_id` | 운영 영상 출처 (크롤링은 빈칸) | 영상단위 split |
| `split` | train / val / test | 학습 분배 (**test = 운영만**) |
| `labeled` | yes / no | 라벨링 추적 |
| `domain` | day / ir_night / closeup / distant | hard case 추적 (recall 약점 분석) |

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

## 현재 통계 (2026-06-17)

- **crawl_breeder 7830** — 분양샵/브리더, crested gecko, 밝은 주간 클로즈업, 312~396px.
- operational / crawl_google / crawl_nighttime / negative: 비어있음 (수집 예정).

⚠️ **도메인 갭**: 크롤링 = 주간 클로즈업 / 운영 = 야간 IR·원거리. 크롤링만으로 학습하면 운영에서 실패 → **운영 프레임 필수**(특히 hard case). test 는 운영만.

## 출처 / 라이선스

- `crawl_breeder`: 분양샵/브리더 공개 이미지 크롤링. 내부 fine-tune 학습용. **상용 확대 시 라이선스 재검토.**
