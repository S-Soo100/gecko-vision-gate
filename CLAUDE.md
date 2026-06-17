# gecko-vision-gate — 에이전트 가이드

RBA 파이프라인의 **Gate**: 펫캠 mp4 → "게코가 보이나?" 판단 + bbox → `clip_prelabels` JSON.
행동분석기가 아니다(drinking/feeding 판단·Claude 호출 안 함). 상세: `README.md` ·
`PROJECT_PLAN.md` · `specs/architecture.md` · `docs/MODEL_AND_TRAINING_PLAN.md`.

## 현재 상태 (2026-06-18) — v0 fine-tune 완료
- RF-DETR(RFDETRNano) gecko detector **v0**: test(28장) **recall@0.25=1.00 · mAP@50 0.90 · val-neg FP 0/7** → 게이트 conf≈0.25.
- **작동 게이트**: `uv run python -m gecko_vision_gate.prelabel --input clip.mp4 --checkpoint runs/gecko_v0/checkpoint_best_total.pth --threshold 0.25`
- 데이터: train 1557(roboflow PD 1430 + 운영 127) / val 30 / **test 28(운영 전용)**. 상세 `datasets/README.md`.
- ⚠️ 약점: test/negative 표본 작음(neg 25), 주간 편중, 같은 펫캠 환경 → 숫자는 좋으나 표본 한계.

## ▶ 다음 작업
**`docs/MODEL_AND_TRAINING_PLAN.md` §9 "다음 세션 로드맵"** 이 단일 출처.
요약: 데이터 확대(① negative 100~300 ② 야간/가림 hard-case ③ 환경 다양성) → **auto-label 루프**
(`scripts/autolabel.py` → Label Studio 교정 → 재인입 → 재학습) → recall·FP 재측정. 첫 걸음도 §9.

## 핵심 규칙
- **데이터 안전**: `raw/` 소스별 격리, **test=운영 영상만**(§4.2 누수 금지), 외부 데이터는 train(+val)만. 변경 전 `datasets/README.md` 확인. `data/`류 덮어쓰기 전 백업.
- **학습**: 학습 deps 는 `train` 의존성 그룹(`[tool.uv] default-groups` 로 자동 포함). MPS(`--accelerator mps`) 가능, CUDA 권장. 무거운 학습은 사용자 터미널에서.
- **git**: 이미지/체크포인트/`staging/`/`rfdetr_build/`/`runs/`/`.env` 는 gitignore. 커밋 대상 = 코드 + 메타(`manifest.csv`·`coco/annotations`·`source_metadata.csv`). `.gitignore` 인라인 주석 금지(2026-06-17 사고).
- **도구**: `scripts/` — build_manifest · extract_operational_frames · fetch_hardcase_images · promote_staging · import_roboflow_coco · import_label_studio_operational · autolabel · check_dataset · train_gecko_detector. 각 docstring 에 usage.
- **검증**: 데이터/스키마 변경 후 `uv run python scripts/check_dataset.py` (test=운영만·domain·출처기록률). 코드 변경 후 `uv run pytest`.
- **브랜치**: v0 는 `feat/hardcase-image-pipeline` → **main 머지 완료(PR #1)**. 새 작업은 main 에서 새 브랜치로. 커밋·푸시는 사용자 요청 시.
