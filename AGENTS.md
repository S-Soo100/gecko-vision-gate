# AGENTS.md — gecko-vision-gate

이 저장소의 에이전트 지침은 **`CLAUDE.md` 와 동일**하다. 먼저 `CLAUDE.md` 를 읽어라.

핵심 요약:
- **프로젝트**: 펫캠 mp4 → 게코 가시성 게이트(`clip_prelabels` JSON). 상세 `README.md` · `PROJECT_PLAN.md` · `specs/architecture.md`.
- **현재 (2026-06-25)**: RF-DETR **v1** (negative 확대) — 클립 test recall@0.25 0.96 · FP 2/21. 게이트 = `prelabel --checkpoint runs/gecko_v1/checkpoint_best_total.pth --threshold 0.25`. 상세 `reports/R0001`.
- **▶ 다음 작업**: `docs/MODEL_AND_TRAINING_PLAN.md` §9 "다음 세션 로드맵" (데이터 확대 → auto-label 루프 → 재학습). 단일 출처.
- **규칙**: test=운영 영상만(누수 금지), 외부 데이터는 train만, 이미지/`runs/`/`staging/`/`.env` gitignore, 학습 deps=`train` 그룹, 데이터 변경 후 `check_dataset.py`·코드 변경 후 `pytest`. 새 작업은 main 에서 새 브랜치. **의미있는 라운드마다 `reports/` 리포트 1개**(규칙 `reports/README.md`).
