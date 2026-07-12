# AGENTS.md — gecko-vision-gate

이 저장소의 에이전트 지침은 **`CLAUDE.md` 와 동일**하다. 먼저 `CLAUDE.md` 를 읽어라.

핵심 요약:
- **프로젝트**: 펫캠 mp4 → 게코 가시성 게이트(`clip_prelabels` JSON). 상세 `README.md` · `PROJECT_PLAN.md` · `specs/architecture.md`.
- **현재 (2026-07-12)**: RF-DETR **v2** 완료. 자체 test는 clip recall@0.25 0.978이지만 petcam backlog 평가는 다른 checkpoint+Claude proxy GT로 0.909라 수치가 충돌한다. 자동 skip 금지.
- **▶ 다음 작업**: [`specs/gate-v3.md`](specs/gate-v3.md) — v2 artifact 감사 → backlog 300 사람 blind GT → 카메라·개체·사육장 다양성 → v3 Nano → shadow/future holdout. 최신 단일 출처.
- **규칙**: test=운영 영상만(누수 금지), 외부 데이터는 train만, 이미지/`runs/`/`staging/`/`.env` gitignore, 학습 deps=`train` 그룹, 데이터 변경 후 `check_dataset.py`·코드 변경 후 `pytest`. 새 작업은 main 에서 새 브랜치. **의미있는 라운드마다 `reports/` 리포트 1개**(규칙 `reports/README.md`).
