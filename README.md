# gecko-vision-gate

RBA 파이프라인에서 Claude 분석 전에 도마뱀이 보이는지 확인하고 시각 단서를 뽑는 Python prelabel 프로젝트.

```text
R2 event clip
→ gecko-vision-gate
→ clip_prelabels JSON
→ Claude RBA Worker
→ analysis_events
→ daily_reports
```

## 역할

이 프로젝트는 행동 분석기가 아니다.

```text
하는 일:
- gecko_visible 판단
- visibility_confidence 계산
- best_frame_ts 선택
- gecko_bbox 추출
- detected_objects JSON 출력

하지 않는 일:
- drinking 확정
- feeding 확정
- defecating 확정
- 건강 상태 판단
- Claude 호출
```

## 문서

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — 전체 기획, JSON 계약, 평가 기준, 백엔드 통합 요청안
- [docs/MODEL_AND_TRAINING_PLAN.md](docs/MODEL_AND_TRAINING_PLAN.md) — 모델 후보, 학습자료, 학습/평가 진행 순서
- [specs/architecture.md](specs/architecture.md) — 확정 아키텍처 (상시 prelabeler) + 구현 Phase

## 상태 (2026-06-18)

**Phase 1 진행 중** — 학습 데이터셋 + fine-tune 파이프라인 구축 완료.
- 데이터: train **2,149** · val **180** · **test 170(운영 전용, negative 56·야간 IR 포함)** COCO 완성. 상세 → [datasets/README.md](datasets/README.md).
- 도구: SerpApi hard-case 크롤러 · Roboflow/Label Studio COCO importer · 무결성 가드 · fine-tune 스크립트(`scripts/`), 26 pytest.
- **RF-DETR v1 완료** (RFDETRNano, MPS) — negative 확대 라운드: **클립단위 test(25 pos/21 neg)** recall@0.25 **0.96** · FP **2/21**. v0는 같은 test에서 recall 1.00이지만 FP **19/21**(빈 클립 90% 오발동 — v0 test에 negative 0이라 그동안 미측정). → **권장 게이트 conf 0.25**, checkpoint `runs/gecko_v1/`.
- **작동하는 게이트**: `prelabel --checkpoint runs/gecko_v1/checkpoint_best_total.pth` 로 실제 gecko 탐지(검증됨).
- 다음: 게이트 운영 적용 · 환경/시간대 다양성 확대 · (선택) `--model small` 시도.

> 핵심 교훈: v0의 "recall 1.00·FP 0"은 **test에 negative가 0**이라 생긴 착시였음. negative 25→445 확대 후 클립 FP 19→2로 실측·해결. (`runs/`는 gitignore — 모델 파일 미커밋)

## 설치 & 실행

```bash
uv sync                       # 의존성 설치 (Python 3.12, rfdetr + opencv)

# 단일 클립 prelabel → JSON 계약 출력
uv run python -m gecko_vision_gate.prelabel \
  --input path/to/clip.mp4 \
  --output samples/outputs/clip.json
# 옵션: --frames 12 --threshold 0.5 --model-size nano|small|medium

uv run pytest                 # 유닛테스트
```
