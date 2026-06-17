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

## 상태 (2026-06-17)

**Phase 0 완료** — RF-DETR core 설치 + 로컬 mp4 prelabel 파이프라인 동작. 현재 `v0-coco`(COCO pretrained라 gecko 미검출 — fine-tune 전, `bird`/`cup`으로 오탐). 다음은 Phase 1 (seed 라벨링 → fine-tune).

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
