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

**Phase 1 진행 중** — 학습 데이터셋 + fine-tune 파이프라인 구축 완료.
- 데이터: train **1,557**(외부 Roboflow PD 1,430 + 운영 127) · val 30 · **test 28(운영 전용)** COCO 완성. 상세 → [datasets/README.md](datasets/README.md).
- 도구: SerpApi hard-case 크롤러 · Roboflow/Label Studio COCO importer · 무결성 가드 · fine-tune 스크립트(`scripts/`), 26 pytest.
- RF-DETR fine-tune: 스모크(CPU/MPS) 검증 완료, **실학습 진행 중** → test recall@threshold 베이스라인 측정 예정.
- 다음: 베이스라인 → fine-tune 가중치를 prelabel 에 연결 → auto-label 루프로 운영/hard-case 라벨 확대(false negative ↓).

> 참고: `prelabel` 은 아직 COCO-pretrained 가중치(gecko 미검출) — fine-tune 완료 후 연결 예정.

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
