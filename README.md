# gecko-vision-gate

RBA 파이프라인에서 게코 검출·추적 단서를 공급하고 Gecko Motion Engine(GME) shadow 코어를 제공하는 프로젝트.

```text
R2 event clip
→ gecko-vision-gate
→ clip_prelabels JSON
→ GME tracker / media QA
→ candidate 움직임 시간 + 추적 품질
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
- 최대 30fps 순차 분석과 다중 게코 trajectory 생성
- observed/tracked/interpolated/unknown provenance 분리
- moving/static/unknown/camera_motion candidate 구간 계산

하지 않는 일:
- drinking 확정
- feeding 확정
- defecating 확정
- 건강 상태 판단
- VLM 호출 여부 결정
- 사용자 활동시간 교체
- 자동 skip·원본 삭제
```

## 문서

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — 전체 기획, JSON 계약, 평가 기준, 백엔드 통합 요청안
- [docs/MODEL_AND_TRAINING_PLAN.md](docs/MODEL_AND_TRAINING_PLAN.md) — 모델 후보, 학습자료, 학습/평가 진행 순서
- [specs/architecture.md](specs/architecture.md) — 확정 아키텍처 (상시 prelabeler) + 구현 Phase
- [specs/gate-v3.md](specs/gate-v3.md) — v2 최종 감사 + 사람 GT·환경 다양성·future holdout 기반 v3 SOT

## 상태 (2026-07-12)

**v2 완료, v3 설계 확정 / 실행 전.**
- 데이터: train **2,770** · val **311** · **test 300(운영 전용, frame 230 positive/70 negative)**. 상세 → [datasets/README.md](datasets/README.md).
- 도구: SerpApi hard-case 크롤러 · Roboflow/Label Studio COCO importer · 무결성 가드 · fine-tune 스크립트(`scripts/`), 26 pytest.
- **RF-DETR v2 완료** (RFDETRNano, MPS) — 동일 test에서 v1→v2 frame recall@0.25 0.80→0.98, clip recall 0.84→0.98. FP 증가로 threshold 0.5를 잠정 운영점으로 기록했다. 상세 [R0002](reports/R0002-evening-recall-v2.md).
- **현재 금지:** v2 결과로 VLM 자동 skip을 켜지 않는다. petcam backlog 평가는 다른 checkpoint와 Claude proxy GT를 사용해 recall 90.9%였으므로 사람 GT 기반 v2 최종 감사가 먼저다.
- **다음:** `specs/gate-v3.md` Phase 0 — best-EMA artifact 고정, backlog 300 전체 blind GT, 카메라·개체·사육장 다양성 확대.

> 핵심 교훈: 데이터셋 내부 점수보다 새 운영 도메인의 사람이 확정한 GT가 우선이다. v3는 같은 카메라 점수를 더 올리는 라운드가 아니라 환경 다양성과 독립 holdout을 확보하는 라운드다.

## 설치 & 실행

```bash
uv sync                       # 의존성 설치 (Python 3.12, rfdetr + opencv)

# 단일 클립 prelabel → JSON 계약 출력
uv run python -m gecko_vision_gate.prelabel \
  --input path/to/clip.mp4 \
  --output samples/outputs/clip.json
# 옵션: --frames 12 --threshold 0.5 --model-size nano|small|medium

uv run pytest                 # 유닛테스트

# GME 단일 클립 shadow 분석(경로는 출력하지 않고 redacted summary만 표시)
uv run gecko-gme --input path/to/clip.mp4 \
  --checkpoint runs/gecko_v2/checkpoint_best_ema.pth --threshold 0.5
```

### 촬영 서버용 3상태 Python wrapper

```python
from gecko_vision_gate.gme_presence import analyze_presence_with_gate

result = analyze_presence_with_gate(
    "/absolute/path/to/clip.mp4",
    checkpoint="/absolute/path/to/checkpoint_best_ema.pth",
    threshold=0.5,
    model_size="nano",
)
payload = result.to_dict()

match result.decision:
    case "detected_candidate":
        destination = "A"  # Gate가 게코 후보를 직접 관측함
    case "not_observed":
        destination = "B_REVIEW"  # 게코 부재 확정이 아니라 미관측
    case "unresolved":
        destination = "RETRY_OR_QUARANTINE"
```

`not_observed`를 원본 삭제나 영구 제외 근거로 사용하지 않는다. `unresolved`도 B에 합치지 않는다.
결과에는 모델·checkpoint SHA-256·threshold·frame 수·unknown/camera-motion 시간이 포함되고,
입력 영상 경로와 credential은 포함되지 않는다.
