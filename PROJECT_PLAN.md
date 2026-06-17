# gecko-vision-gate 기획문서

> RBA 파이프라인에서 Claude 분석 전에 영상을 가볍게 거르는 vision prelabel 프로젝트.

## 1. 한 줄 정의

`gecko-vision-gate`는 R2에 올라온 게코 캠 이벤트 영상을 가볍게 확인해서, **도마뱀이 보이는지 / 어느 프레임이 좋은지 / 어떤 시각 단서가 있는지**를 JSON으로 뽑는 Python 프로젝트다.

이 프로젝트는 행동 분석기가 아니다.

```text
YOLO/RF-DETR 계열 detector
→ gecko_visible / bbox / best_frame_ts / object evidence
→ Claude RBA Worker가 행동 의미 분석
```

## 2. 왜 필요한가

Claude는 비싼 의미 분석기다. 모든 이벤트 영상을 Claude에게 바로 보내면 비용과 시간이 커진다.

그래서 앞단에서 싸고 빠르게 이런 질문에 답한다.

```text
이 영상에 도마뱀이 보이나?
도마뱀이 가장 잘 보이는 프레임은 어디인가?
도마뱀 bbox는 어디인가?
손, 도구, 그릇, prey, 허물 같은 단서가 보이나?
```

이 결과를 이용해 Claude 분석을 라우팅한다.

```text
gecko_visible = true
→ Claude 분석 진행

gecko_visible = false
→ skip 후보 / low priority / 샘플 검수

visibility_confidence 애매함
→ Claude 분석하되 needs_review 후보
```

## 3. 전체 시스템에서 위치

```text
Camera / HW
→ event clip 직접 R2 업로드
→ Supabase camera_clips INSERT
→ gecko-vision-gate prelabel
→ clip_prelabels 저장
→ Claude RBA Worker
→ analysis_events 저장
→ Morning Report Worker
→ daily_reports 생성
```

역할 분담:

```text
R2
= 영상 원본 저장소

gecko-vision-gate
= 가벼운 visibility/evidence prelabeler

Claude RBA Worker
= 무거운 behavior/event interpreter

Morning Report Worker
= 유저가 읽는 밤사이 보고서 생성기
```

## 4. MVP 목표

v0 목표는 작게 잡는다.

```text
- mp4 파일 입력
- 일정 간격으로 프레임 샘플링
- detector 실행
- gecko_visible 판단
- visibility_confidence 산출
- best_frame_ts 선택
- gecko_bbox 반환
- JSON 파일 출력
```

v0에서 하지 않는 것:

```text
- drinking 판정
- feeding 판정
- defecating 판정
- 건강 상태 판정
- Claude 호출
- DB/R2 직접 연동
```

## 5. 모델 후보

상용 가능성을 고려해 라이선스를 먼저 본다.

후보:

```text
1. RF-DETR core
   - Apache 2.0 계열
   - 상용 후보로 비교적 편함
   - detector 성능 검증 필요

2. YOLOX
   - Apache 2.0
   - 상용 친화
   - 라이브러리/운영 편의성 검토 필요

3. Ultralytics YOLOv8/YOLO11
   - 연구 편의성은 좋음
   - 기본 AGPL-3.0 / production은 Enterprise License 필요 가능성 큼
   - 상용 전제에서는 신중
```

결정:

```text
RF-DETR core를 v0 기본 detector로 사용한다.
```

이유:

```text
- gecko visibility + bbox 추출 목적에 맞다.
- core 모델은 Apache 2.0 계열로 안내되어 상용 전환 부담이 낮은 편이다.
- YOLOX보다 최신 detector 계열이고, Ultralytics YOLO보다 라이선스 리스크가 낮다.
```

초기 비교 후보:

```text
1순위: RF-DETR core
2순위: YOLOX
비교용: Ultralytics YOLO
```

## 6. 입력

초기 로컬 스크립트 입력:

```text
local mp4 path
```

예:

```bash
uv run python -m gecko_vision_gate.prelabel \
  --input samples/clips/clip_001.mp4 \
  --output outputs/clip_001.json
```

나중에 백엔드 통합 시 입력:

```text
clip_id
r2_key
local temp mp4 path
```

## 7. 출력 JSON 계약

v0 출력 예시:

```json
{
  "clip_id": null,
  "gecko_visible": true,
  "visibility_confidence": 0.91,
  "best_frame_ts": 18.2,
  "gecko_bbox": [120, 80, 260, 190],
  "detected_objects": [
    {
      "type": "gecko",
      "confidence": 0.91,
      "bbox": [120, 80, 260, 190],
      "frame_ts": 18.2
    }
  ],
  "skip_recommendation": false,
  "skip_reason": null,
  "model_name": "to-be-decided",
  "model_version": "v0",
  "frames_sampled": 12
}
```

도마뱀이 안 보인 경우:

```json
{
  "clip_id": null,
  "gecko_visible": false,
  "visibility_confidence": 0.12,
  "best_frame_ts": null,
  "gecko_bbox": null,
  "detected_objects": [],
  "skip_recommendation": true,
  "skip_reason": "no_gecko_detected",
  "model_name": "to-be-decided",
  "model_version": "v0",
  "frames_sampled": 12
}
```

## 8. 평가 기준

가장 중요한 지표는 false negative다.

```text
false negative
= 도마뱀이 있는데 detector가 없다고 판단
= Claude 분석 전에 중요한 클립을 버릴 위험
```

초기 평가셋:

```text
gecko-present clips: 30~50개
gecko-absent clips: 30~50개
ambiguous clips: 10~20개
```

운영 환경 그대로 포함한다.

```text
- 야간 IR
- 원거리
- 가림
- 위장색
- 흐림
- 벽/바닥에 붙어 있는 자세
```

평가 리포트에 포함할 표:

```text
total clips
gecko-present clips
gecko-absent clips
true positive
false positive
true negative
false negative
recall
precision
recommended threshold
```

초기 운영 판단:

```text
confidence >= 0.7
→ Claude 분석

0.3 <= confidence < 0.7
→ Claude 분석 + review 후보

confidence < 0.3
→ skip 후보
→ 단, 하루 N개는 샘플 검수
```

## 9. 백엔드 통합 요청안

백엔드 개발자에게 넘길 때의 요청 문장:

```text
R2에 event clip 업로드가 끝나고 camera_clips row가 생성되면,
YOLO/RF-DETR prelabel worker가 이 프로젝트의 JSON contract로 clip을 처리하게 해줘.

결과는 clip_prelabels에 저장하고,
gecko_visible=true 또는 visibility_confidence>=threshold인 경우에만
Claude analysis_jobs를 생성해줘.

단, 초기에는 no_gecko 결과를 바로 폐기하지 말고 skipped_candidate로 저장하고
하루 N개 샘플 검수할 수 있게 해줘.
```

필요한 백엔드 테이블:

```text
clip_prelabels
- id
- clip_id
- gecko_visible
- visibility_confidence
- best_frame_ts
- gecko_bbox
- detected_objects
- skip_recommendation
- skip_reason
- model_name
- model_version
- created_at
```

## 10. 전달 패키지 목표

백엔드 개발자에게 최종 전달할 패키지:

```text
gecko-vision-gate/
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
├── src/
│   └── gecko_vision_gate/
│       ├── __init__.py
│       ├── prelabel.py
│       ├── frame_sampling.py
│       ├── detector.py
│       └── schema.py
├── samples/
│   ├── clips/
│   └── outputs/
├── docs/
│   ├── JSON_SCHEMA.md
│   ├── MODEL_NOTES.md
│   └── EVAL_REPORT.md
└── tests/
```

## 11. 개발 순서

1단계: 모델 후보 결정

```text
RF-DETR core / YOLOX / Ultralytics 비교
라이선스와 설치 난이도 확인
```

2단계: 로컬 mp4 1개 처리

```text
프레임 추출
detector 실행
JSON 출력
```

3단계: 평가셋 구성

```text
gecko present / absent / ambiguous 분리
```

4단계: batch 평가

```text
여러 영상 처리
precision / recall / false negative 계산
threshold 후보 도출
```

5단계: 백엔드 전달용 정리

```text
README
JSON schema
eval report
sample outputs
integration notes
```

## 12. 지금 기억할 핵심

```text
gecko-vision-gate는 행동 분석기가 아니다.
Claude 앞단에서 영상의 시각 단서를 가볍게 뽑는 gate다.
```

가장 중요한 성공 조건:

```text
도마뱀이 있는 영상을 놓치지 않는 것.
```

초기에는 비용 절감보다 안정성이 먼저다.

```text
skip보다 priority 조정
자동 폐기보다 샘플 검수
행동 확정보다 evidence 저장
```
