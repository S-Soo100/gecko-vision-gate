# GME Presence Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GME 분석을 안전한 `detected_candidate/not_observed/unresolved` Python 계약으로 제공한다.

**Architecture:** 판정은 `GMEAnalysis`를 입력받는 순수 함수로 분리하고, 영상 분석과 RF-DETR 생성은 얇은 편의 함수가 조합한다. 결과 계약에는 입력 경로나 비밀값을 넣지 않는다.

**Tech Stack:** Python 3.12, dataclasses, pytest, 기존 GME/OpenCV/RF-DETR adapter

## Global Constraints

- `not_observed`는 게코 부재 확정이 아니다.
- unresolved를 B 경로에 합치지 않는다.
- 행동명·GT·하이라이트·VLM route·원본 삭제를 결정하지 않는다.
- 테스트를 먼저 실패시킨 뒤 최소 구현한다.

---

### Task 1: 3상태 판정 계약

**Files:**
- Create: `src/gecko_vision_gate/gme_presence.py`
- Test: `tests/test_gme_presence.py`

**Interfaces:**
- Consumes: `GMEAnalysis`, GME `Detector`, `GMEConfig`
- Produces: `PresenceResult`, `decide_presence()`, `analyze_presence()`, `analyze_presence_with_gate()`

- [x] 직접 관측·미관측·unknown·camera motion·분석 실패의 실패 테스트를 작성한다.
- [x] `uv run pytest tests/test_gme_presence.py -q`로 누락된 모듈 실패를 확인한다.
- [x] 순수 판정 함수와 직렬화 가능한 결과 dataclass를 최소 구현한다.
- [x] 영상 분석 및 Gate detector 편의 함수를 구현한다.
- [x] 전용 테스트와 전체 `uv run pytest -q`를 통과시킨다.

### Task 2: 소비자 사용 계약

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1의 `analyze_presence_with_gate()`
- Produces: 서버 개발자용 최소 Python 예제와 상태별 저장 규칙

- [x] README에 복사 가능한 Python 예제를 추가한다.
- [x] 문서가 부재 확정·자동 삭제를 허용하지 않는지 검토한다.
- [x] diff와 전체 테스트 결과를 확인한다.
