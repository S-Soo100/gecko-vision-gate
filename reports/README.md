# 연구 리포트 (research reports)

gecko-vision-gate 의 연구 라운드별 성과를 **누적 기록**한다. **한 라운드 = 한 리포트(불변 기록)**.

## 작성 규칙
- **언제**: 데이터·모델·평가에 **의미있는 변경 + 결과**가 나온 라운드마다 1개. (trivial 수정·문서 only·탐색만 한 세션은 제외)
- **파일명**: `RNNNN-<kebab-제목>.md` — zero-pad 4자리 순번(`R0001`, `R0002`…).
- **작성**: [`TEMPLATE.md`](./TEMPLATE.md) 복사 → 9개 섹션 채움.
- **불변성(ADR 관례)**: 상태 `✅확정` 후엔 내용 수정 금지. 틀렸거나 뒤집히면 **새 리포트로 supersede** — 옛 리포트 상태를 `⛔대체됨(→ RNNNN)` 으로만 바꾼다. `진행중` 일 때만 본문 갱신.
- **정직성**: 음성 결과·실패·한계를 반드시 남긴다(재현성·신뢰). 잘된 것만 쓰면 다음 사람이 같은 함정에 빠진다.
- **같은 잣대**: 버전 비교는 **반드시 동일 test** 에서. 서로 다른 test 의 지표 직접비교 금지(R0001 교훈 — neg-0 test 가 FP 0 착시를 만들었다).
- **인덱스 갱신**: 새 리포트마다 아래 표에 한 줄(최신 위) + 진행중이면 「진행중」에.

## 설계 근거 (웹 관례 종합)
반복 ML 라운드에 맞춰 5개 표준 관례를 종합·재단:
- **ADR**(M. Nygard) — 번호·상태·불변·결정 중심 → 라운드별 불변 기록 + supersede 모델.
- **Model Cards**(Mitchell et al., Google 2018) — 모델/데이터/평가/한계 문서화 → §3·4·7 구조.
- **NeurIPS 재현성 체크리스트**(Pineau) — 데이터·config·seed·지표정의·인프라 → §3 방법(재현).
- **Keep a Changelog** — Unreleased + 역순 목록 → 아래 「진행중」·인덱스.
- **DS 실험 템플릿**(experiments/ 관례) — 폴더 단위 누적.

출처: [ADR](https://adr.github.io/) · [Model Cards](https://research.google/pubs/pub48120/) · [NeurIPS 재현성](https://arxiv.org/abs/2003.12206) · [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [DS template](https://github.com/michael-ford/data-science-development-project-template)

## 진행중
- (없음)

## 리포트 목록 (최신순)
| 번호 | 날짜 | 제목 | 핵심 결과 | 상태 |
|---|---|---|---|---|
| [R0001](./R0001-negative-expansion-v1.md) | 2026-06-25 | negative 확대 → v1 | 클립 FP 19→2 · recall@0.25 0.96 유지 | ✅확정 |
