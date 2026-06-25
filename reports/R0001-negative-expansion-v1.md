# R0001 — negative 확대 → v1 (야간 IR false-positive 억제)

| | |
|---|---|
| **번호** | R0001 |
| **날짜** | 2026-06-25 (작업 06-24~25) |
| **상태** | ✅확정 |
| **모델 버전** | v0 → **v1** (RF-DETR Nano) |
| **관련** | 커밋 `875a912`(데이터) · `52735d5`(문서) · 메모 [[negative-video-handoff]] · [[lily-white-ir-confound]] |

## 1. 배경·동기 (Context)
v0 는 test(28장)에서 recall@0.25=1.00·"FP 0" 으로 좋아 보였지만, **그 test 에 negative 가 0장**이라 false-positive 가 측정조차 안 된 상태였다. 파일럿에서 관측된 v0 FP 는 **전부 야간 IR**(은신처 실루엣·유리볼 글레어 오인). 게이트로 쓰려면 야간 IR negative 로 FP 를 실측·억제해야 했다.

## 2. 가설·목표 (Hypothesis / Goal)
**같은 펫캠의 야간 IR negative**(FP 가 실제로 나는 도메인)를 대량 학습시키면 recall 손실 없이 FP 가 줄 것이다.
- 성공 기준: negative 가 포함된 test 에서 **FP 대폭 감소 + recall 유지(≥0.95)**.

## 3. 방법 (Method — 재현 가능하게)
- **데이터 확보**: R2 `petcam-clips/terra-clips/clips/p4cam-79b5d844/`(06-22~23 녹화) → mp4 290클립(1.8GB)+썸네일 287 다운로드(`~/petcam-lab/storage/p4cam-79b5d844/`). R2 접근=`~/petcam-lab/.env` 자격증명 + petcam-lab venv boto3.
- **프레임 추출**: `extract_operational_frames.py --source-dir … --per-class 0 --frames 6` → **1675 프레임**(`datasets/raw/operational/2026062*`).
- **autolabel**: `autolabel.py --checkpoint runs/gecko_v0/… --conf 0.25` → bbox 초안 COCO.
- **사람 검수(Label Studio)**: 프로젝트 `p4cam-neg-round1`, 로컬파일 서빙 + COCO→LS tasks 변환(검수 태그). 1632 제출(pos 464 · neg 1168 · skip 6; 미제출 37 스킵).
- **near-dupe 솎기**: 정적 장면 6프레임 거의 동일(neg 1168 중 1114 가 near-dupe) → **negative 클립당 2장 cap=420**, positive 464 전부 유지.
- **인입**: `import_label_studio_operational.py` → clip 단위 split. **train 1557→2149 · val 30→180 · test 28→170(neg 56·야간 포함) · negative domain 25→445**. `check_dataset.py` 통과(test 운영전용·누수 없음).
- **학습 config**: `train_gecko_detector.py --output runs/gecko_v1 --model nano --epochs 30 --accelerator mps --skip-build` (seed 42 · batch 4 · grad_accum 4 · lr 1e-4 · early-stopping patience 10). MPS, ~9분/epoch.
- **평가 셋업**: 게이트 레벨(이미지에 conf≥t 검출 있으면 "게코 있음"). test 170프레임=114 pos/56 neg = **46클립(25 pos/21 neg)**. **공정 비교 위해 v0 도 같은 새 test 에 재평가**(`/tmp/eval_v0_v1.py` 프레임 · `/tmp/eval_clip.py` 클립 — ⚠ /tmp 임시, scripts/ 로 승격 필요).

## 4. 결과 (Results)
**클립단위(게이트 실제 동작) — 같은 test 25 pos/21 neg:**
| conf | v0 recall / FP | v1 recall / FP |
|---|---|---|
| 0.25 | 1.00 · **19/21** | 0.96 · **2/21** |
| 0.5 | 0.88 · 6/21 | 0.92 · 1/21 |
| 0.7 | 0.64 · 2/21 | 0.88 · 0/21 |

**프레임단위 — 같은 test 114 pos/56 neg:**
| conf | v0 recall / FP | v1 recall / FP |
|---|---|---|
| 0.25 | 0.982 · 41/56 | 0.982 · 8/56 |
| 0.5 | 0.807 · 18/56 | 0.921 · 5/56 |
| 0.7 | 0.544 · 7/56 | 0.895 · 2/56 |

→ conf 0.25 에서 **recall 유지(클립 0.96·프레임 0.982)하며 FP 클립 19→2(−89%)·프레임 41→8(−80%)**. 고threshold 에선 v1 이 recall 도 크게 우위(v0 는 붕괴).

## 5. 분석·해석 (Analysis)
- v0 의 "FP 0" 은 **neg-0 test 착시**였다. 같은 신규 test 에서 v0 는 빈 클립 21개 중 19개(90%)에 오발동 — 사실상 "거의 모든 클립에 게코"라 외치는 무용 게이트.
- **근본 원인**: 대상 게코가 **릴리화이트(전신 흰색)** → 야간 IR 에서 흰 몸이 **흰 인조넝쿨·관엽식물 흰무늬·유리/물그릇 IR 글레어**와 같은 밝기. v0 가 "창백한 덩어리=게코"로 과일반화. negative(특히 검수자가 지운 글레어 가짜박스)가 이를 교정.
- **클립 FP > 프레임 FP**: 음성 클립은 프레임 하나만 오발동해도 오발동 클립 → 프레임 많을수록 기회↑. 게이트는 클립단위로 평가해야 정직.

## 6. 결정 (Decisions)
- 게이트 기본 체크포인트를 **`runs/gecko_v1/checkpoint_best_total.pth` 로 교체**한다(문서·명령 갱신 완료, `52735d5`).
- **게이트 conf = 0.25 유지**(클립 recall 0.96·FP 2/21; recall 우선 게이트 역할에 부합, FP 는 이미 충분히 낮음).
- v0 (`runs/gecko_v0`)는 비교용 보존.

## 7. 한계·위협 (Limitations)
- **환경 단일성**: 여전히 **같은 펫캠·2일치 녹화·게코 1마리(릴리화이트)**. 다른 카메라/사육장/계절·다른 모프엔 일반화 미검증.
- **표본 작음**: test 클립 neg 21·pos 25 → 0.25↔0.5 차이는 1~2클립=노이즈 수준.
- **near-dupe cap=2 는 휴리스틱**(시간 양끝 선택). 더 정교한 다양성 선별 여지.
- **재현 아티팩트 미보존**: 모델(`runs/`)은 gitignore, eval 스크립트는 `/tmp`(임시). → scripts/ 승격 + 결과 docs 보존으로 보완.

## 8. 다음 단계 (Next)
1. **환경/시간대 다양성 확대**(최우선 잔여 약점) — 새 카메라·사육장·계절·다른 모프.
2. eval 스크립트 `scripts/eval_gate.py` 로 승격(임의 체크포인트, frame+clip, threshold sweep).
3. §6 운영 mp4 클립단위 파이프라인 적용 검증.
4. (선택) `--model small`(neg 445 로 해금) — v1 nano 대비 A/B.

## 9. 아티팩트·링크 (Artifacts)
- 체크포인트: `runs/gecko_v1/checkpoint_best_total.pth`(v1) · `runs/gecko_v0/…`(v0 비교용) — ⚠ `runs/` gitignore, 로컬 보존.
- 데이터: `datasets/manifest.csv` · `datasets/coco/annotations/{train,val,test}.json`(커밋 `875a912`).
- 원본/검수: `~/petcam-lab/storage/p4cam-79b5d844/`(mp4) · LS 프로젝트 `p4cam-neg-round1` · export `~/Downloads/project-2-…-coco`.
- eval: `/tmp/eval_v0_v1.py`(프레임) · `/tmp/eval_clip.py`(클립) — 임시, 승격 대상.
- 메모리: [[negative-video-handoff]] · [[lily-white-ir-confound]]. 가이드: `docs/NEGATIVE_DATA_GUIDELINE.md` · `docs/MODEL_AND_TRAINING_PLAN.md §9`.
