# R0002 — 운영 저녁/시간대 데이터 확대 → v2 (저녁 은신 게코 recall 회복)

| | |
|---|---|
| **번호** | R0002 |
| **날짜** | 2026-07-06 (작업 07-03~06) |
| **상태** | ✅확정 |
| **모델 버전** | v1 → **v2** (RF-DETR Nano) |
| **관련** | 커밋 `<이 커밋>` · 리포트 [[R0001]] · 메모 [[gecko-evening-retreat]] · [[round2-ls-review-handoff]] |

## 1. 배경·동기 (Context)
v1 의 잔여 약점은 **환경/시간대 다양성**(같은 펫캠 2일치·주간 편중, R0001 §7)이었다. 오늘(07-03) 운영 클립을 게이트로 실측하니 **저녁 18시부터 검출 급락(KST 15→19시 100%→0%)**. 규명 결과 야간 IR FN 이 아니라 **게코의 정상 은신**(채도 15~19시 64~69 일정=IR 전환 無)이었지만([[gecko-evening-retreat]]), 그 저녁·가림 프레임에 **v1 이 놓친 게코가 다수** 섞여 있었다 → 확보·학습 기회.

## 2. 가설·목표 (Hypothesis / Goal)
오늘 운영 클립(특히 저녁 시간대)을 사람검수해 v1 이 놓친 게코를 확보·재학습하면, 저녁/가림 recall 이 오를 것이다.
- 성공 기준: **오늘 저녁 프레임이 포함된 새 test 에서 v2 recall > v1** (같은 잣대).

## 3. 방법 (Method — 재현 가능하게)
- **데이터**: R2 `p4cam-79b5d844/20260703-*` **148클립** 다운로드 → `batch_prelabel.py`(모델 1회 로드) 게이트 실측(게코 86·neg 62·conf중앙 0.82) → `extract_operational_frames.py --glob '20260703-*' --per-class 0 --frames 6` → **886프레임** → `autolabel.py --glob '20260703-*/*' --checkpoint runs/gecko_v1/… --conf 0.25` → COCO 초안 453박스.
- **사람검수(Label Studio)**: 로컬파일 서빙 + `export_to_label_studio.py`(color/ir 태깅). **886장 전수 검수** → COCO export(project-5).
- **인입**: `import_label_studio_operational.py` → clip 단위 split. **train 2149→2770 · val 180→311 · test 170→300 · negative 445→590**. `check_dataset.py` 통과(test 운영전용·누수 없음).
- **검수 효과**: 게코박스 453→**738**(+285) · box0 433→**145**(즉 v1 이 "게코 없음"이라던 ~288장에 실제 게코 존재).
- **학습 config**: `train_gecko_detector.py --model nano --epochs 30 --batch-size 4 --accelerator mps --output runs/gecko_v2` (seed 42 · MPS ~12분/epoch). **epoch 20 에서 사용자 판단 조기중단**(epoch 6 이후 mAP 실질 수렴 0.779→0.7845, 미세개선이 early-stopping 을 계속 리셋) → `checkpoint_best_ema.pth`(epoch 20, val mAP 0.7845).
- **평가**: `eval_gate.py --checkpoint v1,v2 --split test --mode both`. test 300 = **frame 230pos/70neg · clip 45pos/23neg**. 게이트 레벨(conf≥t 검출 있으면 "게코 있음"). **v1 도 같은 새 test 재평가**(공정 비교).

## 4. 결과 (Results)
**프레임단위 — 같은 test 230 pos/70 neg:**
| conf | v1 recall / FP | v2 recall / FP |
|---|---|---|
| 0.25 | 0.800 · 8/70 | **0.983** · 21/70 |
| 0.5 | 0.713 · 5/70 | 0.943 · 12/70 |
| 0.7 | 0.648 · 2/70 | 0.909 · **4/70** |

**클립단위(게이트 실동작) — 같은 test 45 pos/23 neg:**
| conf | v1 recall / FP | v2 recall / FP |
|---|---|---|
| 0.25 | 0.844 · 2/23 | **0.978** · 5/23 |
| 0.5 | 0.800 · 1/23 | 0.933 · 3/23 |
| 0.7 | 0.756 · 0/23 | 0.933 · 1/23 |

→ conf 0.25 에서 **recall 급등**(frame 0.80→0.98 · clip 0.84→0.98). **단 FP 도 증가**(frame 8→21 · clip 2→5). 고threshold 에선 v2 가 recall·FP 모두 v1 압도(v2@0.7 frame 0.909·4/70 vs v1@0.25 0.800·8/70).

## 5. 분석·해석 (Analysis)
- **recall 급등 = 검수 효과의 직접 증거**. v1 이 놓친 저녁 은신·가림 게코 285박스를 사람이 채워 학습 → v2 가 같은 유형을 잡는다. 저녁 저검출(운영 관측)의 근본 해소.
- **FP 증가 = recall 과의 trade-off**. v2 가 더 공격적으로 검출해 빈 프레임 오발동↑. 그러나 **iso-FP 비교에서 v2 우위**: 같은 FP 수준으로 threshold 를 올리면(v2@0.7) recall 이 v1 을 크게 앞서고 FP 는 오히려 낮다. → 모델이 나빠진 게 아니라 동작점(operating point)이 이동한 것.
- **클립 FP > 프레임 FP** 경향은 R0001 과 동일(음성 클립은 프레임 하나만 오발동해도 fire).

## 6. 결정 (Decisions)
- 게이트 기본 체크포인트를 **`runs/gecko_v2/checkpoint_best_ema.pth` 로 교체**한다.
- **게이트 conf 를 0.25 → 0.5 로 잠정 상향**한다(frame recall 0.943·FP 12/70, clip recall 0.933·FP 3/23 — recall 유지하며 FP 억제). 더 보수적으로는 0.7(clip FP 1/23). 실운영 검증 후 확정.
- v1(`runs/gecko_v1`)은 비교용 보존.

## 7. 한계·위협 (Limitations)
- **FP 증가**: conf 0.25 동작점에서 v1 대비 FP 상승. threshold 상향으로 관리하나 **근본 억제(저녁 hard-negative 확대)는 미완**.
- **학습 미완주**: epoch 20/30 조기중단(미세개선 정체). 완주 시 소폭 개선 여지 있으나 best_ema 기준 변화 미미할 것으로 판단.
- **검수 4장 누락**: 886→882(0.5%, LS skip 추정).
- **환경 단일성 지속**: 시간대(저녁)만 추가됐을 뿐 **같은 펫캠·같은 사육장·게코 1마리(릴리화이트)**. 다른 카메라/모프 일반화 미검증.
- **모델 크기 고정**: v2 nano. `--model small` A/B 미실험.
- **test 성장**: test 170→300(오늘 130 추가) → R0001 의 옛 test(170) 지표와 **직접비교 금지**. 본 라운드는 v1 을 새 test 로 재평가해 공정 비교함.

## 8. 다음 단계 (Next)
1. **FP 억제** — 저녁/컬러 hard-negative(v2 오발동 21 frame 유형) 분석·추가 or threshold 정밀 튜닝.
2. **`--model small` A/B**(neg 590 로 여유) — nano v2 대비.
3. **환경 다양성**(근본 잔여 약점) — 다른 카메라·사육장·모프.
4. **게이트 conf 0.5 실운영 검증** → 확정.

## 9. 아티팩트·링크 (Artifacts)
- 체크포인트: `runs/gecko_v2/checkpoint_best_ema.pth`(v2) · `runs/gecko_v1/checkpoint_best_total.pth`(v1 비교) — ⚠ `runs/` gitignore, 로컬 보존.
- 데이터: `datasets/manifest.csv` · `datasets/coco/annotations/{train,val,test}.json`(이 커밋).
- 원본/검수: `~/petcam-lab/storage/p4cam-79b5d844/`(mp4) · LS `project-5` · export `~/Downloads/project-5-at-2026-07-06-07-08-9fb56231`.
- eval: `runs/eval_v1_v2.json` · `scripts/eval_gate.py` · `scripts/batch_prelabel.py`.
- 메모리: [[gecko-evening-retreat]] · [[round2-ls-review-handoff]]. 로드맵: `docs/MODEL_AND_TRAINING_PLAN.md §9`.
