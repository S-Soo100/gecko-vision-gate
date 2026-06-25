"""게이트 평가 — 임의 체크포인트 · frame/clip level · threshold sweep.

게이트 레벨("이미지/클립에 conf≥t 검출이 있으면 게코 있음")로 recall·FP 측정.
train_gecko_detector 가 학습 직후 내는 gate_metrics(단일 모델·이미지 단위)와 달리,
**임의 체크포인트 비교 + clip 단위**를 지원한다(R0001 의 v0 vs v1 A/B 가 이걸로 수행됨).

    uv run python scripts/eval_gate.py --checkpoint runs/gecko_v1/checkpoint_best_total.pth
    uv run python scripts/eval_gate.py \
        --checkpoint runs/gecko_v0/checkpoint_best_total.pth,runs/gecko_v1/checkpoint_best_total.pth \
        --mode both --out runs/eval_v0_v1.json

clip 단위: rfdetr_build/<split> 의 flat 파일명(`operational__<clip>__fNNN.jpg`)을 clip_id 로 묶어
"어떤 프레임이든 conf≥t 면 클립 fire" 로 집계 → mp4 불필요.
⚠ 버전 비교는 반드시 같은 split(test) 에서(R0001 교훈: 다른 test 의 지표 직접비교 금지).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from gecko_vision_gate.detector import GeckoDetector

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="게이트 평가(frame/clip · threshold sweep)")
    ap.add_argument("--checkpoint", required=True, help=".pth (콤마구분 다중 = 비교)")
    ap.add_argument("--build-dir", default=str(ROOT / "datasets" / "rfdetr_build"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresholds", default="0.1,0.25,0.5,0.7")
    ap.add_argument("--mode", choices=["frame", "clip", "both"], default="both")
    ap.add_argument("--det-threshold", type=float, default=0.1, help="검출 하한(이하 무시)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장(선택)")
    args = ap.parse_args(argv)

    split_dir = Path(args.build_dir) / args.split
    doc = json.loads((split_dir / "_annotations.coco.json").read_text())
    boxed = {a["image_id"] for a in doc["annotations"]}
    frames = []  # (clip, is_pos, file_name)
    for im in doc["images"]:
        clip = im["file_name"].rsplit("__", 1)[0].split("operational__", 1)[-1]
        frames.append((clip, im["id"] in boxed, im["file_name"]))
    THRESH = [float(t) for t in args.thresholds.split(",")]
    ckpts = [c.strip() for c in args.checkpoint.split(",")]

    clip_gt: dict[str, bool] = defaultdict(bool)
    for clip, isp, _ in frames:
        clip_gt[clip] = clip_gt[clip] or isp
    n_fpos = sum(p for _, p, _ in frames)
    n_fneg = len(frames) - n_fpos
    n_cpos = sum(1 for v in clip_gt.values() if v)
    n_cneg = sum(1 for v in clip_gt.values() if not v)
    print(f"{args.split}: frame {n_fpos}pos/{n_fneg}neg · clip {n_cpos}pos/{n_cneg}neg")

    results: dict = {}
    for ck in ckpts:
        det = GeckoDetector(checkpoint=ck, threshold=args.det_threshold)
        fmax = {fn: max((d.confidence for d in det.detect(cv2.imread(str(split_dir / fn)))), default=0.0)
                for _, _, fn in frames}
        name = Path(ck).parent.name
        print(f"\n=== {name} ({ck}) ===")
        hdr = f"{'thr':>5}"
        if args.mode in ("frame", "both"):
            hdr += f" | {'FRAME rec/FP':>16}"
        if args.mode in ("clip", "both"):
            hdr += f" | {'CLIP rec/FP':>16}"
        print(hdr)
        row: dict = {}
        for t in THRESH:
            line, rec = f"{t:>5}", {}
            if args.mode in ("frame", "both"):
                fr = sum(1 for _, p, fn in frames if p and fmax[fn] >= t) / n_fpos
                ffp = sum(1 for _, p, fn in frames if (not p) and fmax[fn] >= t)
                line += f" | {fr:>7.3f} {ffp:>3}/{n_fneg:<4}"
                rec["frame"] = {"recall": round(fr, 4), "fp": ffp, "n_neg": n_fneg}
            if args.mode in ("clip", "both"):
                fire: dict[str, bool] = defaultdict(bool)
                for c, _, fn in frames:
                    fire[c] = fire[c] or (fmax[fn] >= t)
                cr = sum(1 for c in clip_gt if clip_gt[c] and fire[c]) / n_cpos
                cfp = sum(1 for c in clip_gt if (not clip_gt[c]) and fire[c])
                line += f" | {cr:>7.3f} {cfp:>3}/{n_cneg:<4}"
                rec["clip"] = {"recall": round(cr, 4), "fp": cfp, "n_neg": n_cneg}
            print(line)
            row[str(t)] = rec
        results[name] = row

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
