"""storage 의 mp4 다수를 게이트(prelabel)로 일괄 분석 → 클립별 JSON + 요약 CSV.

    uv run python scripts/batch_prelabel.py \
        --input-dir ~/petcam-lab/storage/p4cam-79b5d844 \
        --glob '20260703-*.mp4' \
        --checkpoint runs/gecko_v1/checkpoint_best_total.pth \
        --threshold 0.25 --frames 12 \
        --out-dir outputs/prelabel/2026-07-03

prelabel CLI 는 클립 1개씩이라 모델을 매번 로드한다. 이 배치는 GeckoDetector 를
**1회 로드 후 주입**해 전 클립에 재사용한다(detector.py 설계 그대로). clip_id = 파일 stem.

산출물:
- <out-dir>/clips/<clip_id>.json  — 클립별 PrelabelResult(불변 계약, schema.py)
- <out-dir>/summary.csv           — 클립당 1행 요약(gecko_visible·conf·bbox·객체수·skip)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SUMMARY_COLS = [
    "clip_id",
    "gecko_visible",
    "visibility_confidence",
    "best_frame_ts",
    "gecko_bbox",
    "n_objects",
    "skip_reason",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="게이트(prelabel) 일괄 분석 → JSON + 요약 CSV")
    ap.add_argument("--input-dir", required=True, help="mp4 가 있는 폴더")
    ap.add_argument("--glob", default="*.mp4", help="파일 매칭 glob (예: 20260703-*.mp4)")
    ap.add_argument("--checkpoint", default=None, help="fine-tune .pth (없으면 COCO pretrained)")
    ap.add_argument("--threshold", type=float, default=0.25, help="검출 confidence 임계값")
    ap.add_argument("--frames", type=int, default=12, help="클립당 샘플 프레임 수")
    ap.add_argument("--model-size", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--out-dir", required=True, help="산출물 폴더")
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만(스모크용)")
    args = ap.parse_args(argv)

    # 무거운 import 는 인자 파싱 뒤로
    from gecko_vision_gate.detector import GeckoDetector
    from gecko_vision_gate.prelabel import prelabel_clip

    in_dir = Path(args.input_dir).expanduser()
    clips = sorted(in_dir.glob(args.glob))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"매칭 클립 없음: {in_dir}/{args.glob}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    # 모델 1회 로드 → 전 클립 재사용
    detector = GeckoDetector(
        model_size=args.model_size, threshold=args.threshold, checkpoint=args.checkpoint
    )

    rows: list[dict] = []
    n_visible = n_error = 0
    total = len(clips)
    print(f"대상 {total}개 · checkpoint={args.checkpoint or 'COCO(pretrained)'} · thr={args.threshold}")

    for i, clip in enumerate(clips, 1):
        cid = clip.stem
        try:
            res = prelabel_clip(
                clip,
                num_frames=args.frames,
                threshold=args.threshold,
                model_size=args.model_size,
                checkpoint=args.checkpoint,
                clip_id=cid,
                detector=detector,
            )
        except Exception as e:  # noqa: BLE001 — 한 클립 실패가 배치 전체를 죽이지 않게
            n_error += 1
            print(f"  [{i}/{total}] ERROR {cid}: {e}", file=sys.stderr)
            rows.append(
                dict(clip_id=cid, gecko_visible="ERROR", visibility_confidence="",
                     best_frame_ts="", gecko_bbox="", n_objects="", skip_reason=str(e)[:120])
            )
            continue

        (out_dir / "clips" / f"{cid}.json").write_text(
            json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        n_visible += int(res.gecko_visible)
        rows.append(
            dict(
                clip_id=cid,
                gecko_visible=res.gecko_visible,
                visibility_confidence=res.visibility_confidence,
                best_frame_ts="" if res.best_frame_ts is None else res.best_frame_ts,
                gecko_bbox="" if res.gecko_bbox is None else json.dumps(res.gecko_bbox),
                n_objects=len(res.detected_objects),
                skip_reason=res.skip_reason or "",
            )
        )
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} · gecko보임 {n_visible} · 에러 {n_error}", flush=True)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n완료: {total}개 · gecko_visible {n_visible} · 에러 {n_error}\n"
        f"  클립 JSON → {out_dir}/clips/\n  요약 CSV → {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
