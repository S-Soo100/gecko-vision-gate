"""fine-tune 모델로 미라벨 이미지에 bbox 초안(pre-annotation) 생성 → COCO.

MODEL_AND_TRAINING_PLAN §5.5 step3: 초안 모델로 새 프레임 bbox 자동예측 → 사람이 검수.
처음부터 전부 손으로 그리는 부담을 줄이는 단계. 출력 COCO 를 검수툴/Label Studio 로 불러
사람이 교정(빠진 게코 추가·틀린 박스 삭제) → import → 재학습.

    uv run python scripts/autolabel.py --checkpoint runs/gecko_v0/checkpoint_best_total.pth \
        --source crawl_breeder --conf 0.25 --limit 50 --dry-run

recall 우선이라 conf 는 낮게(기본 0.25) — 놓친 게코를 사람이 새로 그리는 것보다 약한
박스를 지우는 게 싸다. category 는 우리 단일 gecko(id 1). 각 박스에 예측 score 기록.
모델 추론이라 학습(특히 MPS)과 동시 실행은 피한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gecko_vision_gate.coco_import import GECKO_CATEGORY
from gecko_vision_gate.detector import GeckoDetector

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fine-tune 모델 → COCO pre-annotation (auto-label)")
    ap.add_argument("--checkpoint", required=True, help="fine-tune .pth")
    ap.add_argument("--source", required=True, help="raw 하위 폴더명 (예: crawl_breeder)")
    ap.add_argument("--out", default=None, help="출력 COCO (기본 datasets/autolabel/<source>.coco.json)")
    ap.add_argument("--conf", type=float, default=0.25, help="검출 임계값 (낮을수록 recall↑)")
    ap.add_argument("--limit", type=int, default=0, help="이미지 상한 (0=전체)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src_dir = RAW / args.source
    if not src_dir.exists():
        raise SystemExit(f"source 없음: {src_dir}")
    imgs = sorted(p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"{args.source}: 대상 이미지 {len(imgs)} · conf≥{args.conf}")
    if args.dry_run:
        print("[dry-run] 추론·기록 안 함.")
        return 0
    if not imgs:
        print("대상 이미지 없음.")
        return 0

    import cv2

    det = GeckoDetector(checkpoint=args.checkpoint, threshold=args.conf)
    images: list[dict] = []
    annotations: list[dict] = []
    iid = aid = 1
    with_box = 0
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        rel = p.relative_to(RAW).as_posix()
        images.append({"id": iid, "file_name": rel, "width": w, "height": h})
        n = 0
        for d in det.detect(frame):
            x, y, bw, bh = d.xywh
            annotations.append({
                "id": aid, "image_id": iid, "category_id": 1,
                "bbox": [x, y, bw, bh], "area": float(bw * bh), "iscrowd": 0,
                "score": round(d.confidence, 4),  # 검수 우선순위용 (표준 COCO 엔 없는 필드)
            })
            aid += 1
            n += 1
        with_box += 1 if n else 0
        iid += 1

    out = Path(args.out) if args.out else ROOT / "datasets" / "autolabel" / f"{args.source}.coco.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [GECKO_CATEGORY]}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✅ {len(images)} 이미지 · {len(annotations)} 박스(이미지 {with_box}장) → {out}")
    print("다음: 검수툴/Label Studio 로 불러 교정 → import → 재학습")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
