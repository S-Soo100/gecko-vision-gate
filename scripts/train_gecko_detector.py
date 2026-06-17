"""RF-DETR gecko detector fine-tune — coco/annotations/{train,val,test} → 학습 + test recall.

RF-DETR(Roboflow)는 dataset_dir/{train,valid,test}/_annotations.coco.json 레이아웃을 기대한다.
우리 coco(file_name=raw 상대경로)를 그 레이아웃으로 materialize(이미지는 symlink — 복사 X)한 뒤
RFDETRNano(기본)로 fine-tune 하고, test 에 대해 **image-level recall**(gate 핵심 지표)을 잰다.

    # 데이터 준비만 검증 (학습 X · 네트워크 X) — 로컬에서 안전
    uv run python scripts/train_gecko_detector.py --build-only

    # 스모크 (split당 8장·1 epoch, 파이프라인 검증 — 사전학습 가중치 다운로드됨)
    uv run python scripts/train_gecko_detector.py --smoke

    # 본 학습 (GPU 권장 — Mac 은 느림)
    uv run python scripts/train_gecko_detector.py --epochs 50 --batch-size 4

설계서: recall 우선(false negative 최소화 §7). 외부데이터는 train 만, test=운영(§4.2).
RF-DETR config 는 pydantic extra='forbid' 이라 확인된 필드만 넘긴다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gecko_vision_gate.coco_import import flatten_name, subset_coco, to_rfdetr_coco

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
ANN = ROOT / "datasets" / "coco" / "annotations"
OUR_TO_RF = {"train": "train", "val": "valid", "test": "test"}  # 우리 split → RF-DETR 폴더


def materialize(build: Path, limit: int = 0) -> dict:
    """우리 coco/annotations/{split}.json → build/{train,valid,test}/ (이미지 symlink)."""
    counts: dict[str, int] = {}
    for our_split, rf_split in OUR_TO_RF.items():
        src_json = ANN / f"{our_split}.json"
        if not src_json.exists():
            counts[rf_split] = 0
            continue
        coco = subset_coco(json.loads(src_json.read_text(encoding="utf-8")), limit)
        # raw 에 실제로 있는 이미지만 (dangling symlink 방지)
        present = [im for im in coco["images"] if (RAW / im["file_name"]).exists()]
        keep = {im["id"] for im in present}
        coco = {
            "images": present,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
            "categories": coco.get("categories", []),
        }
        d = build / rf_split
        d.mkdir(parents=True, exist_ok=True)
        for im in present:
            dst = d / flatten_name(im["file_name"])
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink((RAW / im["file_name"]).resolve(), dst)
        (d / "_annotations.coco.json").write_text(json.dumps(to_rfdetr_coco(coco), ensure_ascii=False), encoding="utf-8")
        counts[rf_split] = len(present)
    return counts


GATE_THRESHOLDS = (0.1, 0.25, 0.5, 0.7)  # §7 게이트 threshold 후보


def eval_recall(model, thresholds: tuple[float, ...] = GATE_THRESHOLDS) -> dict:
    """test image-level recall(gate 지표)을 여러 confidence 임계값에서.

    각 양성 이미지의 '게코 최고 confidence'를 모아 임계값별 recall/FN + 음성에서의 FP 를 낸다.
    recall 우선 게이트라 단일 임계값이 아니라 trade-off 곡선을 본다(§7: 0.3/0.7 경계 결정용).
    """
    model.optimize_for_inference()  # 경고 해소 + 추론 일관성
    test = json.loads((ANN / "test.json").read_text(encoding="utf-8"))
    pos_ids = {a["image_id"] for a in test["annotations"]}
    floor = min(thresholds)
    pos_conf: list[float] = []
    neg_conf: list[float] = []
    for im in test["images"]:
        dets = model.predict(str(RAW / im["file_name"]), threshold=floor)
        conf_arr = getattr(dets, "confidence", None)  # supervision Detections.confidence (ndarray|None)
        mc = float(max(conf_arr)) if conf_arr is not None and len(conf_arr) else 0.0
        (pos_conf if im["id"] in pos_ids else neg_conf).append(mc)
    out: dict = {"test_positives": len(pos_conf), "test_negatives": len(neg_conf), "by_threshold": {}}
    for t in thresholds:
        hits = sum(1 for c in pos_conf if c >= t)
        out["by_threshold"][f"{t}"] = {
            "recall": round(hits / len(pos_conf), 4) if pos_conf else None,
            "false_negatives": len(pos_conf) - hits,
            "false_positives_on_neg": sum(1 for c in neg_conf if c >= t),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RF-DETR gecko detector fine-tune")
    ap.add_argument("--model", default="nano", choices=["nano", "small", "medium", "base"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", default=str(ROOT / "runs" / "gecko_v0"))
    ap.add_argument("--dataset-dir", default=str(ROOT / "datasets" / "rfdetr_build"))
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--accelerator", default=None, help="auto/cpu/mps/gpu (미지정=RF-DETR 자동)")
    ap.add_argument("--build-only", action="store_true", help="materialize+검증만 (학습 X)")
    ap.add_argument("--skip-build", action="store_true", help="기존 build 재사용")
    ap.add_argument("--smoke", action="store_true", help="split당 소수·1 epoch 파이프라인 검증")
    ap.add_argument("--limit", type=int, default=0, help="split당 이미지 상한 (smoke 기본 8)")
    args = ap.parse_args(argv)

    build = Path(args.dataset_dir)
    limit = args.limit or (8 if args.smoke else 0)
    if not args.skip_build:
        counts = materialize(build, limit)
        print(f"materialize → {build}")
        for s in ("train", "valid", "test"):
            print(f"  {s:5}: {counts.get(s, 0)} images")

    from rfdetr.datasets.coco import is_valid_coco_dataset
    if not is_valid_coco_dataset(str(build)):
        raise SystemExit(f"RF-DETR 데이터셋 검증 실패: {build}/train/_annotations.coco.json 확인")
    print("is_valid_coco_dataset: True ✅")
    if args.build_only:
        print("[build-only] 학습 안 함.")
        return 0

    from rfdetr import RFDETRBase, RFDETRMedium, RFDETRNano, RFDETRSmall
    model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase}[args.model]()

    train_kwargs = dict(
        dataset_dir=str(build),
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output,
        run_test=True,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        seed=42,
    )
    if args.accelerator:
        train_kwargs["accelerator"] = args.accelerator
    print(f"train: model={args.model} {train_kwargs}")
    model.train(**train_kwargs)

    metrics = eval_recall(model)
    print("=== test (image-level gate, recall@threshold) ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "gate_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 저장: {args.output}/gate_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
