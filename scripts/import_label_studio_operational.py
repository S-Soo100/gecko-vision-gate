"""Label Studio COCO export(운영 프레임) → manifest + coco/annotations/{train,val,test}.

운영 프레임은 이미 raw/operational/<clip_id>/ 에 있고 manifest 에도 등록돼 있다(labeled=no).
이 스크립트는 LS 에서 그린 gecko bbox(positives)와 빈 제출(negatives)을 읽어:
  - manifest 운영 행을 labeled=yes + split(clip 단위) + domain(negative/빈칸) 으로 갱신
  - 박스를 coco/annotations/<split>.json 에 병합 (train 은 roboflow 와 합쳐짐, val/test 첫 생성)
이미지는 이미 raw 에 있으므로 복사하지 않는다(LS file_name 의 .../datasets/raw/ 뒤로 매칭).

설계서 §4.2·§12: 운영은 clip 단위 split, test 는 운영만. negatives(박스 0)는 false-positive
억제용으로 각 split 에 image-only(annotation 0)로 포함한다.

    uv run python scripts/import_label_studio_operational.py --src ~/Downloads/project-1-coco --dry-run
    uv run python scripts/import_label_studio_operational.py --src ~/Downloads/project-1-coco
    uv run python scripts/check_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from gecko_vision_gate.coco_import import (
    GECKO_CATEGORY,
    MANIFEST_COLUMNS,
    assign_clip_splits,
    gecko_category_ids,
    raw_rel_from_path,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
MANIFEST = ROOT / "datasets" / "manifest.csv"
ANN_DIR = ROOT / "datasets" / "coco" / "annotations"
SPLITS = ("train", "val", "test")


def find_result_json(src: Path) -> Path:
    cands = sorted(src.rglob("result.json")) or sorted(src.rglob("*.json"))
    if not cands:
        raise SystemExit(f"COCO json 못 찾음: {src}")
    return cands[0]


def load_manifest() -> dict[str, dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as fp:
        return {r["filename"]: r for r in csv.DictReader(fp)}


def load_split_coco(split: str) -> dict:
    p = ANN_DIR / f"{split}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"images": [], "annotations": [], "categories": [GECKO_CATEGORY]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label Studio 운영 프레임 COCO 인입")
    ap.add_argument("--src", required=True, help="unzip 한 LS COCO export 폴더(result.json 포함)")
    ap.add_argument("--seed", type=int, default=42, help="clip split 재현용")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    data = json.loads(find_result_json(src).read_text(encoding="utf-8"))
    gids = gecko_category_ids(data.get("categories", []))
    if not gids:
        raise SystemExit(f"gecko 클래스 없음: categories={data.get('categories')}")

    boxes_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for a in data.get("annotations", []):
        if a.get("category_id") in gids:
            boxes_by_img[a["image_id"]].append([float(x) for x in a["bbox"]])

    manifest = load_manifest()
    items: list[tuple[str, str, int, int, list]] = []  # (rel, clip, w, h, boxes)
    unmatched: list[str] = []
    missing_raw: list[str] = []
    zero_wh = 0
    for im in data.get("images", []):
        rel = raw_rel_from_path(im.get("file_name", ""))
        if not rel or rel.count("/") < 2 or rel not in manifest:
            unmatched.append(im.get("file_name", ""))
            continue
        if not (RAW / rel).exists():
            missing_raw.append(rel)
        w = int(im.get("width") or 0)   # LS COCO 는 width/height 를 null 로 줄 수 있음
        h = int(im.get("height") or 0)
        if not w or not h:
            zero_wh += 1
        items.append((rel, rel.split("/")[1], w, h, boxes_by_img.get(im["id"], [])))

    split_of = assign_clip_splits([c for _, c, _, _, _ in items], seed=args.seed)

    clips_per: dict[str, set] = defaultdict(set)
    stat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # split -> [imgs, boxes, negs]
    for rel, clip, w, h, boxes in items:
        s = split_of[clip]
        stat[s][0] += 1
        stat[s][1] += len(boxes)
        stat[s][2] += 0 if boxes else 1
        clips_per[s].add(clip)

    print(f"운영 이미지 {len(items)} 매칭 · 미매칭 {len(unmatched)} · raw없음 {len(missing_raw)} · width/height 0 {zero_wh}")
    for s in SPLITS:
        print(f"  {s:5}: clips {len(clips_per[s])} · images {stat[s][0]} · boxes {stat[s][1]} · negatives {stat[s][2]}")
    print(f"  test clips: {sorted(clips_per['test'])}")
    if unmatched:
        print(f"  ⚠ 미매칭 {len(unmatched)} 예: {unmatched[:2]}")
    if missing_raw:
        print(f"  ⚠ raw없음 {len(missing_raw)} 예: {missing_raw[:2]}")
    if args.dry_run:
        print("[dry-run] 기록 안 함.")
        return 0

    # LS COCO 가 width/height 를 null 로 줘서 raw 이미지에서 실제 dims 보강
    if zero_wh:
        import cv2
        fixed = []
        for rel, clip, w, h, boxes in items:
            if not (w and h):
                img = cv2.imread(str(RAW / rel))
                if img is not None:
                    h, w = img.shape[:2]
            fixed.append((rel, clip, w, h, boxes))
        items = fixed

    # COCO 병합 (split별, file_name 중복 스킵 = 멱등)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        coco = load_split_coco(s)
        have = {im["file_name"] for im in coco["images"]}
        nid = max((im["id"] for im in coco["images"]), default=0) + 1
        aid = max((a["id"] for a in coco["annotations"]), default=0) + 1
        for rel, clip, w, h, boxes in items:
            if split_of[clip] != s or rel in have:
                continue
            coco["images"].append({"id": nid, "file_name": rel, "width": w, "height": h})
            for b in boxes:
                coco["annotations"].append({
                    "id": aid, "image_id": nid, "category_id": 1,
                    "bbox": b, "area": float(b[2] * b[3]), "iscrowd": 0,
                })
                aid += 1
            nid += 1
        (ANN_DIR / f"{s}.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    # manifest 갱신 (운영 행만)
    for rel, clip, w, h, boxes in items:
        row = manifest[rel]
        row["labeled"] = "yes"
        row["split"] = split_of[clip]
        row["domain"] = "negative" if not boxes else (row.get("domain") or "")
    with MANIFEST.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        wr.writeheader()
        for rel in sorted(manifest):
            wr.writerow({k: manifest[rel].get(k, "") for k in MANIFEST_COLUMNS})

    print(f"✅ 운영 {len(items)}장 → manifest(labeled/split/domain) + coco {SPLITS} 병합")
    print("다음: uv run python scripts/check_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
