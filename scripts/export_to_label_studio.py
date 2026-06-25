"""autolabel COCO → Label Studio tasks JSON (사전주석 + 로컬파일 서빙 + 검수 태그).

autolabel.py 출력 COCO 의 v0/v1 박스를 LS 의 편집가능한 prediction 으로 싣고,
data.tag(ir_boxed/color_box0 …)로 검수 우선순위 필터를 만든다.

    uv run python scripts/export_to_label_studio.py \
        --coco datasets/autolabel/<source>.coco.json \
        --out datasets/autolabel/<source>.ls_tasks.json [--doc-root datasets/raw]

LS 기동: `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`
        `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo>/datasets/raw` label-studio start
→ Import 로 이 JSON 업로드(소스 스토리지 불필요). 검수 후 Export COCO →
import_label_studio_operational.py. bbox 는 원본 dim 대비 %.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image"><Label value="gecko" background="green"/></RectangleLabels>
</View>"""


def kind_of(path: Path) -> str:
    """프레임 도메인 휴리스틱: black / ir(흑백) / color."""
    s = cv2.resize(cv2.imread(str(path)), (48, 48))
    spread = (s.max(2).astype(int) - s.min(2).astype(int)).mean()
    return "black" if s.mean() < 12 else ("ir" if spread < 12 else "color")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="COCO → Label Studio tasks JSON")
    ap.add_argument("--coco", required=True, help="autolabel COCO")
    ap.add_argument("--out", required=True, help="출력 tasks JSON")
    ap.add_argument("--doc-root", default=str(ROOT / "datasets" / "raw"),
                    help="LS LOCAL_FILES_DOCUMENT_ROOT (file_name 이 이 아래 상대경로여야)")
    ap.add_argument("--model-version", default="gecko_v0")
    ap.add_argument("--no-tag", action="store_true", help="ir/color 분류 태그 생략(빠름)")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.coco).read_text())
    anns = defaultdict(list)
    for a in doc["annotations"]:
        anns[a["image_id"]].append(a)
    doc_root = Path(args.doc_root)

    tasks, tagcount = [], defaultdict(int)
    for im in doc["images"]:
        W, H, fn = im["width"], im["height"], im["file_name"]
        boxes = anns.get(im["id"], [])
        tag = "untagged"
        if not args.no_tag:
            tag = f"{kind_of(doc_root / fn)}_{'boxed' if boxes else 'box0'}"
            tagcount[tag] += 1
        results = [{
            "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": b["bbox"][0] / W * 100, "y": b["bbox"][1] / H * 100,
                      "width": b["bbox"][2] / W * 100, "height": b["bbox"][3] / H * 100,
                      "rotation": 0, "rectanglelabels": ["gecko"]},
            "score": b.get("score", 0),
        } for b in boxes]
        tasks.append({
            "data": {"image": "/data/local-files/?d=" + fn, "tag": tag, "clip": fn.split("/")[1] if "/" in fn else fn},
            "predictions": [{"model_version": args.model_version, "result": results}] if results else [],
        })

    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"✅ {len(tasks)} tasks → {args.out}")
    if tagcount:
        print("태그 분포(LS data.tag 필터):", dict(sorted(tagcount.items(), key=lambda x: -x[1])))
    print("LS 라벨 설정(Labeling Interface):\n" + LABEL_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
