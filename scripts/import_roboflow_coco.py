"""Roboflow(또는 임의) COCO export → raw/ + manifest + coco/annotations/<split>.json.

외부 '라벨 딸린' 데이터셋 인입 경로. 크롤(staging→promote)과 달리 bbox 가 이미 있어
바로 train 으로 들어간다. 설계서 §4.2: 외부 데이터는 절대 test 에 넣지 않는다.

    # 1) Roboflow Universe: Download Dataset → Format: COCO → "zip to computer" → unzip
    # 2) 점검 (복사·기록 안 함)
    uv run python scripts/import_roboflow_coco.py \
        --src ~/Downloads/gecko.v1i.coco \
        --source-name roboflow_gecko_teddychiu \
        --license "Public Domain" \
        --source-url "https://universe.roboflow.com/teddychiu/gecko" \
        --dry-run
    # 3) 실제 인입 → 가드
    uv run python scripts/import_roboflow_coco.py --src ... --source-name ... --license "Public Domain"
    uv run python scripts/check_dataset.py

동작: Roboflow COCO 의 gecko 클래스만 우리 단일 gecko(id 1) 로 remap, 비-gecko 박스·
gecko 없는 이미지 폐기, 해시 dedup, 이미지→raw/<source-name>/, manifest(labeled=yes)·
source_metadata 기록, 박스→coco/annotations/<split>.json 머지. 그들의 train/valid/test 는
전부 우리 <split>(기본 train) 으로 평탄화한다. 멱등: 같은 sha 는 재실행해도 중복 안 됨.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from gecko_vision_gate.coco_import import (
    GECKO_CATEGORY,
    MANIFEST_COLUMNS,
    collect_gecko_images,
    gecko_category_ids,
)
from gecko_vision_gate.hardcase_fetch import SOURCE_META_COLUMNS, sha256_bytes, source_meta_row

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
MANIFEST = ROOT / "datasets" / "manifest.csv"
SOURCE_META = ROOT / "datasets" / "source_metadata.csv"
ANN_DIR = ROOT / "datasets" / "coco" / "annotations"


def find_coco_jsons(src: Path) -> list[Path]:
    """Roboflow 표준 _annotations.coco.json 우선, 없으면 coco 형태 *.json 폴백."""
    js = sorted(src.rglob("_annotations.coco.json"))
    if js:
        return js
    out: list[Path] = []
    for p in sorted(src.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            out.append(p)
    return out


def load_coco_out(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"images": [], "annotations": [], "categories": [GECKO_CATEGORY]}


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as fp:
        return {r["filename"]: r for r in csv.DictReader(fp)}


def load_meta_filenames() -> set[str]:
    if not SOURCE_META.exists():
        return set()
    with SOURCE_META.open(newline="", encoding="utf-8") as fp:
        return {r["filename"] for r in csv.DictReader(fp)}


def resolve_image(json_dir: Path, file_name: str) -> Path | None:
    cand = json_dir / file_name
    if cand.exists():
        return cand
    cand = json_dir / Path(file_name).name  # json 이 prefix 경로를 붙인 경우
    return cand if cand.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="외부 COCO 데이터셋 인입 (gecko 단일클래스)")
    ap.add_argument("--src", required=True, help="unzip 한 COCO export 폴더")
    ap.add_argument("--source-name", required=True, help="raw/<이름>/ + manifest source")
    ap.add_argument("--license", default="", help="source_metadata.license_hint (예: 'Public Domain')")
    ap.add_argument("--source-url", default="", help="데이터셋 출처 URL")
    ap.add_argument("--gecko-classes", default="gecko", help="gecko 로 매핑할 클래스명 패턴(쉼표)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="우리 split (test 금지 §4.2)")
    ap.add_argument("--domain", default="day_other")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"src 없음: {src}")
    patterns = tuple(s.strip() for s in args.gecko_classes.split(",") if s.strip())
    jsons = find_coco_jsons(src)
    if not jsons:
        raise SystemExit(f"COCO json 못 찾음 under {src} — 'Download Dataset → Format: COCO' 인지 확인")
    print(f"COCO json {len(jsons)}개: {[p.parent.name + '/' + p.name for p in jsons]}")

    out_coco = load_coco_out(ANN_DIR / f"{args.split}.json")
    next_img = max((im["id"] for im in out_coco["images"]), default=0) + 1
    next_ann = max((a["id"] for a in out_coco["annotations"]), default=0) + 1
    manifest = load_manifest()
    meta_files = load_meta_filenames()

    added: list[tuple[Path, str, object]] = []  # (src_path, rel, GeckoImage)
    dup = missing = dropped_imgs = box_total = 0
    seen_rel: set[str] = set()
    for jp in jsons:
        data = json.loads(jp.read_text(encoding="utf-8"))
        gids = gecko_category_ids(data.get("categories", []), patterns)
        if not gids:
            print(f"  ! {jp.parent.name}: gecko 클래스 없음 → 스킵")
            continue
        gimgs = collect_gecko_images(data, gids)
        dropped_imgs += len(data.get("images", [])) - len(gimgs)
        for g in gimgs:
            sp = resolve_image(jp.parent, g.file_name)
            if sp is None:
                missing += 1
                continue
            sha = sha256_bytes(sp.read_bytes())
            ext = ".jpg" if sp.suffix.lower() == ".jpeg" else sp.suffix.lower()
            rel = f"{args.source_name}/{sha[:12]}{ext}"
            if rel in manifest or rel in seen_rel:
                dup += 1
                continue
            seen_rel.add(rel)
            added.append((sp, rel, g))
            box_total += len(g.boxes)

    print(f"gecko 이미지 {len(added)} (박스 {box_total}) · 중복스킵 {dup} · 파일없음 {missing} · gecko없어제외 {dropped_imgs}")
    print(f"  → raw/{args.source_name}/ · split={args.split} · domain={args.domain} · license='{args.license or '(미기재)'}'")
    if args.dry_run:
        print("[dry-run] 복사·기록 안 함.")
        return 0

    (RAW / args.source_name).mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_rows = []
    for sp, rel, g in added:
        shutil.copy2(sp, RAW / rel)  # 외부 폴더라 move 아닌 copy (원본 보존)
        iid = next_img
        next_img += 1
        out_coco["images"].append({"id": iid, "file_name": rel, "width": g.width, "height": g.height})
        for box in g.boxes:
            out_coco["annotations"].append({
                "id": next_ann, "image_id": iid, "category_id": 1,
                "bbox": [float(x) for x in box], "area": float(box[2] * box[3]), "iscrowd": 0,
            })
            next_ann += 1
        manifest[rel] = {
            "filename": rel, "source": args.source_name, "clip_id": "",
            "split": args.split, "labeled": "yes", "domain": args.domain,
        }
        if rel not in meta_files:
            meta_rows.append(source_meta_row(
                filename=rel, search_query="", source_url=args.source_url, page_url=args.source_url,
                license_hint=args.license, downloaded_at=now, collector="roboflow",
                notes=f"COCO import; orig={Path(g.file_name).name}",
            ))

    (ANN_DIR / f"{args.split}.json").write_text(json.dumps(out_coco, ensure_ascii=False), encoding="utf-8")
    with MANIFEST.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for rel in sorted(manifest):
            w.writerow({k: manifest[rel].get(k, "") for k in MANIFEST_COLUMNS})
    new = not SOURCE_META.exists()
    with SOURCE_META.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(SOURCE_META_COLUMNS))
        if new:
            w.writeheader()
        w.writerows(meta_rows)

    print(f"✅ {len(added)}장 → raw/{args.source_name}/ · coco/annotations/{args.split}.json · manifest · source_metadata")
    print("다음: uv run python scripts/check_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
