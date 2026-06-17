"""datasets/raw/ 스캔 → datasets/manifest.csv 생성·갱신 (멱등).

데이터 인입 워크플로우의 핵심 도구. 새 이미지를 raw/<source>/ 에 넣고 재실행하면:
  - 신규 파일 → 새 행 추가 (split/labeled/domain 은 빈값 → 나중에 사람이 채움)
  - 기존 파일 → 사람이 채운 메타(split/labeled/domain) 보존
  - raw 에서 삭제된 파일 → manifest 에서도 제거

    uv run python scripts/build_manifest.py [--dry-run]

source 는 raw/ 바로 아래 폴더명에서 자동 도출. operational 은 영상단위 하위폴더
(raw/operational/<clip_id>/) 를 두므로 clip_id 도 자동 추출한다.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
MANIFEST = ROOT / "datasets" / "manifest.csv"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
COLUMNS = ["filename", "source", "clip_id", "split", "labeled", "domain"]
PRESERVE = ("split", "labeled", "domain")  # 사람이 채우는 값 → 재실행 시 보존


def scan_raw() -> dict[str, dict]:
    """raw/ 하위 모든 이미지 → {상대경로: row} (신규 기본값으로)."""
    rows: dict[str, dict] = {}
    for f in sorted(RAW.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        parts = f.relative_to(RAW).parts
        source = parts[0]
        # operational/<clip_id>/file 형태일 때만 clip_id 추출
        clip_id = parts[1] if source == "operational" and len(parts) >= 3 else ""
        rel = f.relative_to(RAW).as_posix()
        rows[rel] = {
            "filename": rel,
            "source": source,
            "clip_id": clip_id,
            "split": "",
            "labeled": "no",
            "domain": "",
        }
    return rows


def load_existing() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as fp:
        return {r["filename"]: r for r in csv.DictReader(fp)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build/update datasets/manifest.csv")
    ap.add_argument("--dry-run", action="store_true", help="기록 없이 통계만 출력")
    args = ap.parse_args(argv)

    if not RAW.exists():
        raise SystemExit(f"raw dir not found: {RAW}")

    current = scan_raw()
    existing = load_existing()

    added = 0
    for rel, row in current.items():
        if rel in existing:
            for c in PRESERVE:
                if existing[rel].get(c):
                    row[c] = existing[rel][c]
        else:
            added += 1
    removed = sum(1 for k in existing if k not in current)

    # 통계
    by_source = Counter(r["source"] for r in current.values())
    labeled = sum(1 for r in current.values() if r["labeled"] == "yes")
    print(f"raw 이미지 총 {len(current)}")
    for s, n in sorted(by_source.items()):
        print(f"  {s:18} {n}")
    print(f"신규 {added} / 기존보존 {len(current) - added} / 삭제 {removed} / labeled {labeled}")

    if args.dry_run:
        print("[dry-run] manifest 미기록")
        return 0

    with MANIFEST.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        w.writeheader()
        for rel in sorted(current):
            w.writerow(current[rel])
    print(f"✅ wrote {MANIFEST} ({len(current)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
