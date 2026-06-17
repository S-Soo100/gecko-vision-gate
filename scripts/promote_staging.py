"""staging(사람 선별 후) → datasets/raw/<source>/ + source_metadata.csv.

설계: docs/GOOGLE_HARDCASE_CRAWLING_DESIGN.md §9.1(4~5단계)·§6. fetch 가 staging/
<domain>/ 에 후보를 떨군 뒤, 사람이 나쁜 후보를 폴더에서 직접 삭제(육안 선별 = §10 2단계).
남은 파일만 이 스크립트가 raw 로 옮기고 출처를 source_metadata.csv 에 확정 기록한다.

domain_hint 는 notes 에 'domain_hint=...' 로 실어 둔다 → 나중에 manifest.domain 채울 때 참고.
출처 메타가 없는 orphan(직접 끌어다 놓은 파일 등)은 기본 스킵(§13 출처기록률 95% 보호).

    uv run python scripts/promote_staging.py --dry-run
    uv run python scripts/promote_staging.py
    다음: uv run python scripts/build_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from gecko_vision_gate.hardcase_fetch import SOURCE_META_COLUMNS, source_meta_row

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "datasets" / "staging"
STAGING_META = STAGING / "staging_metadata.csv"
RAW = ROOT / "datasets" / "raw"
SOURCE_META = ROOT / "datasets" / "source_metadata.csv"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_staging_meta() -> dict[str, dict]:
    if not STAGING_META.exists():
        return {}
    with STAGING_META.open(newline="", encoding="utf-8") as fp:
        return {r["staged_path"]: r for r in csv.DictReader(fp)}


def load_promoted() -> set[str]:
    """이미 source_metadata 에 있는 filename → 중복 promote 방지."""
    if not SOURCE_META.exists():
        return set()
    with SOURCE_META.open(newline="", encoding="utf-8") as fp:
        return {r["filename"] for r in csv.DictReader(fp)}


def append_source_meta(rows: list[dict]) -> None:
    new = not SOURCE_META.exists()
    with SOURCE_META.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(SOURCE_META_COLUMNS))
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="staging → raw + source_metadata.csv")
    ap.add_argument("--dry-run", action="store_true", help="이동·기록 없이 계획만 출력")
    ap.add_argument("--include-orphans", action="store_true",
                    help="출처 메타 없는 파일도 promote(notes=orphan)")
    args = ap.parse_args(argv)

    if not STAGING.exists():
        print(f"staging 없음: {STAGING} (먼저 fetch_hardcase_images.py 실행)")
        return 0

    meta = load_staging_meta()
    promoted = load_promoted()
    remaining = [
        p for p in sorted(STAGING.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXT
    ]
    if not remaining:
        print("staging 에 이미지 없음 — 선별이 끝났거나 아직 fetch 안 함.")
        return 0

    plan: list[tuple[Path, str, dict]] = []  # (src, dest_rel, source_meta_row)
    orphans = 0
    dupes = 0
    for src in remaining:
        rel = src.relative_to(STAGING).as_posix()
        row = meta.get(rel)
        if row is None:
            if not args.include_orphans:
                orphans += 1
                continue
            target_source, dom, sq, surl, purl, lic, dt = "crawl_google", "", "", "", "", "", ""
            notes = "orphan: staging 메타 없음"
        else:
            target_source = row["target_source"]
            dom, sq = row["domain_hint"], row["search_query"]
            surl, purl, lic, dt = row["source_url"], row["page_url"], row["license_hint"], row["downloaded_at"]
            notes = f"domain_hint={dom}"
        dest_rel = f"{target_source}/{src.name}"
        if dest_rel in promoted:
            dupes += 1
            continue
        plan.append((src, dest_rel, source_meta_row(
            filename=dest_rel, search_query=sq, source_url=surl, page_url=purl,
            license_hint=lic, downloaded_at=dt, notes=notes,
        )))

    print(f"staging 이미지 {len(remaining)} → promote 대상 {len(plan)} "
          f"(orphan 스킵 {orphans} · 이미올림 {dupes})")
    by_src: dict[str, int] = {}
    for _, dest_rel, _ in plan:
        s = dest_rel.split("/")[0]
        by_src[s] = by_src.get(s, 0) + 1
    for s, n in sorted(by_src.items()):
        print(f"  → raw/{s:16} {n}")
    if orphans and not args.include_orphans:
        print(f"  ⚠ orphan {orphans}장: 출처 메타 없음 → --include-orphans 로 강제 promote 가능")

    if args.dry_run:
        print("[dry-run] 이동·기록 안 함.")
        return 0

    rows = []
    for src, dest_rel, meta_row in plan:
        dest = RAW / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        rows.append(meta_row)
    append_source_meta(rows)
    print(f"\n✅ {len(rows)}장 → raw/ 이동 + source_metadata.csv 기록")
    print("다음: uv run python scripts/build_manifest.py  (그 뒤 manifest 에서 domain/split 채우기)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
