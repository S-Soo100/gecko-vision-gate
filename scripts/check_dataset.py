"""데이터셋 무결성 가드 — 설계서 §13 품질게이트 / §14.10.

    uv run python scripts/check_dataset.py

검사:
  (1) test split 은 100% operational  — 누수 가드. 위반 시 exit 1 (하드 실패).
  (2) domain 값이 설계서 §7 허용집합({ir_night,occluded,ir_night_occluded,negative,day_other}) 안.
  (3) crawl_* 이미지의 source_metadata 기록률 ≥ 95% (§13).

manifest.csv 와 source_metadata.csv 를 읽기만 한다(수정 없음).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from gecko_vision_gate.hardcase_fetch import ALLOWED_DOMAINS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "manifest.csv"
SOURCE_META = ROOT / "datasets" / "source_metadata.csv"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def main() -> int:
    rows = load_rows(MANIFEST)
    if not rows:
        print(f"manifest 없음/빈값: {MANIFEST}")
        return 0
    meta_files = {r["filename"] for r in load_rows(SOURCE_META)}

    print(f"manifest {len(rows)}행")
    print("  source:", dict(Counter(r["source"] for r in rows)))
    print("  split :", dict(Counter(r.get("split") or "(빈)" for r in rows)))
    print("  domain:", dict(Counter(r.get("domain") or "(빈)" for r in rows)))

    hard_fail = False

    # (1) test = operational only
    test_bad = [r for r in rows if r.get("split") == "test" and r["source"] != "operational"]
    if test_bad:
        hard_fail = True
        print(f"\n❌ [§14.10] test split 에 비-운영 {len(test_bad)}행 (예: {test_bad[0]['filename']})")
    else:
        n_test = sum(1 for r in rows if r.get("split") == "test")
        print(f"\n✅ [§14.10] test split operational-only (test {n_test}행)")

    # (2) domain 허용집합
    dom_bad = sorted({r["domain"] for r in rows if r.get("domain") and r["domain"] not in ALLOWED_DOMAINS})
    if dom_bad:
        print(f"⚠ [§7] 비표준 domain 값: {dom_bad} (허용: {list(ALLOWED_DOMAINS)})")
    else:
        print("✅ [§7] domain 값 모두 허용집합 안")

    # (3) crawl_* 출처기록률
    crawl = [r for r in rows if r["source"].startswith("crawl_")]
    if crawl:
        covered = sum(1 for r in crawl if r["filename"] in meta_files)
        rate = covered / len(crawl) * 100
        mark = "✅" if rate >= 95 else "⚠"
        print(f"{mark} [§13] crawl_* 출처기록률 {rate:.0f}% ({covered}/{len(crawl)}, 목표 95%)")
    else:
        print("· crawl_* 이미지 아직 없음 (출처기록률 N/A)")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
