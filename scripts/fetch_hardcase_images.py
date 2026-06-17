"""Google hard-case 후보 → datasets/staging/<domain>/ (SerpApi google_images).

설계: docs/GOOGLE_HARDCASE_CRAWLING_DESIGN.md §9. 반자동 1단계 = '후보 떠오기'.
이후 사람이 staging 에서 hard-case 만 남기고(나쁜 후보 삭제 = §10 2단계) →
promote_staging.py 로 raw/ 에 올린다. 자동 대량적재가 아니라 사람 선별 전제(§9.1·§15).

    # 키 없이 계획만 (네트워크 0)
    uv run python scripts/fetch_hardcase_images.py --dry-run

    # 실제 수집 (SerpApi 키 필요 · 무료티어 250검색/월 — 쿼터 보호)
    export SERPAPI_KEY=...                      # 또는 .env 의 SERPAPI_KEY= / --api-key
    uv run python scripts/fetch_hardcase_images.py --pages 1 --max-per-query 60 --resume

feedle 교훈: 매 다운로드 staging_metadata.csv append + 매 페이지 체크포인트 저장 +
요청 재시도. 끊겨도 --resume 이면 last_page+1 부터 이어받음. requests 는 지연 import 라
--dry-run 은 의존성 없이도 돈다.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from gecko_vision_gate.hardcase_fetch import (
    STAGING_META_COLUMNS,
    dedupe_by_url,
    ext_from,
    load_checkpoint,
    load_queries,
    mark_query,
    parse_serpapi_images,
    resume_start_page,
    save_checkpoint,
    sha256_bytes,
    staged_filename,
    validate_queries,
)

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "datasets" / "crawl_queries.json"
STAGING = ROOT / "datasets" / "staging"
META = STAGING / "staging_metadata.csv"
CKPT = STAGING / ".fetch_progress.json"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
UA = "Mozilla/5.0 (gecko-vision-gate hardcase collector)"
MIN_BYTES = 2048  # 1x1 트래커/깨진 파일 컷


def load_api_key(cli_key: str | None) -> str:
    """우선순위: --api-key > env SERPAPI_KEY > .env 의 SERPAPI_KEY=."""
    if cli_key:
        return cli_key
    if os.environ.get("SERPAPI_KEY"):
        return os.environ["SERPAPI_KEY"]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SERPAPI_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def with_retry(fn, attempts: int = 3, base: float = 1.0):
    """요청 재시도 (지수 backoff). 마지막 실패는 그대로 raise."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 네트워크/파싱 전반 재시도
            last = e
            if i < attempts - 1:
                time.sleep(base * (2**i))
    assert last is not None
    raise last


def existing_hashes() -> set[str]:
    """staging_metadata 의 기존 sha256 → 재실행/resume 시 재다운로드 방지(§10 1단계)."""
    out: set[str] = set()
    if META.exists():
        with META.open(newline="", encoding="utf-8") as fp:
            out = {r["sha256"] for r in csv.DictReader(fp) if r.get("sha256")}
    return out


def append_meta(row: dict) -> None:
    """매 다운로드마다 즉시 append (끊겨도 받은 만큼 보존)."""
    new = not META.exists()
    with META.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(STAGING_META_COLUMNS))
        if new:
            w.writeheader()
        w.writerow(row)


def looks_image(content_type: str, url: str) -> bool:
    if "image" in content_type.lower():
        return True
    u = url.lower().split("?")[0]
    return any(u.endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp"))


def serpapi_images(session, key: str, query: str, page: int, timeout: int = 30) -> dict:
    params = {"engine": "google_images", "q": query, "ijn": str(page), "api_key": key}
    r = session.get(SERPAPI_ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download(session, url: str, timeout: int = 30) -> tuple[bytes, str]:
    r = session.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def print_plan(active, pages: int, max_per_query: int) -> None:
    print(f"검색어 {len(active)}개 × {pages}페이지 → SerpApi 호출 {len(active) * pages}회 "
          f"(무료티어 250/월 기준 {250 // max(1, len(active) * pages)}회 실행분)")
    print(f"쿼리당 최대 {max_per_query}장 → 후보 상한 ≈ {len(active) * max_per_query}장 "
          f"(사람이 hard-case 만 선별 → 실제 채택은 일부)\n")
    by_dom: dict[str, int] = {}
    for q in active:
        by_dom[q.domain_hint] = by_dom.get(q.domain_hint, 0) + 1
    for dom, n in sorted(by_dom.items()):
        print(f"  {dom:18} {n} 쿼리")
    print("\n[dry-run] 네트워크 호출 없음. 실제 수집: SERPAPI_KEY 설정 후 --dry-run 빼고 실행.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SerpApi → staging hard-case 후보 수집")
    ap.add_argument("--query-file", default=str(QUERIES))
    ap.add_argument("--pages", type=int, default=1, help="쿼리당 SerpApi 페이지 수(ijn). 1=~100결과")
    ap.add_argument("--max-per-query", type=int, default=60, help="쿼리당 최대 다운로드 수")
    ap.add_argument("--limit", type=int, default=0, help="이번 실행 전체 다운로드 상한(0=무제한, 스모크용)")
    ap.add_argument("--resume", action="store_true", help="체크포인트에서 last_page+1 부터 이어받기")
    ap.add_argument("--include-disabled", action="store_true", help="enabled=false 쿼리도 포함")
    ap.add_argument("--api-key", default=None, help="SerpApi 키(미지정 시 env/.env)")
    ap.add_argument("--dry-run", action="store_true", help="계획만 출력(네트워크 0)")
    args = ap.parse_args(argv)

    queries = load_queries(args.query_file)
    problems = validate_queries(queries)
    if problems:
        print("❌ 검색어 큐 정합성 오류:")
        for p in problems:
            print(f"  - {p}")
        return 2
    active = [q for q in queries if q.enabled or args.include_disabled]
    if not active:
        print("활성 쿼리 없음.")
        return 0

    if args.dry_run:
        print_plan(active, args.pages, args.max_per_query)
        return 0

    key = load_api_key(args.api_key)
    if not key:
        print("❌ SerpApi 키 없음. export SERPAPI_KEY=... 또는 .env 에 SERPAPI_KEY= 추가 후 재실행.")
        print("   (무료 키: https://serpapi.com 가입 → 250검색/월)")
        return 2

    import requests  # 지연 import — dry-run 은 requests 없이도 동작

    STAGING.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    ckpt = load_checkpoint(CKPT) if args.resume else {"queries": {}}
    seen_urls: set[str] = set()
    hashes = existing_hashes()
    total = 0

    for q in active:
        start = resume_start_page(ckpt, q.query) if args.resume else 0
        dl_q = ckpt.get("queries", {}).get(q.query, {}).get("downloaded", 0) if args.resume else 0
        print(f"▶ '{q.query}' [{q.domain_hint} → {q.target_source}] page {start}..{start + args.pages - 1}")
        for page in range(start, start + args.pages):
            try:
                payload = with_retry(lambda: serpapi_images(session, key, q.query, page))
            except Exception as e:  # noqa: BLE001
                print(f"  ! SerpApi 실패(page {page}): {e} — 이 쿼리 중단")
                break
            if payload.get("error"):
                print(f"  ! SerpApi error: {payload['error']} — 중단(쿼터 소진 가능)")
                save_checkpoint(CKPT, ckpt)
                return 1
            hits = dedupe_by_url(parse_serpapi_images(payload), seen_urls)
            for h in hits:
                if dl_q >= args.max_per_query or (args.limit and total >= args.limit):
                    break
                try:
                    content, ctype = with_retry(lambda: download(session, h.image_url))
                except Exception:  # noqa: BLE001 — 죽은 링크 흔함, 스킵
                    continue
                if len(content) < MIN_BYTES or not looks_image(ctype, h.image_url):
                    continue
                sha = sha256_bytes(content)
                if sha in hashes:
                    continue
                hashes.add(sha)
                rel = staged_filename(q.domain_hint, sha, ext_from(h.image_url, ctype))
                dest = STAGING / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
                append_meta({
                    "staged_path": rel,
                    "sha256": sha,
                    "search_query": q.query,
                    "domain_hint": q.domain_hint,
                    "target_source": q.target_source,
                    "source_url": h.image_url,
                    "page_url": h.page_url,
                    "license_hint": h.license_hint,
                    "downloaded_at": _now(),
                })
                dl_q += 1
                total += 1
            mark_query(ckpt, q.query, last_page=page, downloaded=dl_q)
            save_checkpoint(CKPT, ckpt)  # 매 페이지 저장
            if args.limit and total >= args.limit:
                break
        print(f"  · 누적 {dl_q}장")
        if args.limit and total >= args.limit:
            print(f"--limit {args.limit} 도달 → 종료")
            break

    print(f"\n✅ staging 다운로드 {total}장 → {STAGING}")
    print("다음: staging/<domain>/ 에서 나쁜 후보 삭제 → uv run python scripts/promote_staging.py")
    return 0


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
