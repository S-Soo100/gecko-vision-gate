"""hardcase_fetch 순수 로직 테스트 (네트워크 0 · 파일은 tmp_path 만).

실제 SerpApi 호출/다운로드는 사람이 키로 직접 검증하고, 여기선 파싱·해시·dedup·
체크포인트(resume) 경계를 못 박는다.
"""

import json

from gecko_vision_gate.hardcase_fetch import (
    ALLOWED_DOMAINS,
    ALLOWED_SOURCES,
    SOURCE_META_COLUMNS,
    Query,
    dedupe_by_url,
    ext_from,
    load_checkpoint,
    load_queries,
    mark_query,
    parse_serpapi_images,
    resume_start_page,
    save_checkpoint,
    sha256_bytes,
    source_meta_row,
    staged_filename,
    validate_queries,
)


# ── 검색어 큐 ──

def test_load_queries_dict_and_defaults(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"queries": [
        {"query": "gecko night vision", "domain_hint": "ir_night", "target_source": "crawl_nighttime"},
    ]}), encoding="utf-8")
    qs = load_queries(p)
    assert len(qs) == 1
    assert qs[0].query == "gecko night vision"
    assert qs[0].enabled is True  # 기본값


def test_validate_queries_flags_bad_domain_and_source():
    bad = [
        Query("ok", "ir_night", "crawl_nighttime"),          # 정상
        Query("x", "daytime", "crawl_google"),                # domain 비표준
        Query("y", "occluded", "raw_negative"),               # source 비표준
        Query("", "occluded", "crawl_google"),                # 빈 query
    ]
    problems = validate_queries(bad)
    assert len(problems) == 3
    assert any("daytime" in p for p in problems)
    assert any("raw_negative" in p for p in problems)


def test_validate_queries_skips_disabled():
    qs = [Query("parked", "negative", "raw_negative", enabled=False)]
    assert validate_queries(qs) == []  # disabled 는 검증 제외


def test_real_query_file_is_valid():
    # 실제 큐 파일이 스키마를 지키는지 (회귀 방지)
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    qs = load_queries(root / "datasets" / "crawl_queries.json")
    assert validate_queries(qs) == []
    assert all(q.domain_hint in ALLOWED_DOMAINS for q in qs)
    assert all(q.target_source in ALLOWED_SOURCES for q in qs)
    assert len(qs) >= 20  # §8.1~8.3 = 23개


# ── SerpApi 파싱 ──

def test_parse_serpapi_images_extracts_and_skips_missing_original():
    payload = {"images_results": [
        {"original": "https://a.com/g.jpg", "link": "https://a.com/post", "source": "a.com", "title": "T"},
        {"thumbnail": "https://b.com/t.jpg"},  # original 없음 → 스킵
        {"original": "https://c.com/x.png", "link": "https://c.com/p", "license": "CC"},
    ]}
    hits = parse_serpapi_images(payload)
    assert len(hits) == 2
    assert hits[0].image_url == "https://a.com/g.jpg"
    assert hits[0].page_url == "https://a.com/post"
    assert hits[1].license_hint == "CC"


def test_parse_serpapi_empty():
    assert parse_serpapi_images({}) == []


def test_dedupe_by_url_across_calls():
    from gecko_vision_gate.hardcase_fetch import ImageHit
    seen: set[str] = set()
    a = [ImageHit("u1", "p1"), ImageHit("u2", "p2")]
    b = [ImageHit("u2", "p2"), ImageHit("u3", "p3")]  # u2 중복
    assert len(dedupe_by_url(a, seen)) == 2
    assert len(dedupe_by_url(b, seen)) == 1  # u3 만 신규


# ── 해시 / 파일명 ──

def test_ext_from_prefers_url_then_content_type():
    assert ext_from("https://x.com/a.png") == ".png"
    assert ext_from("https://x.com/a.JPEG") == ".jpg"          # jpeg → jpg 정규화
    assert ext_from("https://x.com/a.webp?w=5") == ".webp"     # 쿼리스트링 무시
    assert ext_from("https://x.com/noext", "image/png") == ".png"
    assert ext_from("https://x.com/noext") == ".jpg"           # 최후 기본값


def test_staged_filename_is_hash_deterministic():
    data = b"\x89PNG fake bytes"
    h = sha256_bytes(data)
    f1 = staged_filename("ir_night", h, ext_from("x.png"))
    f2 = staged_filename("ir_night", sha256_bytes(data), ".png")
    assert f1 == f2  # 같은 바이트 → 같은 경로 (dedup 자연 성립)
    assert f1.startswith("ir_night/ir_night_")
    assert f1.endswith(".png")


# ── 메타데이터 ──

def test_source_meta_row_has_design_columns():
    row = source_meta_row(
        filename="crawl_nighttime/ir_night_abc.jpg",
        search_query="gecko night vision",
        source_url="https://a.com/g.jpg",
        page_url="https://a.com/post",
        downloaded_at="2026-06-17T10:00:00",
        notes="domain_hint=ir_night",
    )
    assert tuple(row.keys()) == SOURCE_META_COLUMNS  # §6 컬럼·순서 일치
    assert row["collector"] == "serpapi"  # 기본값


# ── 체크포인트 / resume ──

def test_checkpoint_roundtrip_and_resume(tmp_path):
    p = tmp_path / "ckpt.json"
    ckpt = load_checkpoint(p)            # 없으면 빈 구조
    assert resume_start_page(ckpt, "q") == 0   # 첫 실행 → 0페이지부터
    mark_query(ckpt, "q", last_page=2, downloaded=37)
    save_checkpoint(p, ckpt)

    reloaded = load_checkpoint(p)
    assert resume_start_page(reloaded, "q") == 3       # 마지막 완료(2)+1
    assert reloaded["queries"]["q"]["downloaded"] == 37
    assert resume_start_page(reloaded, "other") == 0   # 미진행 쿼리는 0
