"""Google hard-case 이미지 수집 — 순수 로직 (네트워크/파일 I/O 없음).

설계: docs/GOOGLE_HARDCASE_CRAWLING_DESIGN.md
수집 방식은 SerpApi google_images. (Google Custom Search JSON API 는 2025 신규마감·
2027.1.1 종료라 대체.) 사람이 hard-case 만 선별하는 반자동 파이프라인에서 '후보를
떠오는' 단계의 계산만 담는다 — 검색어 큐 로딩 · SerpApi 응답 파싱 · 파일명/해시 산출 ·
메타데이터 행 · 체크포인트. 전부 단위 테스트 대상.

실제 HTTP(requests) · 파일 쓰기 · 재시도 루프는 scripts/fetch_hardcase_images.py 가
담당한다 (feedle 교훈: 재시도·매건 중간저장·resume 는 1차 구현부터).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# domain 태그는 설계서 §7 로 통일 (README 의 day/closeup/distant 는 폐기 — recall 약점이
# 'hard case' 축이라 ir/occlusion 기준이 맞다). manifest.csv 의 domain 컬럼이 이 값을 쓴다.
ALLOWED_DOMAINS = ("ir_night", "occluded", "ir_night_occluded", "negative", "day_other")
# SerpApi 후보를 떨구는 raw 하위 소스. negative 는 운영프레임에서 뽑으므로(§4.1) 큐에 없음.
ALLOWED_SOURCES = ("crawl_nighttime", "crawl_google")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# 설계서 §6 권장 컬럼 — manifest.csv 와 분리한 보수적 선택(§16). build_manifest.py 의
# 6컬럼 스키마를 건드리지 않고 출처를 별도 보관.
SOURCE_META_COLUMNS = (
    "filename", "search_query", "source_url", "page_url",
    "license_hint", "downloaded_at", "collector", "notes",
)
# staging 작업대장 — promote 전까지의 중간 기록(매 다운로드마다 append).
STAGING_META_COLUMNS = (
    "staged_path", "sha256", "search_query", "domain_hint", "target_source",
    "source_url", "page_url", "license_hint", "downloaded_at",
)


@dataclass(frozen=True)
class Query:
    query: str
    domain_hint: str
    target_source: str
    enabled: bool = True
    note: str = ""


@dataclass(frozen=True)
class ImageHit:
    image_url: str           # SerpApi 'original' — 원본 이미지 URL
    page_url: str            # 'link' — 이미지를 담은 페이지 URL
    source: str = ""         # 'source' — 출처 도메인/사이트명
    title: str = ""
    license_hint: str = ""   # 'license' (대개 비어있음 — v0 는 rights 기록을 미룸 §15)


def load_queries(path: str | Path) -> list[Query]:
    """검색어 큐 JSON → Query 리스트. dict({queries:[...]}) 또는 list 둘 다 허용."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["queries"] if isinstance(raw, dict) else raw
    return [
        Query(
            query=it["query"],
            domain_hint=it["domain_hint"],
            target_source=it["target_source"],
            enabled=it.get("enabled", True),
            note=it.get("note", ""),
        )
        for it in items
    ]


def validate_queries(queries: list[Query]) -> list[str]:
    """enabled 큐의 정합성 문제를 문자열 리스트로 반환(빈 리스트면 정상)."""
    problems: list[str] = []
    for i, q in enumerate(queries):
        if not q.enabled:
            continue
        if not q.query.strip():
            problems.append(f"[{i}] 빈 query")
        if q.domain_hint not in ALLOWED_DOMAINS:
            problems.append(f"[{i}] '{q.query}': domain_hint '{q.domain_hint}' 비표준 {ALLOWED_DOMAINS}")
        if q.target_source not in ALLOWED_SOURCES:
            problems.append(f"[{i}] '{q.query}': target_source '{q.target_source}' 비표준 {ALLOWED_SOURCES}")
    return problems


def parse_serpapi_images(payload: dict) -> list[ImageHit]:
    """SerpApi google_images 응답 → ImageHit 리스트.

    images_results[*] 의 original(이미지) · link(페이지) · source · title · license 를 뽑는다.
    필드 누락에 방어적(.get). original 이 없는 항목은 스킵(썸네일만 있는 광고 등).
    """
    hits: list[ImageHit] = []
    for r in payload.get("images_results", []):
        img = r.get("original") or r.get("original_image") or ""
        if not img:
            continue
        hits.append(
            ImageHit(
                image_url=img,
                page_url=r.get("link", "") or r.get("source", ""),
                source=r.get("source", ""),
                title=r.get("title", ""),
                license_hint=str(r.get("license", "")),
            )
        )
    return hits


def dedupe_by_url(hits: list[ImageHit], seen_urls: set[str]) -> list[ImageHit]:
    """이미 본 image_url 은 거른다(다운로드 전 1차 dedup). seen_urls 를 갱신."""
    out: list[ImageHit] = []
    for h in hits:
        if h.image_url in seen_urls:
            continue
        seen_urls.add(h.image_url)
        out.append(h)
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ext_from(url: str, content_type: str = "") -> str:
    """확장자 결정: URL 우선 → content-type → 최후 .jpg. jpeg 는 .jpg 로 정규화."""
    url_l = url.lower().split("?")[0]
    for e in IMG_EXT:
        if url_l.endswith(e):
            return ".jpg" if e == ".jpeg" else e
    ct = content_type.lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def staged_filename(domain_hint: str, sha256: str, ext: str) -> str:
    """staging 내 결정적 파일명: <domain_hint>/<domain_hint>_<sha12><ext>.

    해시 기반이라 같은 바이트면 같은 경로 → 파일해시 dedup 이 자연 성립(§10 1단계).
    domain_hint 를 파일명에 박아 promote 후에도 도메인이 눈에 보인다.
    """
    if not ext.startswith("."):
        ext = "." + ext
    return f"{domain_hint}/{domain_hint}_{sha256[:12]}{ext}"


def source_meta_row(
    *,
    filename: str,
    search_query: str,
    source_url: str,
    page_url: str,
    license_hint: str = "",
    downloaded_at: str = "",
    collector: str = "serpapi",
    notes: str = "",
) -> dict[str, str]:
    """설계서 §6 source_metadata.csv 한 행. filename 은 raw 상대경로(manifest 와 조인 키)."""
    return {
        "filename": filename,
        "search_query": search_query,
        "source_url": source_url,
        "page_url": page_url,
        "license_hint": license_hint,
        "downloaded_at": downloaded_at,
        "collector": collector,
        "notes": notes,
    }


# ── 체크포인트 (resume) — feedle 교훈: 매 페이지 진행상황 저장, 끊겨도 이어받기 ──

def load_checkpoint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"queries": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("queries", {})
    return data


def save_checkpoint(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def query_last_page(ckpt: dict, query: str) -> int:
    """해당 query 의 마지막 완료 페이지(없으면 -1)."""
    return ckpt.get("queries", {}).get(query, {}).get("last_page", -1)


def mark_query(ckpt: dict, query: str, last_page: int, downloaded: int) -> None:
    ckpt.setdefault("queries", {})[query] = {"last_page": last_page, "downloaded": downloaded}


def resume_start_page(ckpt: dict, query: str) -> int:
    """resume 시 시작 페이지 = 마지막 완료 + 1 (feedle 의 last_page+1 패턴)."""
    return query_last_page(ckpt, query) + 1
