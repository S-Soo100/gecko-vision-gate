"""Gate evidence provenance — 어떤 모델/체크포인트/샘플러로 뽑았는지 불변 기록.

지시문 §164/§205: clip_prelabels 에 model_name·model_version·checkpoint SHA-256·threshold·
sampler/schema version 을 남긴다. 재현성과 감사를 위해서다("이 evidence 를 어떤 artifact 가
만들었나"). 체크포인트 hash 는 실행 환경에서 실측한다(문서 값 신뢰 금지).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# 샘플러 알고리즘 버전 (frame_sampling: 균등 N장). 알고리즘이 바뀌면 올린다.
SAMPLER_VERSION = "even-uniform-v1"
# evidence 스키마 버전 (clip_prelabels 저장 구조). 필드가 바뀌면 올린다 → 멱등 키 일부.
SCHEMA_VERSION = "gate-evidence-v1"


@lru_cache(maxsize=8)
def checkpoint_sha256(path: str | Path) -> str:
    """체크포인트 파일의 SHA-256 (streaming). 같은 경로는 캐시(운영 체크포인트 불변 전제)."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class GateProvenance:
    """clip_prelabels 의 provenance 컬럼과 1:1. frozen = 사실, mutate 금지."""

    model_name: str
    model_version: str
    checkpoint_sha256: str
    threshold: float
    sampler_version: str
    schema_version: str
    frames_sampled: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold": self.threshold,
            "sampler_version": self.sampler_version,
            "schema_version": self.schema_version,
            "frames_sampled": self.frames_sampled,
        }
