"""GME 영구/14일 debug artifact의 결정론적 직렬화."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass

from .gme_contracts import GMEAnalysis

ARTIFACT_SCHEMA_VERSION = "gme-artifact-v1"


@dataclass(frozen=True, slots=True)
class SerializedArtifacts:
    permanent_gzip: bytes
    debug_gzip: bytes
    permanent_sha256: str
    debug_sha256: str


def _canonical_gzip(payload: dict) -> bytes:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def serialize_artifacts(analysis: GMEAnalysis) -> SerializedArtifacts:
    base = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_identity": asdict(analysis.artifact_identity),
        "duration_sec": analysis.duration_sec,
        "intervals": [asdict(v) for v in analysis.intervals],
        "track_points": [asdict(v) for v in analysis.track_points],
        "tracking_quality": asdict(analysis.tracking_quality),
    }
    permanent = _canonical_gzip(base)
    debug = _canonical_gzip({**base, "frame_debug": list(analysis.frame_debug)})
    return SerializedArtifacts(
        permanent, debug, hashlib.sha256(permanent).hexdigest(), hashlib.sha256(debug).hexdigest()
    )
