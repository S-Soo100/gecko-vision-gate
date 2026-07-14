"""provenance — checkpoint SHA-256 + sampler/schema version + GateProvenance 계약.

지시문 §205: checkpoint 의 실제 SHA-256 을 실행 환경에서 기록. threshold 는 코드 상수로
숨기지 말고 provenance 로 남긴다(§206).
"""

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from gecko_vision_gate.provenance import (
    SAMPLER_VERSION,
    SCHEMA_VERSION,
    GateProvenance,
    checkpoint_sha256,
)


def test_checkpoint_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "ckpt.pth"
    p.write_bytes(b"hello gecko checkpoint")
    assert checkpoint_sha256(p) == hashlib.sha256(b"hello gecko checkpoint").hexdigest()


def test_checkpoint_sha256_is_cached_by_path(tmp_path):
    # lru_cache: 같은 경로 재호출은 파일 재읽기 없이 첫 값 반환(운영 체크포인트는 불변)
    p = tmp_path / "ckpt.pth"
    p.write_bytes(b"original")
    first = checkpoint_sha256(p)
    p.write_bytes(b"tampered-after-cache")
    assert checkpoint_sha256(p) == first


def test_versions_are_nonempty_strings():
    assert isinstance(SAMPLER_VERSION, str) and SAMPLER_VERSION
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION


def test_gate_provenance_to_dict_has_all_provenance_fields():
    gp = GateProvenance(
        model_name="rf-detr-nano",
        model_version="gecko_v2 (checkpoint_best_ema)",
        checkpoint_sha256="deadbeef",
        threshold=0.25,
        sampler_version=SAMPLER_VERSION,
        schema_version=SCHEMA_VERSION,
        frames_sampled=12,
    )
    d = gp.to_dict()
    assert d == {
        "model_name": "rf-detr-nano",
        "model_version": "gecko_v2 (checkpoint_best_ema)",
        "checkpoint_sha256": "deadbeef",
        "threshold": 0.25,
        "sampler_version": SAMPLER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "frames_sampled": 12,
    }


def test_gate_provenance_is_frozen():
    gp = GateProvenance("m", "v", "sha", 0.25, SAMPLER_VERSION, SCHEMA_VERSION, 12)
    with pytest.raises(FrozenInstanceError):
        gp.threshold = 0.5  # type: ignore[misc]
