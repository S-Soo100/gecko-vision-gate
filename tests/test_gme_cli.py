from __future__ import annotations

from gecko_vision_gate import gme_cli
from gecko_vision_gate.gme_contracts import ArtifactIdentity, GMEAnalysis, StateInterval, TrackingQuality


def test_cli_prints_redacted_summary_without_source_path(monkeypatch, capsys):
    analysis = GMEAnalysis.minimal(
        duration_sec=1.0,
        intervals=(StateInterval(0.0, 1.0, "unknown", ()),),
        tracking_quality=TrackingQuality.empty(),
        artifact_identity=ArtifactIdentity.test_identity(),
    )
    monkeypatch.setattr(gme_cli, "build_detector", lambda **_: object())
    monkeypatch.setattr(gme_cli, "analyze_clip", lambda *_a, **_k: analysis)
    code = gme_cli.main(["--input", "/private/secret/clip.mp4", "--checkpoint", "/private/model.pth"])
    out = capsys.readouterr().out
    assert code == 0
    assert "status=ok" in out
    assert "secret" not in out
    assert "model.pth" not in out
