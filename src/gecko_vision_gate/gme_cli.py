"""로컬 GME 단일 영상 smoke CLI. 경로·credential은 출력하지 않는다."""

from __future__ import annotations

import argparse

from .gme_detector import build_detector
from .gme_engine import analyze_clip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gecko Motion Engine shadow analyzer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-size", choices=("nano", "small", "medium"), default="nano")
    args = parser.parse_args(argv)
    detector = build_detector(checkpoint=args.checkpoint, threshold=args.threshold, model_size=args.model_size)
    result = analyze_clip(args.input, detector=detector)
    print(
        f"status={result.status} decoded={result.decoded_frame_count} analyzed={result.analyzed_frame_count} "
        f"candidate_moving_sec={result.candidate_moving_sec_any_gecko:.3f} "
        f"unknown_sec={result.unknown_sec:.3f} max_geckos={result.max_simultaneous_geckos}"
    )
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
