"""mp4 → clip_prelabels JSON. Gate v0 오케스트레이션 + CLI.

    uv run python -m gecko_vision_gate.prelabel --input clip.mp4 --output out.json

흐름: 프레임 균등 샘플 → 각 프레임 RF-DETR 추론 → target(gecko) 최고 conf 프레임을
best_frame 으로 → JSON 계약 출력.

--checkpoint 없이 실행하면 COCO pretrained 라 gecko 클래스가 없어 gecko_visible 은 항상
false (파이프라인 sanity check 용). --checkpoint runs/.../checkpoint_best_total.pth 를 주면
fine-tune 된 gecko detector 로 로드되어 실제 gecko_visible/bbox 를 낸다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import GeckoDetector
from .frame_sampling import sample_frames
from .schema import DetectedObject, PrelabelResult

TARGET_CLASS = "gecko"
MODEL_VERSION = "v0-coco"  # fine-tune 전 COCO pretrained 단계임을 명시


def prelabel_clip(
    video_path: str | Path,
    *,
    num_frames: int = 12,
    threshold: float = 0.5,
    model_size: str = "nano",
    checkpoint: str | None = None,
    clip_id: str | None = None,
) -> PrelabelResult:
    frames = sample_frames(video_path, num_frames)
    detector = GeckoDetector(model_size=model_size, threshold=threshold, checkpoint=checkpoint)

    objects: list[DetectedObject] = []
    best: tuple[float, float, list[int]] | None = None  # (conf, ts, bbox) for TARGET

    for ts, frame in frames:
        for d in detector.detect(frame):
            objects.append(DetectedObject(d.class_name, round(d.confidence, 4), d.xywh, ts))
            if d.class_name == TARGET_CLASS and (best is None or d.confidence > best[0]):
                best = (d.confidence, ts, d.xywh)

    model_name = f"rf-detr-{model_size}"
    model_version = MODEL_VERSION if checkpoint is None else f"v1-gecko ({Path(checkpoint).stem})"
    common = dict(
        detected_objects=tuple(objects),
        frames_sampled=len(frames),
        model_name=model_name,
        model_version=model_version,
        clip_id=clip_id,
    )

    if best is not None:
        return PrelabelResult(
            gecko_visible=True,
            visibility_confidence=round(best[0], 4),
            best_frame_ts=best[1],
            gecko_bbox=best[2],
            skip_recommendation=False,
            skip_reason=None,
            **common,
        )
    return PrelabelResult(
        gecko_visible=False,
        visibility_confidence=0.0,
        best_frame_ts=None,
        gecko_bbox=None,
        skip_recommendation=True,
        skip_reason="no_gecko_detected",
        **common,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="gecko-vision-gate prelabel (v0)")
    p.add_argument("--input", required=True, help="입력 mp4 경로")
    p.add_argument("--output", help="출력 JSON 경로 (생략 시 stdout)")
    p.add_argument("--frames", type=int, default=12, help="샘플링할 프레임 수")
    p.add_argument("--threshold", type=float, default=0.5, help="검출 confidence 임계값")
    p.add_argument("--model-size", default="nano", choices=list(("nano", "small", "medium")))
    p.add_argument("--checkpoint", default=None, help="fine-tune .pth (주면 gecko 탐지, 없으면 COCO pretrained)")
    p.add_argument("--clip-id", default=None)
    args = p.parse_args(argv)

    result = prelabel_clip(
        args.input,
        num_frames=args.frames,
        threshold=args.threshold,
        model_size=args.model_size,
        checkpoint=args.checkpoint,
        clip_id=args.clip_id,
    )
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"✅ wrote {out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
