"""운영 펫캠 영상 → 학습용 프레임 → datasets/raw/operational/<clip_id>/.

클래스 균형으로 영상을 고르고(파일명 prefix), 영상당 균등 N프레임을 추출한다.
영상단위 폴더라 train/test split 누수를 막는다. gecko detector seed 라벨용
(MODEL_AND_TRAINING_PLAN §6 현실적 최소 루프: 영상 20~30, 장당 5~10).

    uv run python scripts/extract_operational_frames.py \
      --source-dir ~/petcam-lab/storage/dataset-203 \
      --per-class 5 --frames 6 [--dry-run]

frame_sampling.sample_frames(균등 N장)를 재사용. 원본 해상도 그대로 저장(no-upscale).
멱등 아님 — 같은 clip_id 폴더에 다시 쓰면 덮어씀. 추출 후 build_manifest.py 재실행.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2

from gecko_vision_gate.frame_sampling import sample_frames

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "raw" / "operational"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def class_of(path: Path) -> str:
    """파일명 prefix = 클래스. moving__drinking__abc.mp4 → moving."""
    return path.stem.split("__")[0]


def clip_id_of(path: Path) -> str:
    """파일명 마지막 토큰 = clip_id. moving__drinking__48b5582e → 48b5582e."""
    parts = path.stem.split("__")
    return parts[-1] if len(parts) >= 2 else path.stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="운영 영상 → 학습 프레임 추출")
    ap.add_argument("--source-dir", required=True, help="운영 영상 폴더 (예: dataset-203)")
    ap.add_argument("--per-class", type=int, default=5, help="클래스당 영상 수 (0=전체)")
    ap.add_argument("--frames", type=int, default=6, help="영상당 프레임 수")
    ap.add_argument("--seed", type=int, default=42, help="영상 샘플링 재현용 seed")
    ap.add_argument("--dry-run", action="store_true", help="선택만 출력, 추출 안 함")
    args = ap.parse_args(argv)

    src = Path(args.source_dir).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"source dir not found: {src}")

    videos = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    by_class: dict[str, list[Path]] = defaultdict(list)
    for v in videos:
        by_class[class_of(v)].append(v)

    rng = random.Random(args.seed)
    chosen: list[Path] = []
    for _cls, vs in sorted(by_class.items()):
        vs_sorted = sorted(vs)  # 정렬 후 셔플 → seed 재현성
        rng.shuffle(vs_sorted)
        chosen.extend(vs_sorted if args.per_class == 0 else vs_sorted[: args.per_class])

    print(f"영상 {len(videos)} → 선택 {len(chosen)} (클래스 {len(by_class)}, per_class={args.per_class}, frames={args.frames})")
    for cls, vs in sorted(by_class.items()):
        picked = len(vs) if args.per_class == 0 else min(args.per_class, len(vs))
        print(f"  {cls:16} {picked}/{len(vs)}")
    print(f"예상 프레임 ≈ {len(chosen) * args.frames}")

    if args.dry_run:
        print("[dry-run] 추출 안 함")
        return 0

    total = 0
    for v in chosen:
        outdir = OUT / clip_id_of(v)
        outdir.mkdir(parents=True, exist_ok=True)
        for i, (ts, frame) in enumerate(sample_frames(v, args.frames)):
            cv2.imwrite(str(outdir / f"f{i:03d}_t{ts:.1f}.jpg"), frame)
            total += 1
    print(f"✅ {len(chosen)} 영상 → {total} 프레임 → {OUT}")
    print("다음: uv run python scripts/build_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
