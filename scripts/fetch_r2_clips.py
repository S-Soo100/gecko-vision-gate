"""Cloudflare R2 → 로컬로 펫캠 클립 다운로드 (멱등 — same-size skip).

운영 펫캠은 R2 버킷에 즉시 업로드된다(파일명=UTC, mp4 1개당 키프레임 jpg 1개).
gecko-vision-gate 엔 R2 설정이 없어 **자격증명은 petcam-lab(.env) 것을 재사용**한다.

    uv run python scripts/fetch_r2_clips.py \
        --prefix terra-clips/clips/p4cam-79b5d844/ \
        --dest ~/petcam-lab/storage/p4cam-79b5d844 [--ext .mp4] [--dry-run]

다운로드 후: extract_operational_frames.py --source-dir <dest>.
deps: boto3 (gecko venv 엔 없을 수 있음 → petcam-lab venv 로 실행하거나 `uv run --with boto3`).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_env(env_path: Path) -> dict:
    env = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R2 → 로컬 클립 다운로드(멱등)")
    ap.add_argument("--prefix", required=True, help="버킷 내 key prefix")
    ap.add_argument("--dest", required=True, help="로컬 저장 폴더")
    ap.add_argument("--bucket", default=None, help="버킷명(기본 .env R2_BUCKET)")
    ap.add_argument("--env", default="~/petcam-lab/.env", help="R2_* 자격증명 .env")
    ap.add_argument("--ext", default=".mp4", help="받을 확장자(콤마구분, 빈값=전부)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    import boto3
    from botocore.config import Config

    env = load_env(Path(args.env).expanduser())
    bucket = args.bucket or env["R2_BUCKET"]
    exts = tuple(e.strip() for e in args.ext.split(",")) if args.ext else ()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    def client():
        return boto3.client("s3", endpoint_url=env["R2_ENDPOINT"],
                            aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                            aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                            region_name="auto",
                            config=Config(signature_version="s3v4", max_pool_connections=args.workers * 2))

    s3 = client()
    objs = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=args.prefix):
        for o in page.get("Contents", []):
            if not exts or o["Key"].endswith(exts):
                objs.append((o["Key"], o["Size"]))
    print(f"대상 {len(objs)}개 (bucket={bucket}, prefix={args.prefix}, ext={args.ext or '전부'}) → {dest}")
    if args.dry_run:
        print("[dry-run] 다운로드 안 함")
        return 0

    def fetch(item):
        key, size = item
        dst = dest / Path(key).name
        if dst.exists() and dst.stat().st_size == size:
            return ("skip", size)
        s3.download_file(bucket, key, str(dst))
        return ("get", size)

    done = got = skipped = total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed(ex.submit(fetch, o) for o in objs):
            status, size = f.result()
            done += 1
            total += size
            got += status == "get"
            skipped += status == "skip"
            if done % 25 == 0 or done == len(objs):
                print(f"  {done}/{len(objs)} (받음 {got}·skip {skipped}·{total/1e9:.2f}GB)", flush=True)
    print(f"완료: 받음 {got} · skip {skipped} · 총 {total/1e9:.2f}GB → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
