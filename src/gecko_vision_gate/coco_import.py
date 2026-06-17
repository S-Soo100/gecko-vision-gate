"""외부 COCO 데이터셋(Roboflow 등) → 우리 단일-클래스 gecko 변환 순수 로직.

Roboflow Universe 등 '라벨 딸린' 외부 데이터셋을 인입할 때, 그들의 (다중)클래스에서
gecko 만 골라 우리 category(id=1) 로 remap 하고 gecko 박스가 없는 이미지는 버린다.
파일/네트워크 I/O 는 scripts/import_roboflow_coco.py 가, 변환 규칙은 여기(단위 테스트
대상)가 담당한다.

설계서 §4.2: 외부 데이터는 train/val 만 — test 는 운영 프레임만(누수 방지).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

MANIFEST_COLUMNS = ["filename", "source", "clip_id", "split", "labeled", "domain"]
GECKO_CATEGORY = {"id": 1, "name": "gecko"}  # 우리 단일 클래스


def gecko_category_ids(categories: list[dict], patterns: tuple[str, ...] = ("gecko",)) -> set[int]:
    """COCO categories 중 이름에 gecko 패턴이 든 category id 집합 (대소문자 무시).

    Roboflow 가 클래스명에 프로젝트 태그를 붙이거나(예: 'gecko-xyz') 종을 나눠도
    ('leopard gecko') 'gecko' 부분문자열로 다 잡아 우리 단일 gecko 로 흡수한다.
    """
    pats = tuple(p.lower() for p in patterns)
    return {c["id"] for c in categories if any(p in str(c["name"]).lower() for p in pats)}


@dataclass
class GeckoImage:
    file_name: str
    width: int
    height: int
    boxes: list[list[float]]  # COCO [x,y,w,h], gecko 박스만


def collect_gecko_images(coco: dict, gecko_ids: set[int]) -> list[GeckoImage]:
    """COCO dict → gecko 박스가 1개 이상인 이미지들.

    비-gecko 박스(towel 등)는 버리고, gecko 박스가 하나도 없는 이미지는 통째로 제외한다
    (우리는 gecko-present 만 train 에 넣음 — negative 는 운영프레임에서 따로 §4.1).
    """
    by_img: dict[int, list[list[float]]] = {}
    for a in coco.get("annotations", []):
        if a.get("category_id") in gecko_ids:
            by_img.setdefault(a["image_id"], []).append(a["bbox"])
    out: list[GeckoImage] = []
    for im in coco.get("images", []):
        boxes = by_img.get(im["id"])
        if not boxes:
            continue
        out.append(GeckoImage(im["file_name"], int(im.get("width", 0)), int(im.get("height", 0)), boxes))
    return out


def raw_rel_from_path(file_name: str, marker: str = "datasets/raw/") -> str | None:
    """Label Studio export 의 file_name → 우리 raw 상대경로(operational/<clip>/<frame>).

    LS 는 file_name 을 '../../.../datasets/raw/operational/<clip>/f000.jpg' 로 내보낸다.
    'datasets/raw/' 뒤를 잘라 manifest.filename 키와 맞춘다(없으면 None).
    """
    p = file_name.replace("\\", "/")
    i = p.find(marker)
    return p[i + len(marker):] if i != -1 else None


def assign_clip_splits(clip_ids, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                       seed: int = 42) -> dict[str, str]:
    """clip_id → train/val/test 배정. 같은 clip 은 한 split (누수 방지 §12).

    결정적 셔플(seed) 후 비율로 자른다. 운영 프레임 전용 — test 는 운영만 허용(§4.2).
    """
    clips = sorted(set(clip_ids))
    random.Random(seed).shuffle(clips)
    n = len(clips)
    n_tr, n_va = round(n * ratios[0]), round(n * ratios[1])
    out: dict[str, str] = {}
    for i, c in enumerate(clips):
        out[c] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


# ── RF-DETR(Roboflow) 학습 레이아웃 변환 ──
# RF-DETR 은 dataset_dir/{train,valid,test}/_annotations.coco.json + 이미지(같은 폴더)를
# 기대하고, category id 0=더미·1=실클래스인 Roboflow 관례를 따른다.
ROBOFLOW_CATEGORIES = [
    {"id": 0, "name": "gecko", "supercategory": "none"},
    {"id": 1, "name": "gecko", "supercategory": "gecko"},
]


def flatten_name(raw_rel: str) -> str:
    """raw 상대경로 → split 폴더 내 충돌없는 평면 파일명.

    operational/9af1ba2e/f000.jpg → operational__9af1ba2e__f000.jpg
    (clip 마다 같은 basename(f000…)이라 평면 폴더에선 충돌 → 경로를 파일명에 인코딩).
    """
    return raw_rel.replace("/", "__")


def to_rfdetr_coco(our_coco: dict) -> dict:
    """우리 coco(file_name=raw상대경로, category 1) → RF-DETR 레이아웃용 coco.

    file_name 평면화 + categories 를 Roboflow 관례로. annotations(category_id=1) 그대로.
    """
    images = [{**im, "file_name": flatten_name(im["file_name"])} for im in our_coco.get("images", [])]
    return {
        "images": images,
        "annotations": list(our_coco.get("annotations", [])),
        "categories": [dict(c) for c in ROBOFLOW_CATEGORIES],
    }


def subset_coco(coco: dict, limit: int) -> dict:
    """앞 limit 개 이미지 + 그 annotations 만 (smoke 용). limit<=0 이면 원본 그대로."""
    if limit <= 0:
        return coco
    imgs = coco.get("images", [])[:limit]
    keep = {im["id"] for im in imgs}
    anns = [a for a in coco.get("annotations", []) if a["image_id"] in keep]
    return {"images": imgs, "annotations": anns, "categories": coco.get("categories", [])}
