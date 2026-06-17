"""외부 COCO 데이터셋(Roboflow 등) → 우리 단일-클래스 gecko 변환 순수 로직.

Roboflow Universe 등 '라벨 딸린' 외부 데이터셋을 인입할 때, 그들의 (다중)클래스에서
gecko 만 골라 우리 category(id=1) 로 remap 하고 gecko 박스가 없는 이미지는 버린다.
파일/네트워크 I/O 는 scripts/import_roboflow_coco.py 가, 변환 규칙은 여기(단위 테스트
대상)가 담당한다.

설계서 §4.2: 외부 데이터는 train/val 만 — test 는 운영 프레임만(누수 방지).
"""

from __future__ import annotations

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
