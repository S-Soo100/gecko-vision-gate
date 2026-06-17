"""coco_import 순수 로직 테스트 — 외부 COCO → 우리 gecko 단일클래스 변환 규칙."""

from collections import Counter

from gecko_vision_gate.coco_import import (
    assign_clip_splits,
    collect_gecko_images,
    flatten_name,
    gecko_category_ids,
    raw_rel_from_path,
    subset_coco,
    to_rfdetr_coco,
)


def test_gecko_category_ids_case_insensitive_and_ignores_others():
    cats = [
        {"id": 0, "name": "Gecko"},
        {"id": 1, "name": "towel"},
        {"id": 2, "name": "leopard gecko"},
        {"id": 3, "name": "objects-l6Lr"},
    ]
    assert gecko_category_ids(cats) == {0, 2}  # Gecko + leopard gecko, towel/objects 제외


def test_collect_drops_nongecko_boxes_and_geckoless_images():
    coco = {
        "categories": [{"id": 1, "name": "gecko"}, {"id": 2, "name": "towel"}],
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 80},
            {"id": 11, "file_name": "b.jpg", "width": 50, "height": 50},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4]},   # gecko
            {"id": 2, "image_id": 10, "category_id": 2, "bbox": [5, 5, 5, 5]},   # towel → 버림
            {"id": 3, "image_id": 11, "category_id": 2, "bbox": [0, 0, 1, 1]},   # b.jpg gecko 없음 → 제외
        ],
    }
    gids = gecko_category_ids(coco["categories"])
    imgs = collect_gecko_images(coco, gids)
    assert len(imgs) == 1
    assert imgs[0].file_name == "a.jpg"
    assert imgs[0].boxes == [[1, 2, 3, 4]]   # towel 박스 빠짐
    assert (imgs[0].width, imgs[0].height) == (100, 80)


def test_single_class_keeps_all_boxes():
    coco = {
        "categories": [{"id": 0, "name": "gecko"}],
        "images": [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 5, 5]},
            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [5, 5, 5, 5]},
        ],
    }
    imgs = collect_gecko_images(coco, gecko_category_ids(coco["categories"]))
    assert len(imgs) == 1 and len(imgs[0].boxes) == 2


def test_raw_rel_from_path_extracts_operational_path():
    p = "../../../../Users/baek/x/gecko-vision-gate/datasets/raw/operational/9af1ba2e/f000_t0.0.jpg"
    assert raw_rel_from_path(p) == "operational/9af1ba2e/f000_t0.0.jpg"
    assert raw_rel_from_path("/no/marker/here.jpg") is None


def test_assign_clip_splits_no_leak_ratio_deterministic():
    clips = [f"c{i}" for i in range(32)]
    sp = assign_clip_splits(clips)
    c = Counter(sp.values())
    assert (c["train"], c["val"], c["test"]) == (22, 5, 5)  # 70/15/15 of 32
    assert len(sp) == 32  # 모든 clip 정확히 한 split
    assert assign_clip_splits(clips) == sp  # 결정적
    assert assign_clip_splits([*clips, *clips]) == sp  # 중복 입력도 동일


def test_flatten_name_encodes_path():
    assert flatten_name("operational/9af1ba2e/f000_t0.0.jpg") == "operational__9af1ba2e__f000_t0.0.jpg"


def test_to_rfdetr_coco_flattens_and_roboflow_categories():
    our = {
        "images": [{"id": 1, "file_name": "operational/c/f.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        "categories": [{"id": 1, "name": "gecko"}],
    }
    rf = to_rfdetr_coco(our)
    assert rf["images"][0]["file_name"] == "operational__c__f.jpg"
    assert rf["categories"][0]["id"] == 0 and rf["categories"][1]["name"] == "gecko"  # Roboflow 관례
    assert rf["annotations"] == our["annotations"]  # 박스 그대로


def test_subset_coco_keeps_first_n_and_their_anns():
    coco = {
        "images": [{"id": 1, "file_name": "a"}, {"id": 2, "file_name": "b"}, {"id": 3, "file_name": "c"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1]}, {"id": 2, "image_id": 3, "bbox": [0, 0, 1, 1]}],
        "categories": [{"id": 1, "name": "gecko"}],
    }
    s = subset_coco(coco, 2)
    assert [im["id"] for im in s["images"]] == [1, 2]
    assert len(s["annotations"]) == 1 and s["annotations"][0]["image_id"] == 1  # img3 제외
    assert subset_coco(coco, 0) == coco  # limit<=0 → 원본
