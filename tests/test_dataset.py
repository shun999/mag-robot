from collections import Counter

import numpy as np
import pytest

from dino_ingot.config import Config
from dino_ingot.dataset import load_records, save_sample, split_items, validate_records
from dino_ingot.vision import Region


def records(n=5):
    return [
        {"sample_id": f"{label}-{item}-{photo}", "item_id": f"{label}-{item}", "label": label}
        for label in ("normal", "anomaly")
        for item in range(n)
        for photo in range(3)
    ]


def test_group_split_disjoint_reproducible_stratified():
    rows = records(7)
    split = split_items(rows)
    assert split == split_items(rows)
    assert split != split_items(rows, seed=1)
    ids = {name: {rows[i]["item_id"] for i in indices} for name, indices in split.items()}
    assert not (
        ids["train"] & ids["test"]
        or ids["train"] & ids["validation"]
        or ids["test"] & ids["validation"]
    )
    assert sorted(sum(split.values(), [])) == list(range(len(rows)))
    for indices in split.values():
        assert set(Counter(rows[i]["label"] for i in indices)) == {"normal", "anomaly"}


def test_minimum_groups_and_conflicting_labels():
    with pytest.raises(ValueError, match="最低5個体"):
        split_items(records(4))
    rows = records()
    rows[0]["label"] = "anomaly"
    with pytest.raises(ValueError, match="異なるラベル"):
        validate_records(rows)


def test_capture_record_rejects_conflicts_and_invalid_region(tmp_path):
    config = Config()
    image = np.zeros((50, 50, 3), np.uint8)
    region = Region("OK", bbox=(0, 0, 50, 50), crop=image)
    target = save_sample(tmp_path, image, region, config, "item-01", "normal")
    assert (target / "raw.png").exists() and (target / "crop.png").exists()
    saved = load_records(tmp_path)
    assert saved[0]["item_id"] == "item-01"
    with pytest.raises(ValueError, match="異なるラベル"):
        save_sample(tmp_path, image, region, config, "item-01", "anomaly")
    config.extraction.threshold += 1
    with pytest.raises(ValueError, match="異なる背景"):
        save_sample(tmp_path, image, region, config, "item-02", "normal")
    with pytest.raises(ValueError, match="領域"):
        save_sample(tmp_path, image, Region("INVALID"), config, "item-03", "normal")
    assert len(load_records(tmp_path)) == 1
