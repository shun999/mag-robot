from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_ingot.config import Config, ExtractionConfig, load_config, sha256_file
from dino_ingot.vision import MEAN, STD, extract_region, load_background, pool_features, preprocess


def test_regions_and_invalid_states():
    background = np.zeros((160, 240, 3), np.uint8)
    frame = background.copy()
    cfg = ExtractionConfig(min_area=100)
    assert extract_region(frame, background, cfg).status == "NO OBJECT"
    frame[40:100, 60:180] = 150
    region = extract_region(frame, background, cfg)
    assert region.status == "OK"
    x, y, w, h = region.bbox
    assert x < 60 and y < 40 and x + w > 180 and y + h > 100
    np.testing.assert_array_equal(region.crop, frame[y : y + h, x : x + w])
    frame[110:140, 15:40] = 150
    assert extract_region(frame, background, cfg).reason == "multiple objects"
    frame = background.copy()
    frame[40:100, :180] = 150
    assert extract_region(frame, background, cfg).status == "INVALID"
    # A one-pixel connection must remain invalid even after morphology removes it.
    frame = background.copy()
    frame[40:100, 60:180] = 150
    frame[70, :60] = 150
    assert extract_region(frame, background, cfg).status == "INVALID"


def test_roi_and_camera_geometry():
    background = np.zeros((160, 240, 3), np.uint8)
    frame = background.copy()
    frame[50:90, 80:120] = 255
    cfg = ExtractionConfig(roi=[50, 30, 150, 100], min_area=100)
    region = extract_region(frame, background, cfg)
    assert region.status == "OK"
    assert region.bbox[0] < 80 and region.bbox[1] < 50
    frame[50:90, 50:120] = 255
    assert extract_region(frame, background, cfg).status == "INVALID"
    with pytest.raises(ValueError, match="形状"):
        extract_region(frame[:100], background, cfg)


def test_preprocessing_rgb_values_letterbox_and_size():
    bgr = np.zeros((32, 32, 3), np.uint8)
    bgr[:, :, 2] = 255
    tensor = preprocess(bgr, 32)
    assert tensor.shape == (3, 32, 32) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, 0, 0], (np.array([1, 0, 0]) - MEAN) / STD)
    wide = np.full((16, 32, 3), 255, np.uint8)
    padded = preprocess(wide, 32)
    np.testing.assert_allclose(padded[:, 0, 0], -MEAN / STD)
    np.testing.assert_allclose(padded[:, 12, 10], (1 - MEAN) / STD)
    assert padded.flags.c_contiguous
    for invalid in (0, 15, 257):
        with pytest.raises(ValueError):
            preprocess(bgr, invalid)


def test_pooling_preserves_cls_mean_and_max():
    cls = np.array([[1, 2]], np.float32)
    patches = np.array([[[3, 4], [5, 8]]], np.float32)
    np.testing.assert_array_equal(pool_features(cls, patches), [[1, 2, 4, 6, 5, 8]])
    with pytest.raises(ValueError):
        pool_features(cls, patches * np.nan)


def test_config_roundtrip_background_relocation_and_tampering(tmp_path):
    config = Config()
    background = tmp_path / "background.png"
    cv2.imwrite(str(background), np.zeros((720, 1280, 3), np.uint8))
    config.background = "background.png"
    config.background_sha256 = sha256_file(background)
    config.save(tmp_path / "config.json")
    loaded = load_config(tmp_path / "config.json")
    assert Path(loaded.background) == background
    assert config.identity() == loaded.identity()
    assert load_background(loaded).shape == (720, 1280, 3)
    background.write_bytes(b"modified")
    with pytest.raises(ValueError, match="ハッシュ"):
        load_background(loaded)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("features", "input_size", 255),
        ("camera", "width", -1),
        ("extraction", "kernel_size", 2),
        ("extraction", "threshold", 256),
        ("extraction", "roi", [0, 0, 5000, 10]),
    ],
)
def test_invalid_config(section, key, value):
    config = Config()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        config.validate()
