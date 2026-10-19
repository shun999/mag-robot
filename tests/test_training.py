import json

import numpy as np
import pytest

from dino_ingot.config import BACKBONE, DINOV3_COMMIT, FEATURE_SPEC, Config, sha256_file
from dino_ingot.dataset import save_sample
from dino_ingot.training import load_bundle, metrics, train, verify_extractor
from dino_ingot.vision import extract_region, preprocess, save_image


class FakeExtractor:
    """Use the production preprocessing; replace only the unavailable GPU backbone."""

    def __init__(self, size=32):
        self.size = size

    def identity(self):
        return {
            "backbone": BACKBONE,
            "code_commit": DINOV3_COMMIT,
            "feature_spec": FEATURE_SPEC,
            "input_size": self.size,
            "weights_sha256": "fake-test-weights",
        }

    def extract(self, crop):
        image = preprocess(crop, self.size)
        return np.concatenate([image.mean((1, 2)), image.max((1, 2))])


def make_dataset(root):
    config = Config()
    config.camera.width, config.camera.height = 120, 80
    config.extraction.min_area = 100
    config.features.input_size = 32
    background = np.zeros((80, 120, 3), np.uint8)
    config.background = str(root / "reference.png")
    save_image(config.background, background)
    config.background_sha256 = sha256_file(config.background)
    for label, color in (("normal", (80, 150, 210)), ("anomaly", (210, 80, 80))):
        for item in range(5):
            for photo in range(2):
                frame = background.copy()
                frame[20:60, 25:95] = np.array(color, np.uint8) + item + photo
                # Keep files unique even when color sums coincide.
                frame[25, 35] = (item + 50, photo + 50, 90)
                region = extract_region(frame, background, config.extraction)
                save_sample(root / "data", frame, region, config, f"{label}-{item}", label)
    return config


def test_full_cpu_training_bundle_roundtrip_and_inference_parity(tmp_path):
    config = make_dataset(tmp_path)
    extractor = FakeExtractor()
    target = tmp_path / "model"
    report = train(tmp_path / "data", target, config, extractor)
    assert report["test"]["anomaly_recall"] == 1.0
    assert report["counts"]["test"]["normal"] == {"images": 2, "items": 1}
    loaded, background, model, metadata = load_bundle(target)
    verify_extractor(metadata, extractor)
    assert loaded.identity() == config.identity()
    frame = background.copy()
    frame[20:60, 25:95] = (210, 80, 80)
    region = extract_region(frame, background, loaded.extraction)
    features = extractor.extract(region.crop)[None]
    assert model.predict(features)[0] == 1
    assert model.decision_function(features)[0] > 0
    assert metadata["feature_dimension"] == features.shape[1]
    assignments = json.loads((target / "split.json").read_text())
    assert len(assignments["train"]) == 12
    with pytest.raises(ValueError, match="既に存在"):
        train(tmp_path / "data", target, config, extractor)
    with pytest.raises(ValueError, match="一致"):
        verify_extractor(metadata, FakeExtractor(64))
    (target / "background.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="一致"):
        load_bundle(target)


def test_reject_modified_original(tmp_path):
    config = make_dataset(tmp_path)
    path = next((tmp_path / "data").glob("samples/*/raw.png"))
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="原画像"):
        train(tmp_path / "data", tmp_path / "model", config, FakeExtractor())


def test_metrics_confusion_and_false_positive_rate():
    result = metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["normal_false_positive_rate"] == 0.5
    assert result["anomaly_recall"] == 0.5
