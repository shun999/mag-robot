import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pytest

from dino_ingot.config import DINOV3_COMMIT
from dino_ingot.features import DinoExtractor, verify_repository
from dino_ingot.vision import preprocess


def test_dino_adapter_uses_local_frozen_model_and_normalized_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr("dino_ingot.features.verify_repository", lambda _: DINOV3_COMMIT)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fake-local-weights")
    torch = MagicMock()
    torch.inference_mode.side_effect = nullcontext
    model = torch.hub.load.return_value
    cls, patches = MagicMock(), MagicMock()
    cls.float.return_value.cpu.return_value.numpy.return_value = np.ones((1, 384))
    patches.float.return_value.cpu.return_value.numpy.return_value = np.full((1, 4, 384), 2)
    model.forward_features.return_value = {
        "x_norm_clstoken": cls,
        "x_norm_patchtokens": patches,
    }
    monkeypatch.setitem(sys.modules, "torch", torch)
    extractor = DinoExtractor(tmp_path, weights, 32, "cpu")
    crop = np.full((40, 60, 3), (20, 70, 200), np.uint8)
    vector = extractor.extract(crop)
    assert vector.shape == (1152,)
    np.testing.assert_array_equal(vector[:384], 1)
    np.testing.assert_array_equal(vector[384:], 2)
    np.testing.assert_array_equal(torch.from_numpy.call_args.args[0], preprocess(crop, 32))
    assert torch.hub.load.call_args.kwargs == {"source": "local", "weights": str(weights)}
    model.eval.return_value.requires_grad_.assert_called_once_with(False)
    assert extractor.identity()["code_commit"] == DINOV3_COMMIT


@pytest.mark.parametrize("commit,dirty", [("other-commit", ""), (DINOV3_COMMIT, " M hubconf.py")])
def test_reject_wrong_or_modified_dino_repository(tmp_path, monkeypatch, commit, dirty):
    (tmp_path / "hubconf.py").write_text("")
    results = iter([commit, dirty])
    monkeypatch.setattr(
        "dino_ingot.features.subprocess.check_output", lambda *a, **kw: next(results)
    )
    with pytest.raises(ValueError, match="指定コミット"):
        verify_repository(tmp_path)


def test_cuda_failure_does_not_silently_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr("dino_ingot.features.verify_repository", lambda _: DINOV3_COMMIT)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"test")
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(ValueError, match="CUDA"):
        DinoExtractor(tmp_path, weights, 32)
    torch.hub.load.assert_not_called()
