from argparse import Namespace
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pytest

from dino_ingot.camera import open_camera, read_frame
from dino_ingot.cli import main
from dino_ingot.config import CameraConfig


class Camera:
    def __init__(self, ok=True, shape=(720, 1280, 3)):
        self.ok, self.shape, self.released = ok, shape, False

    def read(self):
        return self.ok, np.zeros(self.shape, np.uint8) if self.ok else None

    def isOpened(self):
        return self.ok

    def set(self, *args):
        return True

    def release(self):
        self.released = True


def test_read_failure_and_resolution():
    with pytest.raises(RuntimeError, match="読取"):
        read_frame(Camera(False), CameraConfig())
    with pytest.raises(ValueError, match="解像度"):
        read_frame(Camera(shape=(480, 640, 3)), CameraConfig())


def test_camera_released_even_when_open_or_pipeline_fails(monkeypatch):
    camera = Camera(False)
    monkeypatch.setattr("dino_ingot.camera.cv2.VideoCapture", lambda *args: camera)
    with pytest.raises(RuntimeError):
        with open_camera(CameraConfig()):
            pass
    assert camera.released
    camera.ok = True
    camera.released = False
    with pytest.raises(ValueError):
        with open_camera(CameraConfig()):
            raise ValueError("pipeline")
    assert camera.released


def test_cli_missing_configuration_returns_actionable_error(capsys):
    assert main(["train", "--config", "/missing/config.json"]) == 1
    assert "エラー" in capsys.readouterr().err


@pytest.mark.parametrize("ur3_enabled", [False, True])
def test_inference_clears_verdict_on_no_object_and_camera_disconnect(monkeypatch, ur3_enabled):
    from dino_ingot.cli import infer
    from dino_ingot.config import Config

    config = Config()
    config.camera.width, config.camera.height = 120, 80
    config.extraction.min_area = 100
    background = np.zeros((80, 120, 3), np.uint8)
    valid = background.copy()
    valid[20:60, 25:95] = 200
    frames = iter([valid, valid, valid, background])
    model = MagicMock()
    model.predict.return_value = [1]
    model.decision_function.return_value = [0.7]
    metadata = {"feature_dimension": 6, "feature_identity": {"test": True}}
    monkeypatch.setattr(
        "dino_ingot.training.load_bundle", lambda _: (config, background, model, metadata)
    )
    extractor = MagicMock()
    extractor.identity.return_value = metadata["feature_identity"]
    extractor.extract.return_value = np.ones(6, np.float32)
    monkeypatch.setattr("dino_ingot.cli.make_extractor", lambda *a: extractor)
    monkeypatch.setattr("dino_ingot.cli.require_gui", lambda: None)
    monkeypatch.setattr("dino_ingot.cli.open_camera", lambda _: nullcontext(object()))

    def read(*args):
        try:
            return next(frames)
        except StopIteration:
            raise RuntimeError("camera disconnected")

    monkeypatch.setattr("dino_ingot.cli.read_frame", read)
    show, clear = MagicMock(), MagicMock()
    monkeypatch.setattr("dino_ingot.cli.show_inspection", show)
    monkeypatch.setattr("dino_ingot.cli.clear_failure", clear)
    monkeypatch.setattr("dino_ingot.cli.cv2.waitKey", lambda _: 0)
    monkeypatch.setattr("dino_ingot.cli.quit_requested", lambda *a: False)
    robot = None
    if ur3_enabled:
        from dino_ingot.ur3 import UR3Config, WristController

        robot = WristController(UR3Config(), dry_run=True)
        monkeypatch.setattr("dino_ingot.ur3.load_ur3_config", lambda _: robot.config)
        monkeypatch.setattr("dino_ingot.ur3.WristController", lambda *a, **kw: robot)

        def cleared_after_stop(*args, **kwargs):
            assert robot.stopped_confirmed
            assert "STOP CONFIRMED" in kwargs["detail"]

        clear.side_effect = cleared_after_stop
    with pytest.raises(RuntimeError, match="disconnected"):
        infer(Namespace(
            model="test", camera=None, ur3_config="ur3.json" if ur3_enabled else None,
            ur3_dry_run=ur3_enabled,
        ))
    assert show.call_args_list[0].args[-1].startswith("ANOMALY")
    assert show.call_args_list[3].args[-1].startswith("NO OBJECT")
    assert model.predict.call_count == 3
    if ur3_enabled:
        assert "cmd=-1" in show.call_args_list[2].kwargs["extra_text"]
        assert "cmd=+0" in show.call_args_list[3].kwargs["extra_text"]
        assert not robot._worker.is_alive()
    clear.assert_called_once()


def test_dry_run_requires_ur3_configuration(capsys):
    assert main(["infer", "--ur3-dry-run"]) == 1
    assert "--ur3-config" in capsys.readouterr().err
