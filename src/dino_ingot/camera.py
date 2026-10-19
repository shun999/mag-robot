"""Camera lifetime, frame validation, and OpenCV desktop displays."""

import os
from contextlib import contextmanager

import cv2
import numpy as np

from .config import CameraConfig
from .vision import Region


def require_gui() -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("画面が利用できません。Jetsonのデスクトップ端末から実行してください")
    gui_line = next(
        (
            line
            for line in cv2.getBuildInformation().splitlines()
            if line.strip().startswith("GUI:")
        ),
        "",
    )
    if not gui_line or "NONE" in gui_line or gui_line.strip().endswith("NO"):
        raise ValueError("GUI対応OpenCVが必要です。opencv-python-headless は使用できません")


def read_frame(capture, config: CameraConfig):
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("カメラの読取に失敗しました。接続を確認してください")
    if frame.shape != (config.height, config.width, 3):
        raise ValueError(
            f"撮影解像度が設定と異なります: 実際={frame.shape[:2]}, "
            f"設定={(config.height, config.width)}"
        )
    return frame


@contextmanager
def open_camera(config: CameraConfig):
    capture = cv2.VideoCapture(config.source, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"USBカメラを開けません: {config.source}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):
            read_frame(capture, config)
        yield capture
    finally:
        capture.release()


def show_inspection(window: str, frame, region: Region, roi, text: str) -> None:
    canvas = frame.copy()
    if roi:
        x, y, w, h = roi
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 1)
    color = (0, 210, 0) if text.startswith("NORMAL") else (0, 100, 255)
    if region.bbox:
        x, y, w, h = region.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 64), (0, 0, 0), cv2.FILLED)
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(canvas, region.reason, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Bound display size; detection and stored images always retain original resolution.
    scale = min(1.0, 1280 / canvas.shape[1], 800 / canvas.shape[0])
    cv2.imshow(window, cv2.resize(canvas, None, fx=scale, fy=scale))
    preview = np.zeros((256, 256, 3), dtype=np.uint8)
    if region.crop is not None:
        h, w = region.crop.shape[:2]
        scale = 256 / max(h, w)
        resized = cv2.resize(region.crop, None, fx=scale, fy=scale)
        preview[: resized.shape[0], : resized.shape[1]] = resized
    cv2.imshow("Ingot crop", preview)


def clear_failure(window: str) -> None:
    """Replace both windows so a disconnected camera never leaves an old verdict."""
    blank = np.zeros((240, 800, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        "CAMERA / PIPELINE ERROR - stopped",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    cv2.imshow(window, blank)
    cv2.imshow("Ingot crop", np.zeros((256, 256, 3), dtype=np.uint8))
    cv2.waitKey(1000)


def quit_requested(window: str, key: int) -> bool:
    return key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
