"""Shared region extraction and deterministic image preprocessing."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config, ExtractionConfig, sha256_file

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Region:
    status: str
    reason: str = ""
    bbox: tuple[int, int, int, int] | None = None
    crop: np.ndarray | None = None
    mask: np.ndarray | None = None


def read_image(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません: {path}")
    return image


def save_image(path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"画像を保存できません: {path}")


def load_background(config: Config) -> np.ndarray:
    if not config.background or not config.background_sha256:
        raise ValueError("先に calibrate で背景を保存してください")
    if sha256_file(config.background) != config.background_sha256:
        raise ValueError("背景画像のハッシュが一致しません。再調整してください")
    image = read_image(config.background)
    if image.shape[:2] != (config.camera.height, config.camera.width):
        raise ValueError("背景画像と設定の解像度が一致しません")
    return image


def extract_region(frame: np.ndarray, background: np.ndarray, cfg: ExtractionConfig) -> Region:
    if frame.shape != background.shape or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("入力フレームと背景画像の形状が一致しません")
    height, width = frame.shape[:2]
    rx, ry, rw, rh = cfg.roi or (0, 0, width, height)
    if rx < 0 or ry < 0 or rw <= 0 or rh <= 0 or rx + rw > width or ry + rh > height:
        raise ValueError("検査範囲が画像の外にあります")
    source = frame[ry : ry + rh, rx : rx + rw]
    reference = background[ry : ry + rh, rx : rx + rw]
    difference = cv2.absdiff(source, reference).max(axis=2)
    mask = (difference > cfg.threshold).astype(np.uint8) * 255
    kernel = np.ones((cfg.kernel_size, cfg.kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= cfg.min_area]
    if not contours:
        return Region("NO OBJECT", mask=cleaned)
    if len(contours) != 1:
        return Region("INVALID", "multiple objects", mask=cleaned)
    # Include original connected components in the edge check, so opening cannot
    # hide a thin extension that connects the ingot to the inspection boundary.
    component = np.zeros_like(mask)
    cv2.drawContours(component, contours, -1, 255, cv2.FILLED)
    _, labels = cv2.connectedComponents(mask)
    object_labels = np.unique(labels[component > 0])
    object_labels = object_labels[object_labels != 0]
    border_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    if np.intersect1d(object_labels, border_labels).size:
        return Region("INVALID", "object touches inspection boundary", mask=cleaned)
    x, y, w, h = cv2.boundingRect(contours[0])
    if x <= 1 or y <= 1 or x + w >= rw - 1 or y + h >= rh - 1:
        return Region("INVALID", "object touches inspection boundary", mask=cleaned)
    mx, my = round(w * cfg.margin), round(h * cfg.margin)
    left, top = max(0, x - mx) + rx, max(0, y - my) + ry
    right, bottom = min(rw, x + w + mx) + rx, min(rh, y + h + my) + ry
    return Region(
        "OK",
        bbox=(left, top, right - left, bottom - top),
        crop=frame[top:bottom, left:right].copy(),
        mask=cleaned,
    )


def preprocess(crop: np.ndarray, input_size: int) -> np.ndarray:
    """BGR uint8 -> square letterbox -> resize -> RGB float32 normalized CHW."""
    if input_size <= 0 or input_size % 16:
        raise ValueError("input_size は16の正の倍数です")
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        raise ValueError("画像は空でない uint8 BGR 画像である必要があります")
    h, w = crop.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top : top + h, left : left + w] = crop
    resized = cv2.resize(
        padded,
        (input_size, input_size),
        interpolation=cv2.INTER_AREA if side > input_size else cv2.INTER_CUBIC,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))


def pool_features(cls: np.ndarray, patches: np.ndarray) -> np.ndarray:
    if cls.ndim != 2 or patches.ndim != 3 or patches.shape[0] != cls.shape[0]:
        raise ValueError("DINOv3 出力の形状が不正です")
    if patches.shape[1] == 0 or patches.shape[2] != cls.shape[1]:
        raise ValueError("DINOv3 パッチ特徴の形状が不正です")
    result = np.concatenate([cls, patches.mean(axis=1), patches.max(axis=1)], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("特徴量に NaN/Inf が含まれています")
    return result.astype(np.float32)
