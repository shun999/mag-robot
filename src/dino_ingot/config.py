"""Validated, portable configuration and reproducible calibration identity."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
BACKBONE = "dinov3_vits16"
FEATURE_SPEC = "cls-patch_mean-patch_max-rgb-imagenet-letterbox-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ExtractionConfig:
    roi: list[int] | None = None  # x, y, width, height in the original frame
    threshold: int = 35  # maximum absolute difference across BGR channels
    min_area: int = 2000
    kernel_size: int = 3
    margin: float = 0.05


@dataclass
class FeatureConfig:
    input_size: int = 256


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    background: str | None = None
    background_sha256: str | None = None

    def validate(self) -> None:
        c, e = self.camera, self.extraction
        if not isinstance(c.source, (int, str)) or isinstance(c.source, bool):
            raise ValueError("camera.source は番号または /dev/video... を指定してください")
        if isinstance(c.source, int) and c.source < 0:
            raise ValueError("カメラ番号は0以上です")
        if isinstance(c.source, str) and not c.source.startswith("/dev/"):
            raise ValueError("カメラパスは /dev/ から始めてください（映像ファイルは指定不可）")
        for name, value in {
            "width": c.width,
            "height": c.height,
            "fps": c.fps,
            "min_area": e.min_area,
            "kernel_size": e.kernel_size,
            "input_size": self.features.input_size,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} は正の整数にしてください")
        if type(e.threshold) is not int or not 0 <= e.threshold <= 255:
            raise ValueError("threshold は0〜255の整数です")
        if e.kernel_size % 2 != 1:
            raise ValueError("kernel_size は奇数です")
        if not math.isfinite(e.margin) or not 0 <= e.margin <= 0.5:
            raise ValueError("margin は0〜0.5です")
        if self.features.input_size % 16:
            raise ValueError("input_size は16の倍数です")
        if e.roi is not None:
            if len(e.roi) != 4 or any(type(v) is not int for v in e.roi):
                raise ValueError("roi は [x, y, width, height] の整数配列です")
            x, y, w, h = e.roi
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > c.width or y + h > c.height:
                raise ValueError("roi が撮影範囲外です")

    def to_dict(self) -> dict:
        return asdict(self)

    def identity(self) -> str:
        data = self.to_dict()
        # Relocating the background must not change calibration identity.
        data.pop("background")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        self.validate()
        write_json(path, self.to_dict())


def load_config(path: Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = Config(
        camera=CameraConfig(**raw.pop("camera", {})),
        extraction=ExtractionConfig(**raw.pop("extraction", {})),
        features=FeatureConfig(**raw.pop("features", {})),
        **raw,
    )
    config.validate()
    if config.background:
        config.background = str((Path(path).parent / config.background).resolve())
    return config
