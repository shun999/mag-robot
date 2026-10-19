"""Immutable per-photo records and leakage-free physical-item splits."""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import Config, sha256_file, write_json
from .vision import Region, save_image

LABELS = {"normal": 0, "anomaly": 1}


def validate_item_id(item_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", item_id):
        raise ValueError("個体IDは英数字から始まる英数字・_・-・. の80文字以内にしてください")


def load_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(Path(root).glob("samples/*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_directory"] = path.parent
        records.append(record)
    validate_records(records)
    return records


def validate_records(records: list[dict]) -> None:
    known = {}
    seen = set()
    for record in records:
        validate_item_id(record["item_id"])
        if record["label"] not in LABELS:
            raise ValueError("normal / anomaly 以外のラベルがあります")
        if record["sample_id"] in seen:
            raise ValueError("サンプルIDが重複しています")
        seen.add(record["sample_id"])
        item_id, label = record["item_id"], record["label"]
        if item_id in known and known[item_id] != label:
            raise ValueError(f"同じ個体IDに異なるラベルがあります: {item_id}")
        known[item_id] = label


def save_sample(
    root: Path, frame, region: Region, config: Config, item_id: str, label: str
) -> Path:
    validate_item_id(item_id)
    if label not in LABELS:
        raise ValueError("ラベルは normal または anomaly です")
    if region.status != "OK" or region.crop is None or region.bbox is None:
        raise ValueError("判定可能な領域がないため撮影を保存できません")
    records = load_records(root)
    for record in records:
        if record["item_id"] == item_id and record["label"] != label:
            raise ValueError(f"同じ個体IDに異なるラベルを付けられません: {item_id}")
        if record["config_id"] != config.identity():
            raise ValueError("異なる背景・設定の撮影は別のデータディレクトリに保存してください")
    sample_id = uuid4().hex
    samples = Path(root) / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    temporary = samples / f".pending-{sample_id}"
    temporary.mkdir()
    try:
        save_image(temporary / "raw.png", frame)
        save_image(temporary / "crop.png", region.crop)
        write_json(
            temporary / "record.json",
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "item_id": item_id,
                "label": label,
                "bbox": list(region.bbox),
                "config_id": config.identity(),
                "config": config.to_dict(),
                "raw_sha256": sha256_file(temporary / "raw.png"),
            },
        )
        target = samples / sample_id
        temporary.rename(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def split_items(records: list[dict], seed: int = 42) -> dict[str, list[int]]:
    validate_records(records)
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": []}
    for label in LABELS:
        items = sorted({r["item_id"] for r in records if r["label"] == label})
        if len(items) < 5:
            raise ValueError(f"{label} は最低5個体必要です（現在{len(items)}個体）")
        rng.shuffle(items)
        n_holdout = max(1, int(len(items) * 0.2))
        assignment = {
            **dict.fromkeys(items[:n_holdout], "test"),
            **dict.fromkeys(items[n_holdout : 2 * n_holdout], "validation"),
            **dict.fromkeys(items[2 * n_holdout :], "train"),
        }
        for index, record in enumerate(records):
            if record["label"] == label:
                split[assignment[record["item_id"]]].append(index)
    return split
