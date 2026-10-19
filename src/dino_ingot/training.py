"""Train/validation model selection, held-out test evaluation and model bundles."""

import json
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    BACKBONE,
    DINOV3_COMMIT,
    FEATURE_SPEC,
    Config,
    load_config,
    sha256_file,
    write_json,
)
from .dataset import LABELS, load_records, split_items
from .vision import extract_region, load_background, read_image

BUNDLE_VERSION = 1


def metrics(y_true, y_pred) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    return {
        "confusion_matrix": matrix.tolist(),
        "matrix_labels": ["normal", "anomaly"],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "anomaly_precision": float(precision[0]),
        "anomaly_recall": float(recall[0]),
        "anomaly_f1": float(f1[0]),
        "normal_false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
    }


def select_svm(x, y, split: dict) -> tuple[Pipeline, list[dict]]:
    results, best, best_key = [], None, None
    for c in (0.1, 1.0, 10.0):
        for gamma in ("scale", 0.001, 0.01):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced")),
                ]
            )
            model.fit(x[split["train"]], y[split["train"]])
            report = metrics(y[split["validation"]], model.predict(x[split["validation"]]))
            results.append({"C": c, "gamma": gamma, **report})
            key = (report["balanced_accuracy"], report["anomaly_recall"])
            if best_key is None or key > best_key:
                best, best_key = model, key
    return best, results


def versions() -> dict:
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None  # CPU tests deliberately use an injected feature extractor.
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch_version,
    }


def train(dataset: Path, output: Path, config: Config, extractor, seed: int = 42) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("出力先が既に存在します。新しいモデルディレクトリを指定してください")
    background = load_background(config)
    records = load_records(dataset)
    split = split_items(records, seed)
    vectors = []
    hashes = set()
    for i, record in enumerate(records):
        if record["config_id"] != config.identity():
            raise ValueError("撮影時と学習時の設定・背景が一致しません")
        raw_path = record["_directory"] / "raw.png"
        digest = sha256_file(raw_path)
        if digest != record["raw_sha256"]:
            raise ValueError(f"撮影後に原画像が変更されています: {raw_path}")
        if digest in hashes:
            raise ValueError("同一原画像の重複があります。重複撮影を整理してください")
        hashes.add(digest)
        region = extract_region(read_image(raw_path), background, config.extraction)
        if region.status != "OK" or list(region.bbox) != record["bbox"]:
            raise ValueError(f"撮影時の領域抽出を再現できません: {record['sample_id']}")
        vectors.append(extractor.extract(region.crop))
        print(f"特徴抽出 {i + 1}/{len(records)}", flush=True)
    x = np.stack(vectors)
    if not np.isfinite(x).all():
        raise ValueError("特徴量に NaN/Inf があります")
    y = np.array([LABELS[r["label"]] for r in records])
    model, candidates = select_svm(x, y, split)
    report = {
        "seed": seed,
        "selection_metric": "balanced_accuracy, anomaly_recall, candidate_order",
        "selected": {"C": model.named_steps["svm"].C, "gamma": model.named_steps["svm"].gamma},
        "validation_candidates": candidates,
        "test": metrics(y[split["test"]], model.predict(x[split["test"]])),
        "counts": {
            name: {
                label: {
                    "images": sum(records[i]["label"] == label for i in indices),
                    "items": len(
                        {records[i]["item_id"] for i in indices if records[i]["label"] == label}
                    ),
                }
                for label in LABELS
            }
            for name, indices in split.items()
        },
    }
    assignments = {
        name: [
            {
                "sample_id": records[i]["sample_id"],
                "item_id": records[i]["item_id"],
                "label": records[i]["label"],
            }
            for i in indices
        ]
        for name, indices in split.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ingot-model-", dir=output.parent))
    try:
        shutil.copyfile(config.background, temporary / "background.png")
        portable = config.to_dict()
        portable["background"] = "background.png"
        write_json(temporary / "config.json", portable)
        joblib.dump(model, temporary / "svm.joblib")
        write_json(temporary / "report.json", report)
        write_json(temporary / "split.json", assignments)
        write_json(
            temporary / "metadata.json",
            {
                "bundle_version": BUNDLE_VERSION,
                "labels": LABELS,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_id": config.identity(),
                "feature_identity": extractor.identity(),
                "feature_dimension": int(x.shape[1]),
                "versions": versions(),
                "files": {
                    name: sha256_file(temporary / name)
                    for name in (
                        "config.json",
                        "background.png",
                        "svm.joblib",
                        "report.json",
                        "split.json",
                    )
                },
            },
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


def load_bundle(path: Path):
    """Only load bundles created by a trusted local operator (joblib uses pickle)."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["bundle_version"] != BUNDLE_VERSION or metadata["labels"] != LABELS:
        raise ValueError("モデル形式またはラベル定義が一致しません")
    for name in ("config.json", "background.png", "svm.joblib", "report.json", "split.json"):
        if sha256_file(path / name) != metadata["files"][name]:
            raise ValueError(f"モデルファイルが保存時と一致しません: {name}")
    identity = metadata["feature_identity"]
    if (
        identity["backbone"] != BACKBONE
        or identity["code_commit"] != DINOV3_COMMIT
        or identity["feature_spec"] != FEATURE_SPEC
    ):
        raise ValueError("DINOv3・特徴量の仕様が一致しません")
    config = load_config(path / "config.json")
    if config.identity() != metadata["config_id"]:
        raise ValueError("モデルと設定の識別情報が一致しません")
    if identity["input_size"] != config.features.input_size:
        raise ValueError("入力サイズが一致しません")
    if metadata["versions"]["scikit_learn"] != sklearn.__version__:
        raise ValueError("学習時と同じ scikit-learn バージョンを使用してください")
    model = joblib.load(path / "svm.joblib")
    if model.n_features_in_ != metadata["feature_dimension"] or list(model.classes_) != [0, 1]:
        raise ValueError("SVMの特徴量数・クラスが一致しません")
    return config, load_background(config), model, metadata


def verify_extractor(metadata: dict, extractor) -> None:
    if metadata["feature_identity"] != extractor.identity():
        raise ValueError("学習時とDINOv3重み・コード・前処理が一致しません")
