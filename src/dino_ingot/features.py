"""Local-only, frozen DINOv3 loader. torch is intentionally an external dependency."""

import subprocess
from pathlib import Path

from .config import BACKBONE, DINOV3_COMMIT, FEATURE_SPEC, sha256_file
from .vision import pool_features, preprocess


def verify_repository(repo: Path) -> str:
    repo = Path(repo).resolve()
    if not (repo / "hubconf.py").is_file():
        raise ValueError(f"DINOv3 の hubconf.py がありません: {repo}")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if commit != DINOV3_COMMIT or dirty:
        raise ValueError(f"DINOv3 は変更のない指定コミットを使用してください: {DINOV3_COMMIT}")
    return commit


class DinoExtractor:
    def __init__(self, repo: Path, weights: Path, input_size: int, device: str = "cuda"):
        if device not in {"cuda", "cpu"}:
            raise ValueError("device は cuda または cpu です")
        self.commit = verify_repository(repo)
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise ValueError(f"DINOv3 の重みがありません: {weights}")
        self.weights_sha256 = sha256_file(weights)
        self.input_size = input_size
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA が利用できません。Jetson用PyTorchを確認するか --device cpu を指定"
            )
        self.torch = torch
        self.device = device
        self.model = torch.hub.load(
            str(Path(repo).resolve()), BACKBONE, source="local", weights=str(weights)
        )
        self.model.eval().requires_grad_(False).to(device)

    def identity(self) -> dict:
        return {
            "backbone": BACKBONE,
            "code_commit": self.commit,
            "weights_sha256": self.weights_sha256,
            "feature_spec": FEATURE_SPEC,
            "input_size": self.input_size,
        }

    def extract(self, crop):
        torch = self.torch
        tensor = torch.from_numpy(preprocess(crop, self.input_size)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model.forward_features(tensor)
        cls = outputs["x_norm_clstoken"].float().cpu().numpy()
        patches = outputs["x_norm_patchtokens"].float().cpu().numpy()
        return pool_features(cls, patches)[0]
