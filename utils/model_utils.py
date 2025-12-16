from __future__ import annotations

from pathlib import Path

from loguru import logger as log
import torch

from models.convnext_tiny.main import ConvNeXtTiny
from models.simple_cnn.main import SimpleCNN
from models.simple_cnn_v2.main import SimpleCNN as SimpleCNNv2


def build_model(model_name: str, device: torch.device):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=2).to(device)
    if model_name == "simple_cnn_v2":
        return SimpleCNNv2(num_classes=2).to(device)
    if model_name == "convnext_tiny":
        return ConvNeXtTiny(num_classes=2).to(device)
    raise ValueError(f"Unknown model name: {model_name}")


def load_trained_model(
    model_name: str,
    model_dir: Path,
    device: torch.device,
    prefer_checkpoint: bool = False,
):
    model = build_model(model_name, device)
    ckpt_path = model_dir / "last_ckpt.pth"
    weights_path = model_dir / "pose_cnn.pth"
    source = None
    if prefer_checkpoint and ckpt_path.exists():
        source = ckpt_path
    elif weights_path.exists():
        source = weights_path
    elif ckpt_path.exists():
        source = ckpt_path
    else:
        seed_dirs = sorted(
            [p for p in model_dir.glob("seed_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for seed_dir in seed_dirs:
            ckpt_candidate = seed_dir / "last_ckpt.pth"
            weights_candidate = seed_dir / "pose_cnn.pth"
            if prefer_checkpoint and ckpt_candidate.exists():
                source = ckpt_candidate
                log.info(f"Using checkpoint from {ckpt_candidate}")
                break
            if weights_candidate.exists():
                source = weights_candidate
                log.info(f"Using weights from {weights_candidate}")
                break
            if ckpt_candidate.exists():
                source = ckpt_candidate
                log.info(f"Using checkpoint from {ckpt_candidate}")
                break
    if source is None:
        raise FileNotFoundError(f"No weights found in {model_dir} or its seed_* subfolders")

    state = torch.load(source, map_location=device)
    payload = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(payload)
    model.eval()
    log.info(f"Loaded weights from {source}")
    return model
