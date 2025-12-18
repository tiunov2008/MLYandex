import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from loguru import logger as log
from PIL import Image

# ImageNet stats kept for RGB channels to stay aligned with previous runs
DEFAULT_RGB_MEAN = [0.485, 0.456, 0.406]
DEFAULT_RGB_STD = [0.229, 0.224, 0.225]


def _highpass_kernel():
    kernel = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return kernel.view(1, 1, 3, 3)


_HIGHPASS_KERNEL = _highpass_kernel()


def _compute_highpass(
    tensor,
    scale=4.0,
    kernel=_HIGHPASS_KERNEL,
):
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    gray = tensor.mean(dim=0, keepdim=True)  # 1 x H x W
    high = F.conv2d(gray.unsqueeze(0), kernel, padding=1) / scale
    return high.squeeze(0)  # 1 x H x W


class AddHighPassChannel:
    """Append a high-pass filtered grayscale channel after ToTensor."""

    def __init__(self, scale=4.0):
        self.scale = scale

    def __call__(self, tensor):
        high = _compute_highpass(tensor, scale=self.scale)
        return torch.cat([tensor, high], dim=0)


def _load_stats(stats_path):
    if not stats_path.exists():
        return None
    try:
        data = json.loads(stats_path.read_text())
        return float(data["mean"]), float(data["std"])
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"Failed to read high-pass stats from {stats_path}: {exc}")
        return None


def load_highpass_stats(stats_path):
    """Public wrapper to load cached high-pass statistics if present."""
    return _load_stats(stats_path)


def estimate_highpass_stats(
    df,
    img_dir,
    image_size,
    stats_path=None,
    step_logger=None,
):
    """
    Estimate mean/std for the high-pass channel over the training set.
    """
    if stats_path:
        cached = _load_stats(stats_path)
        if cached:
            hp_mean, hp_std = cached
            logger = getattr(step_logger, "log", None)
            if callable(logger):
                logger(f"Loaded cached high-pass stats from {stats_path}")
            else:
                log.info(f"Loaded cached high-pass stats from {stats_path}")
            return hp_mean, hp_std

    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()

    total_pixels = 0
    sum_val = 0.0
    sum_sq = 0.0

    for image_name in df["image"]:
        path = img_dir / image_name
        image = Image.open(path).convert("RGB")
        tensor = to_tensor(resize(image))
        high = _compute_highpass(tensor)
        pixels = high.numel()
        sum_val += high.sum().item()
        sum_sq += (high * high).sum().item()
        total_pixels += pixels

    if total_pixels == 0:
        raise ValueError("No pixels processed while estimating high-pass stats")

    hp_mean = sum_val / total_pixels
    variance = max(sum_sq / total_pixels - hp_mean * hp_mean, 0.0)
    hp_std = math.sqrt(variance)

    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "mean": hp_mean,
                    "std": hp_std,
                    "image_size": image_size,
                    "kernel": "laplacian_3x3_div4",
                },
                indent=2,
            )
        )
        logger = getattr(step_logger, "log", None)
        if callable(logger):
            logger(f"Saved high-pass stats to {stats_path}")
        else:
            log.info(f"Saved high-pass stats to {stats_path}")

    return hp_mean, hp_std


def normalize_stats_with_highpass(hp_mean, hp_std):
    mean = list(DEFAULT_RGB_MEAN) + [hp_mean]
    std = list(DEFAULT_RGB_STD) + [hp_std]
    return mean, std
