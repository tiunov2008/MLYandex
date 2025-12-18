from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from loguru import logger as log
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import load_dataframe
from utils.model_utils import load_trained_model
from utils.transforms import (
    AddHighPassChannel,
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    estimate_highpass_stats,
    load_highpass_stats,
    normalize_stats_with_highpass,
)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = sorted(img_paths)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path.stem


def _prepare_transform(image_size, use_highpass, hp_stats_path):
    if not use_highpass:
        mean, std = DEFAULT_RGB_MEAN, DEFAULT_RGB_STD
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    hp_mean = hp_std = None
    if hp_stats_path:
        cached = load_highpass_stats(hp_stats_path)
        if cached:
            hp_mean, hp_std = cached
            log.info(f"Loaded high-pass stats from {hp_stats_path}")
    if hp_mean is None or hp_std is None:
        log.info("High-pass stats not found; estimating from training set")
        import config

        df = load_dataframe()
        hp_mean, hp_std = estimate_highpass_stats(
            df=df,
            img_dir=config.TRAIN_DIR,
            image_size=image_size,
            stats_path=hp_stats_path or config.HP_STATS_PATH,
            step_logger=None,
        )

    mean, std = normalize_stats_with_highpass(hp_mean, hp_std)
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            AddHighPassChannel(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _prepare_loader(
    image_size=128,
    use_highpass=False,
    hp_stats_path=None,
):
    transform = _prepare_transform(image_size=image_size, use_highpass=use_highpass, hp_stats_path=hp_stats_path)

    import config

    test_dir = config.DATASET_DIR / "test_images"
    img_paths = list(test_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg files found in {test_dir}")

    dataset = TestDataset(img_paths, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)


def _load_models(model_specs, device):
    models = []
    for model_name, model_dir in model_specs:
        models.append(load_trained_model(model_name, model_dir, device))
    if not models:
        raise ValueError("model_specs is empty; provide at least one model")
    return models


def ensemble_predict(
    model_specs=None,
    voting="soft",
    out_path=Path("predictions.csv"),
):
    import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    if model_specs is None:
        model_specs = [("simple_cnn_v2", config.MODELS_ROOT / "simple_cnn_v2")]

    names = {name for name, _ in model_specs}
    use_highpass = "simple_cnn_v2" in names
    if use_highpass and len(names) > 1:
        raise ValueError("simple_cnn_v2 expects 4-channel inputs and cannot be ensembled with 3-channel models")
    stats_path = config.HP_STATS_PATH if use_highpass else None

    loader = _prepare_loader(use_highpass=use_highpass, hp_stats_path=stats_path)
    models = _load_models(model_specs, device)
    for model in models:
        model.eval()

    ids = []
    preds = []
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            probs = [torch.softmax(model(images), dim=1) for model in models]

            if voting == "soft":
                combined = torch.stack(probs, dim=0).mean(dim=0)
                batch_preds = combined.argmax(1)
            elif voting == "hard":
                votes = torch.stack([p.argmax(1) for p in probs], dim=0)
                batch_preds = votes.mode(dim=0).values
            else:
                raise ValueError(f"Unknown voting strategy: {voting}")

            ids.extend(names)
            preds.extend(batch_preds.cpu().tolist())

    df = pd.DataFrame({"id": ids, "label": preds})
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved ensemble predictions to {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    VOTING = "soft"  # "soft" | "hard"
    OUT_PATH = Path("predictions.csv")

    MODEL_SPECS = [
        ("simple_cnn_v2", Path("models") / "simple_cnn_v2" / "seed_100"),
        ("simple_cnn_v2", Path("models") / "simple_cnn_v2" / "seed_101"),
        ("simple_cnn_v2", Path("models") / "simple_cnn_v2" / "seed_102"),
        ("simple_cnn_v2", Path("models") / "simple_cnn_v2" / "seed_103"),
        ("simple_cnn_v2", Path("models") / "simple_cnn_v2" / "seed_104"),
    ]

    ensemble_predict(model_specs=MODEL_SPECS, voting=VOTING, out_path=OUT_PATH)
