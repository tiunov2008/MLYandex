import os
from pathlib import Path

import kagglehub
from loguru import logger as log

DATASET_HANDLE = "zv3zdochka/ml-intensive-yandex-academy-autumn-2025"
_DATASET_LOGGED_FLAG = "DATASET_LOGGED"
_LOGGED = False


def _maybe_log_once(message):
    global _LOGGED
    if _LOGGED:
        return
    if os.environ.get(_DATASET_LOGGED_FLAG) == "1":
        _LOGGED = True
        return
    log.info(message)
    os.environ[_DATASET_LOGGED_FLAG] = "1"
    _LOGGED = True


def get_dataset_dir():
    local_path = Path("dataset")
    if local_path.exists():
        _maybe_log_once(f"Using local dataset at {local_path.resolve()}")
        return local_path

    _maybe_log_once("Local dataset not found. Downloading from Kaggle...")
    base_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    dataset_path = base_path / "dataset"
    if dataset_path.exists():
        return dataset_path

    return base_path


MODEL_NAME = "simple_cnn"
DATASET_DIR = get_dataset_dir()
CSV_PATH = DATASET_DIR / "train_solution.csv"
TRAIN_DIR = DATASET_DIR / "train_images"

MODELS_ROOT = Path("models")
MODEL_DIR = MODELS_ROOT / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = MODEL_DIR / "training_history.csv"
HISTORY_PLOT_PATH = MODEL_DIR / "training_history.png"
HP_STATS_PATH = MODEL_DIR / "hp_stats.json"

