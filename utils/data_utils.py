from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import CSV_PATH, TRAIN_DIR
from utils.train_utils import StepLogger


def load_dataframe(step_logger: StepLogger | None = None) -> pd.DataFrame:
    if step_logger:
        step_logger.log("Reading train_solution.csv")
    df = pd.read_csv(CSV_PATH)

    if list(df.columns) == ["0", "0.1"]:
        if step_logger:
            step_logger.log("Renaming columns 0/0.1 -> id/label")
        df = df.rename(columns={"0": "id", "0.1": "label"})

    if step_logger:
        step_logger.log("Filtering rows by existing jpg files")
    df["image"] = df["id"].astype(str) + ".jpg"
    df["label"] = df["label"].astype(int)
    existing = {p.name for p in TRAIN_DIR.glob("*.jpg")}
    df = df[df["image"].isin(existing)].reset_index(drop=True)
    return df


class PosesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label
