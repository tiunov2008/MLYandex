from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(df: pd.DataFrame, out_path: Path) -> None:
    """Plot loss/accuracy/F1 curves and save them to a PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["epoch"], df["train_f1"], label="train_f1")
    axes[2].plot(df["epoch"], df["val_f1"], label="val_f1")
    axes[2].set_title("F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1-score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


__all__ = ["plot_history"]
