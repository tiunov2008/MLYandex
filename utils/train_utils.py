import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger as log
from torchmetrics.classification import F1Score

import torch


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def save_history(history, path):
    df = pd.DataFrame(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class StepLogger:
    def __init__(self, total_steps=None):
        self.total = total_steps
        self.step = 1

    def log(self, message):
        prefix = f"[{self.step}/{self.total}] " if self.total else ""
        log.info(f"{prefix}{message}")
        self.step += 1


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    f1_metric_kwargs,
    train=True,
):
    model.train(mode=train)
    total_loss = 0.0
    correct = 0
    total = 0
    f1_metric = F1Score(**f1_metric_kwargs).to(device)

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)
            f1_metric.update(logits, labels)

            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    f1 = f1_metric.compute().item()
    return total_loss / total, correct / total, f1
