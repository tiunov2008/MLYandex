import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader, random_split

from utils.data_utils import PosesDataset, load_dataframe
from utils.model_utils import build_model
from utils.plots import plot_history
from utils.train_utils import StepLogger, run_epoch, save_history, set_seed
from utils.transforms import (
    AddHighPassChannel,
    DEFAULT_RGB_MEAN,
    DEFAULT_RGB_STD,
    estimate_highpass_stats,
    normalize_stats_with_highpass,
)

class TrainArgs:
    def __init__(
        self,
        model_name,
        augmentation,
        seeds,
        output_root,
        run_name,
        image_size,
        batch_size,
        lr,
        num_epochs,
        early_stop_patience,
        dl_workers,
    ):
        self.model_name = model_name
        self.augmentation = augmentation
        self.seeds = tuple(seeds)
        self.output_root = output_root
        self.run_name = run_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.dl_workers = dl_workers


SEEDS = [105]
STOP_FILE = Path("stop.txt")
EARLY_STOP_PATIENCE = 20
IMAGE_SIZE = 128
NUM_EPOCHS = 300


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_transforms(
    *,
    model_name,
    augmentation,
    df,
    train_dir,
    image_size,
    hp_stats_path,
    step_logger,
):
    mean = list(DEFAULT_RGB_MEAN)
    std = list(DEFAULT_RGB_STD)

    add_highpass = None
    if model_name == "simple_cnn_v2":
        hp_mean, hp_std = estimate_highpass_stats(
            df=df,
            img_dir=train_dir,
            image_size=image_size,
            stats_path=hp_stats_path,
            step_logger=step_logger,
        )
        mean, std = normalize_stats_with_highpass(hp_mean, hp_std)
        add_highpass = AddHighPassChannel()
        step_logger.log(f"High-pass stats: mean={hp_mean:.6f}, std={hp_std:.6f}")

    if augmentation == "none":
        train_pil = [T.Resize((image_size, image_size))]
    elif augmentation == "baseline":
        train_pil = [
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
        ]
    elif augmentation == "ensemble_1":
        train_pil = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
        ]
    elif augmentation == "ensemble_2":
        train_pil = [
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        ]
    else:
        raise ValueError(f"Unknown augmentation preset: {augmentation}")

    train_transforms = [*train_pil, T.ToTensor()]
    if add_highpass is not None:
        train_transforms.append(add_highpass)
    train_transforms.append(T.Normalize(mean=mean, std=std))
    train_transform = T.Compose(train_transforms)

    val_transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]
    if add_highpass is not None:
        val_transforms.append(add_highpass)
    val_transforms.append(T.Normalize(mean=mean, std=std))
    val_transform = T.Compose(val_transforms)

    return train_transform, val_transform


def train_one_seed(seed, args):
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    step_logger = StepLogger(total_steps=18)

    import config

    run_group_dir = args.output_root / args.run_name
    run_group_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_group_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.csv"
    history_plot_path = run_dir / "training_history.png"
    checkpoint_path = run_dir / "last_ckpt.pth"
    weights_path = run_dir / "pose_cnn.pth"
    hp_stats_path = run_group_dir / "hp_stats.json"
    stopped = False

    df = load_dataframe(step_logger=step_logger)
    train_dir = config.TRAIN_DIR
    label_counts = Counter(df["label"])
    step_logger.log(f"Samples: {len(df)}, label distribution: {label_counts}")

    step_logger.log("Preparing transforms")
    train_transform, val_transform = _build_transforms(
        model_name=args.model_name,
        augmentation=args.augmentation,
        df=df,
        train_dir=train_dir,
        image_size=args.image_size,
        hp_stats_path=hp_stats_path,
        step_logger=step_logger,
    )

    step_logger.log("Building base dataset")
    full_dataset = PosesDataset(df, train_dir, transform=None)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    step_logger.log("Splitting train/val (random_split)")
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    step_logger.log("Creating train/val datasets with transforms")
    train_dataset = PosesDataset(
        df.iloc[train_subset.indices].reset_index(drop=True),
        train_dir,
        transform=train_transform,
    )
    val_dataset = PosesDataset(
        df.iloc[val_subset.indices].reset_index(drop=True),
        train_dir,
        transform=val_transform,
    )

    step_logger.log("Creating dataloaders")
    dl_workers = args.dl_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dl_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_logger.log(f"Using device: {device}")

    history_records = []
    history_df = None
    start_epoch = 1
    checkpoint_state = None

    if checkpoint_path.exists():
        step_logger.log(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        history_records = list(checkpoint_state.get("history", []))
        if history_records:
            history_df = pd.DataFrame(history_records)
        start_epoch = int(checkpoint_state.get("epoch", 0)) + 1
        step_logger.log(f"Resuming from epoch {start_epoch}")
    elif history_path.exists():
        step_logger.log(f"Loading existing history from {history_path}")
        history_df = pd.read_csv(history_path)
        if not history_df.empty:
            history_records.extend(history_df.to_dict(orient="records"))
            start_epoch = int(history_df["epoch"].max()) + 1
            step_logger.log(f"Continuing history from epoch {start_epoch}")

    step_logger.log(f"Model selected: {args.model_name}")
    step_logger.log(f"Augmentation: {args.augmentation}")
    step_logger.log(f"Saving artifacts to {run_dir}")

    model = build_model(args.model_name, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    f1_metric_kwargs = {
        "task": "multiclass",
        "num_classes": 2,
        "average": "macro",
    }

    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0
    if history_df is not None and not history_df.empty:
        best_val_f1 = float(history_df["val_f1"].max())

    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state["model"])
        if "optimizer" in checkpoint_state:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        best_val_f1 = float(checkpoint_state.get("best_val_f1", best_val_f1))
        best_model_state = checkpoint_state.get("best_model_state") or best_model_state
        epochs_without_improvement = int(
            checkpoint_state.get("epochs_without_improvement", epochs_without_improvement)
        )
        step_logger.log(
            f"Checkpoint restored: epoch={checkpoint_state.get('epoch', 0)}, "
            f"best_val_f1={best_val_f1 if best_val_f1 != float('-inf') else 'n/a'}"
        )
    elif weights_path.exists():
        step_logger.log("Loading existing best weights")
        weights_state = torch.load(weights_path, map_location=device)
        payload = (
            weights_state["model"]
            if isinstance(weights_state, dict) and "model" in weights_state
            else weights_state
        )
        model.load_state_dict(payload)
        best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    step_logger.log("Starting training loop")

    def save_checkpoint(epoch):
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history_records,
                "best_val_f1": best_val_f1,
                "best_model_state": best_model_state,
                "epochs_without_improvement": epochs_without_improvement,
            },
            checkpoint_path,
        )

    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            if STOP_FILE.exists():
                step_logger.log(f"Found {STOP_FILE}, saving checkpoint and exiting")
                save_checkpoint(max(epoch - 1, 0))
                break

            tr_loss, tr_acc, tr_f1 = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                f1_metric_kwargs=f1_metric_kwargs,
                train=True,
            )
            val_loss, val_acc, val_f1 = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                f1_metric_kwargs=f1_metric_kwargs,
                train=False,
            )

            history_records.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "train_f1": tr_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }
            )

            log.info(
                f"Seed {seed} | Epoch {epoch}: "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                log.info(
                    f"Seed {seed} | New best val_f1={best_val_f1:.4f} at epoch {epoch}, "
                    "weights snapshotted"
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    step_logger.log(
                        f"Early stopping at epoch {epoch} "
                        f"(no val f1 improvement for {args.early_stop_patience} epochs)"
                    )
                    save_checkpoint(epoch)
                    break

            save_checkpoint(epoch)
    except KeyboardInterrupt:
        log.info("Training interrupted by user, saving checkpoint.")
        save_checkpoint(epoch if "epoch" in locals() else 0)
        stopped = True
    except Exception as exc:
        log.exception(f"Training failed: {exc}")
        save_checkpoint(epoch if "epoch" in locals() else 0)
        stopped = True
        raise
    finally:
        if history_records:
            step_logger.log(f"Saving training history to {history_path}")
            history_df = save_history(history_records, history_path)
            step_logger.log(f"Saving training curves to {history_plot_path}")
            plot_history(history_df, history_plot_path)
        else:
            step_logger.log("No history records collected")

        step_logger.log("Saving model weights to disk")
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), weights_path)
        log.info(f"Saved model to {weights_path}")

    return stopped


def _parse_args():
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument(
        "--model",
        dest="model_name",
        default="simple_cnn",
        choices=["simple_cnn", "simple_cnn_v2"],
        help="Имя модели (см. utils/model_utils.py)",
    )
    parser.add_argument(
        "--augmentation",
        default="baseline",
        choices=["none", "baseline", "ensemble_1", "ensemble_2"],
        help="Пресет аугментаций",
    )
    parser.add_argument("--seed", type=int, default=None, help="Один seed (alias для --seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Список seed-ов")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--dl-workers", type=int, default=2)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models"),
        help="Корневая папка для артефактов",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Имя группы запуска (по умолчанию = --model)",
    )
    ns = parser.parse_args()

    seeds = ns.seeds
    if ns.seed is not None:
        if seeds is not None:
            raise SystemExit("Use either --seed or --seeds, not both")
        seeds = [ns.seed]
    if seeds is None:
        seeds = list(SEEDS)

    run_name = ns.run_name or ns.model_name

    return TrainArgs(
        model_name=ns.model_name,
        augmentation=ns.augmentation,
        seeds=seeds,
        output_root=ns.output_root,
        run_name=run_name,
        image_size=ns.image_size,
        batch_size=ns.batch_size,
        lr=ns.lr,
        num_epochs=ns.num_epochs,
        early_stop_patience=ns.early_stop_patience,
        dl_workers=ns.dl_workers,
    )


def main():
    args = _parse_args()
    for seed in args.seeds:
        log.info(f"=== Training seed {seed} ===")
        if train_one_seed(seed, args):
            break


if __name__ == "__main__":
    main()
