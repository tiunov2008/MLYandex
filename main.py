import random  # импортируем модуль random для воспроизводимости
from pathlib import Path  # файл-флаг для мягкой остановки
from collections import Counter  # счётчик для статистики классов

import numpy as np  # numpy для численных операций
import pandas as pd  # работа с табличными данными
from loguru import logger as log  # логгер для удобного логирования

import torch  # фреймворк PyTorch
from torch import nn  # модуль нейросетевых слоёв
from torch.utils.data import DataLoader, random_split  # загрузчики данных и разбиение
import torchvision.transforms as T  # преобразования изображений

from config import HP_STATS_PATH, MODEL_DIR, MODEL_NAME, TRAIN_DIR  # пути и конфиг
from utils.data_utils import PosesDataset, load_dataframe  # датасет и загрузка датафрейма
from utils.model_utils import build_model  # фабрика модели
from utils.plots import plot_history  # построение графиков обучения
from utils.train_utils import StepLogger, run_epoch, save_history, set_seed  # утилиты обучения
from utils.transforms import (  # импортируем имена из модуля
    AddHighPassChannel,  # выполняем шаг логики
    DEFAULT_RGB_MEAN,  # выполняем шаг логики
    DEFAULT_RGB_STD,  # выполняем шаг логики
    estimate_highpass_stats,  # выполняем шаг логики
    normalize_stats_with_highpass,  # выполняем шаг логики
)  # закрываем скобки/структуру


SEEDS = [105]  # список сидов для запусков
STOP_FILE = Path("stop.txt")  # файл-флаг для мягкой остановки
EARLY_STOP_PATIENCE = 20  # simple early stopping to curb overfitting


def seed_worker(_worker_id: int) -> None:  # объявляем функцию
    worker_seed = torch.initial_seed() % 2**32  # уникальный сид для воркера
    np.random.seed(worker_seed)  # сидим numpy
    random.seed(worker_seed)  # сидим random


def train_one_seed(seed: int) -> bool:  # объявляем функцию
    set_seed(seed)  # выполняем шаг логики
    generator = torch.Generator().manual_seed(seed)  # присваиваем значение переменной
    step_logger = StepLogger(total_steps=18)  # присваиваем значение переменной
    run_dir = MODEL_DIR / f"seed_{seed}"  # присваиваем значение переменной
    run_dir.mkdir(parents=True, exist_ok=True)  # создаём директорию, если её нет
    history_path = run_dir / "training_history.csv"  # присваиваем значение переменной
    history_plot_path = run_dir / "training_history.png"  # присваиваем значение переменной
    checkpoint_path = run_dir / "last_ckpt.pth"  # присваиваем значение переменной
    weights_path = run_dir / "pose_cnn.pth"  # присваиваем значение переменной
    stopped = False  # присваиваем значение переменной

    df = load_dataframe(step_logger=step_logger)  # присваиваем значение переменной
    label_counts = Counter(df["label"])  # присваиваем значение переменной
    step_logger.log(f"Samples: {len(df)}, label distribution: {label_counts}")  # логируем этап/метрику
    step_logger.log("Preparing transforms")  # логируем этап/метрику
    image_size = 128  # присваиваем значение переменной

    hp_mean = hp_std = None  # присваиваем значение переменной
    mean = list(DEFAULT_RGB_MEAN)  # присваиваем значение переменной
    std = list(DEFAULT_RGB_STD)  # присваиваем значение переменной
    add_highpass = None  # присваиваем значение переменной
    if MODEL_NAME == "simple_cnn_v2":  # начинаем новый блок
        hp_mean, hp_std = estimate_highpass_stats(  # присваиваем значение переменной
            df=df,  # присваиваем значение переменной
            img_dir=TRAIN_DIR,  # присваиваем значение переменной
            image_size=image_size,  # присваиваем значение переменной
            stats_path=HP_STATS_PATH,  # присваиваем значение переменной
            step_logger=step_logger,  # присваиваем значение переменной
        )  # закрываем скобки/структуру
        mean, std = normalize_stats_with_highpass(hp_mean, hp_std)  # присваиваем значение переменной
        add_highpass = AddHighPassChannel()  # присваиваем значение переменной
        step_logger.log(f"High-pass stats: mean={hp_mean:.6f}, std={hp_std:.6f}")  # логируем этап/метрику

    train_transforms = [  # присваиваем значение переменной
        T.RandomResizedCrop(size=(image_size, image_size), scale=(0.9, 1.0)),  # присваиваем значение переменной
        T.RandomHorizontalFlip(),  # выполняем шаг логики
        T.RandomRotation(15),  # выполняем шаг логики
        T.ToTensor(),  # выполняем шаг логики
    ]  # закрываем скобки/структуру
    if add_highpass:  # начинаем новый блок
        train_transforms.append(add_highpass)  # выполняем шаг логики
    train_transforms.append(T.Normalize(mean=mean, std=std))  # присваиваем значение переменной
    train_transform = T.Compose(train_transforms)  # присваиваем значение переменной

    val_transforms = [  # присваиваем значение переменной
        T.Resize((image_size, image_size)),  # выполняем шаг логики
        T.ToTensor(),  # выполняем шаг логики
    ]  # закрываем скобки/структуру
    if add_highpass:  # начинаем новый блок
        val_transforms.append(add_highpass)  # выполняем шаг логики
    val_transforms.append(T.Normalize(mean=mean, std=std))  # присваиваем значение переменной
    val_transform = T.Compose(val_transforms)  # присваиваем значение переменной

    step_logger.log("Building base dataset")  # логируем этап/метрику
    full_dataset = PosesDataset(df, TRAIN_DIR, transform=None)  # присваиваем значение переменной

    val_size = int(0.2 * len(full_dataset))  # присваиваем значение переменной
    train_size = len(full_dataset) - val_size  # присваиваем значение переменной

    step_logger.log("Splitting train/val (random_split)")  # логируем этап/метрику
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)  # делим датасет на части

    step_logger.log("Creating train/val datasets with transforms")  # логируем этап/метрику
    train_dataset = PosesDataset(  # присваиваем значение переменной
        df.iloc[train_subset.indices].reset_index(drop=True),  # присваиваем значение переменной
        TRAIN_DIR,  # выполняем шаг логики
        transform=train_transform,  # присваиваем значение переменной
    )  # закрываем скобки/структуру

    val_dataset = PosesDataset(  # присваиваем значение переменной
        df.iloc[val_subset.indices].reset_index(drop=True),  # присваиваем значение переменной
        TRAIN_DIR,  # выполняем шаг логики
        transform=val_transform,  # присваиваем значение переменной
    )  # закрываем скобки/структуру

    step_logger.log("Creating dataloaders")  # логируем этап/метрику
    dl_workers = 2  # присваиваем значение переменной
    train_loader = DataLoader(  # создаём загрузчик батчей
        train_dataset,  # выполняем шаг логики
        batch_size=64,  # присваиваем значение переменной
        shuffle=True,  # присваиваем значение переменной
        num_workers=dl_workers,  # присваиваем значение переменной
        generator=generator,  # присваиваем значение переменной
        worker_init_fn=seed_worker,  # присваиваем значение переменной
    )  # закрываем скобки/структуру
    val_loader = DataLoader(  # создаём загрузчик батчей
        val_dataset,  # выполняем шаг логики
        batch_size=64,  # присваиваем значение переменной
        shuffle=False,  # присваиваем значение переменной
        num_workers=dl_workers,  # присваиваем значение переменной
        generator=generator,  # присваиваем значение переменной
        worker_init_fn=seed_worker,  # присваиваем значение переменной
    )  # закрываем скобки/структуру
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # выбираем устройство вычислений (GPU/CPU)
    step_logger.log(f"Using device: {device}")  # логируем этап/метрику

    history_records: list[dict] = []  # присваиваем значение переменной
    history_df = None  # присваиваем значение переменной
    start_epoch = 1  # присваиваем значение переменной
    checkpoint_state = None  # присваиваем значение переменной
    if checkpoint_path.exists():  # начинаем новый блок
        step_logger.log(f"Loading checkpoint from {checkpoint_path}")  # логируем этап/метрику
        checkpoint_state = torch.load(checkpoint_path, map_location=device)  # загружаем состояние с диска
        history_records = checkpoint_state.get("history", [])  # присваиваем значение переменной
        if history_records:  # начинаем новый блок
            history_df = pd.DataFrame(history_records)  # присваиваем значение переменной
        start_epoch = checkpoint_state.get("epoch", 0) + 1  # присваиваем значение переменной
        step_logger.log(f"Resuming from epoch {start_epoch}")  # логируем этап/метрику
    elif history_path.exists():  # начинаем новый блок
        step_logger.log(f"Loading existing history from {history_path}")  # логируем этап/метрику
        history_df = pd.read_csv(history_path)  # присваиваем значение переменной
        if not history_df.empty:  # начинаем новый блок
            history_records.extend(history_df.to_dict(orient="records"))  # присваиваем значение переменной
            start_epoch = int(history_df["epoch"].max()) + 1  # присваиваем значение переменной
            step_logger.log(f"Continuing history from epoch {start_epoch}")  # логируем этап/метрику

    step_logger.log(f"Model selected: {MODEL_NAME}")  # логируем этап/метрику
    step_logger.log(f"Saving artifacts to {run_dir}")  # логируем этап/метрику
    model = build_model(MODEL_NAME, device)  # присваиваем значение переменной
    criterion = nn.CrossEntropyLoss()  # присваиваем значение переменной

    f1_metric_kwargs = {"task": "multiclass", "num_classes": 2, "average": "macro"}  # присваиваем значение переменной

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # присваиваем значение переменной
    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # присваиваем значение переменной
    best_val_f1 = float("-inf")  # присваиваем значение переменной
    epochs_without_improvement = 0  # присваиваем значение переменной
    if history_df is not None and not history_df.empty:  # начинаем новый блок
        best_val_f1 = history_df["val_f1"].max()  # присваиваем значение переменной

    if checkpoint_state:  # начинаем новый блок
        model.load_state_dict(checkpoint_state["model"])  # выполняем шаг логики
        if "optimizer" in checkpoint_state:  # начинаем новый блок
            optimizer.load_state_dict(checkpoint_state["optimizer"])  # выполняем шаг логики
        best_val_f1 = checkpoint_state.get("best_val_f1", best_val_f1)  # присваиваем значение переменной
        best_model_state = checkpoint_state.get("best_model_state") or {  # присваиваем значение переменной
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()  # выполняем шаг логики
        }  # закрываем скобки/структуру
        epochs_without_improvement = checkpoint_state.get("epochs_without_improvement", epochs_without_improvement)  # присваиваем значение переменной
        step_logger.log(  # логируем этап/метрику
            f"Checkpoint restored: epoch={checkpoint_state.get('epoch', 0)}, "  # присваиваем значение переменной
            f"best_val_f1={best_val_f1 if best_val_f1 != float('-inf') else 'n/a'}"  # выполняем шаг логики
        )  # закрываем скобки/структуру
    elif weights_path.exists():  # начинаем новый блок
        step_logger.log("Loading existing best weights")  # логируем этап/метрику
        weights_state = torch.load(weights_path, map_location=device)  # загружаем состояние с диска
        payload = weights_state["model"] if isinstance(weights_state, dict) and "model" in weights_state else weights_state  # присваиваем значение переменной
        model.load_state_dict(payload)  # выполняем шаг логики
        best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # присваиваем значение переменной

    step_logger.log("Starting training loop")  # логируем этап/метрику

    def save_checkpoint(epoch: int) -> None:  # объявляем функцию
        torch.save(  # сохраняем состояние на диск
            {  # выполняем шаг логики
                "epoch": epoch,  # выполняем шаг логики
                "model": model.state_dict(),  # выполняем шаг логики
                "optimizer": optimizer.state_dict(),  # выполняем шаг логики
                "history": history_records,  # выполняем шаг логики
                "best_val_f1": best_val_f1,  # выполняем шаг логики
                "best_model_state": best_model_state,  # выполняем шаг логики
                "epochs_without_improvement": epochs_without_improvement,  # выполняем шаг логики
            },  # закрываем скобки/структуру
            checkpoint_path,  # выполняем шаг логики
        )  # закрываем скобки/структуру

    num_epochs = 300  # присваиваем значение переменной
    try:  # начинаем новый блок
        for epoch in range(start_epoch, num_epochs + 1):  # начинаем новый блок
            if STOP_FILE.exists():  # начинаем новый блок
                step_logger.log(f"Found {STOP_FILE}, saving checkpoint and exiting")  # логируем этап/метрику
                save_checkpoint(max(epoch - 1, 0))  # выполняем шаг логики
                break  # выполняем шаг логики
            tr_loss, tr_acc, tr_f1 = run_epoch(  # присваиваем значение переменной
                model=model,  # присваиваем значение переменной
                loader=train_loader,  # присваиваем значение переменной
                criterion=criterion,  # присваиваем значение переменной
                optimizer=optimizer,  # присваиваем значение переменной
                device=device,  # присваиваем значение переменной
                f1_metric_kwargs=f1_metric_kwargs,  # присваиваем значение переменной
                train=True,  # присваиваем значение переменной
            )  # закрываем скобки/структуру
            val_loss, val_acc, val_f1 = run_epoch(  # присваиваем значение переменной
                model=model,  # присваиваем значение переменной
                loader=val_loader,  # присваиваем значение переменной
                criterion=criterion,  # присваиваем значение переменной
                optimizer=optimizer,  # присваиваем значение переменной
                device=device,  # присваиваем значение переменной
                f1_metric_kwargs=f1_metric_kwargs,  # присваиваем значение переменной
                train=False,  # присваиваем значение переменной
            )  # закрываем скобки/структуру
            history_records.append(  # выполняем шаг логики
                {  # выполняем шаг логики
                    "epoch": epoch,  # выполняем шаг логики
                    "train_loss": tr_loss,  # выполняем шаг логики
                    "train_acc": tr_acc,  # выполняем шаг логики
                    "train_f1": tr_f1,  # выполняем шаг логики
                    "val_loss": val_loss,  # выполняем шаг логики
                    "val_acc": val_acc,  # выполняем шаг логики
                    "val_f1": val_f1,  # выполняем шаг логики
                }  # закрываем скобки/структуру
            )  # закрываем скобки/структуру
            log.info(  # выполняем шаг логики
                f"Seed {seed} | Epoch {epoch}: "  # выполняем шаг логики
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "  # присваиваем значение переменной
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}"  # присваиваем значение переменной
            )  # закрываем скобки/структуру
            if val_f1 > best_val_f1:  # начинаем новый блок
                best_val_f1 = val_f1  # присваиваем значение переменной
                epochs_without_improvement = 0  # присваиваем значение переменной
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # присваиваем значение переменной
                log.info(f"Seed {seed} | New best val_f1={best_val_f1:.4f} at epoch {epoch}, weights snapshotted")  # присваиваем значение переменной
            else:  # начинаем новый блок
                epochs_without_improvement += 1  # присваиваем значение переменной
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:  # начинаем новый блок
                    step_logger.log(  # логируем этап/метрику
                        f"Early stopping at epoch {epoch} "  # выполняем шаг логики
                        f"(no val f1 improvement for {EARLY_STOP_PATIENCE} epochs)"  # выполняем шаг логики
                    )  # закрываем скобки/структуру
                    save_checkpoint(epoch)  # выполняем шаг логики
                    break  # выполняем шаг логики
            save_checkpoint(epoch)  # выполняем шаг логики
    except KeyboardInterrupt:  # начинаем новый блок
        log.info("Training interrupted by user, saving checkpoint.")  # выполняем шаг логики
        save_checkpoint(epoch if "epoch" in locals() else 0)  # выполняем шаг логики
        stopped = True  # присваиваем значение переменной
    except Exception as exc:  # начинаем новый блок
        log.exception(f"Training failed: {exc}")  # выполняем шаг логики
        save_checkpoint(epoch if "epoch" in locals() else 0)  # выполняем шаг логики
        stopped = True  # присваиваем значение переменной
        raise  # выполняем шаг логики
    finally:  # начинаем новый блок
        if history_records:  # начинаем новый блок
            step_logger.log(f"Saving training history to {history_path}")  # логируем этап/метрику
            history_df = save_history(history_records, history_path)  # присваиваем значение переменной
            step_logger.log(f"Saving training curves to {history_plot_path}")  # логируем этап/метрику
            plot_history(history_df, history_plot_path)  # выполняем шаг логики
        else:  # начинаем новый блок
            step_logger.log("No history records collected")  # логируем этап/метрику

        step_logger.log("Saving model weights to disk")  # логируем этап/метрику
        if best_model_state is not None:  # начинаем новый блок
            model.load_state_dict(best_model_state)  # выполняем шаг логики
        torch.save(model.state_dict(), weights_path)  # сохраняем состояние на диск
        log.info(f"Saved model to {weights_path}")  # выполняем шаг логики
    return stopped  # возвращаем значение

def main():  # объявляем функцию
    for seed in SEEDS:  # начинаем новый блок
        log.info(f"=== Training seed {seed} ===")  # выполняем шаг логики
        if train_one_seed(seed):  # начинаем новый блок
            break  # выполняем шаг логики


if __name__ == "__main__":  # точка входа при прямом запуске
    main()  # выполняем шаг логики
