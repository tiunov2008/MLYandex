# Построчные комментарии для разбора кода (учебная разметка).
# Комментарии добавлены автоматически; логика файла не менялась.

import os  # доступ к переменным окружения и простым системным утилитам
from pathlib import Path  # удобная и кроссплатформенная работа с путями

from loguru import logger as log  # логгер (красивые сообщения + удобный API)
import kagglehub  # загрузка датасетов с Kaggle в локальный кэш

DATASET_HANDLE = "zv3zdochka/ml-intensive-yandex-academy-autumn-2025"  # идентификатор датасета в KaggleHub
_DATASET_LOGGED_FLAG = "DATASET_LOGGED"  # имя env-флага: чтобы не повторять один и тот же лог
_LOGGED = False  # локальный флаг: уже логировали сообщение в этом процессе


def get_dataset_dir() -> Path:  # выбираем директорию датасета (локально или скачиваем)
    local_path = Path("dataset")  # ожидаем папку `dataset/` рядом с проектом

    if local_path.exists():  # если датасет уже лежит рядом с проектом
        _maybe_log_once(f"Using local dataset at {local_path.resolve()}")  # сообщаем абсолютный путь (один раз)
        return local_path  # используем локальную директорию без скачивания

    _maybe_log_once("Local dataset not found. Downloading from Kaggle...")  # предупреждаем, что пойдём в сеть/кэш
    base_path = Path(kagglehub.dataset_download(DATASET_HANDLE))  # KaggleHub возвращает путь к скачанным данным

    dataset_path = base_path / "dataset"  # иногда нужная папка лежит внутри base_path/dataset
    if dataset_path.exists():  # если структура именно такая
        return dataset_path  # возвращаем путь на вложенную папку `dataset/`

    return base_path  # иначе считаем base_path корнем датасета


def _maybe_log_once(message: str) -> None:  # логируем сообщение только один раз (в текущем процессе)
    global _LOGGED  # объявляем, что хотим изменять модульную переменную
    if _LOGGED:  # если уже логировали
        return  # выходим без повторного log.info(...)
    if os.environ.get(_DATASET_LOGGED_FLAG) == "1":  # если env-флаг уже выставлен (в рамках процесса)
        _LOGGED = True  # синхронизируем локальный флаг
        return  # и выходим
    log.info(message)  # печатаем сообщение в лог
    os.environ[_DATASET_LOGGED_FLAG] = "1"  # ставим env-флаг, чтобы не повторять лог ниже по коду
    _LOGGED = True  # отмечаем локально, что лог уже был


MODEL_NAME = "simple_cnn"  # имя выбранной архитектуры/папки для артефактов
DATASET_DIR = get_dataset_dir()  # базовая директория датасета (локальная или скачанная)
CSV_PATH = DATASET_DIR / "train_solution.csv"  # путь к разметке/таргетам
TRAIN_DIR = DATASET_DIR / "train_images"  # путь к изображениям обучения
MODELS_ROOT = Path("models")  # корневая папка для моделей/весов
MODEL_DIR = MODELS_ROOT / MODEL_NAME  # папка конкретной модели (по имени)
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # гарантируем, что папка модели существует
HISTORY_PATH = MODEL_DIR / "training_history.csv"  # куда сохранять историю обучения (таблица)
HISTORY_PLOT_PATH = MODEL_DIR / "training_history.png"  # куда сохранять графики history
HP_STATS_PATH = MODEL_DIR / "hp_stats.json"  # куда сохранять статистики high-pass (для v2)
