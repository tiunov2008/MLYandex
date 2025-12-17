# Отчет по проекту

## Быстрый старт

- Установка: `pip install -r requirements.txt`
- Обучение: `python main.py` (модель/параметры в `config.py`)
- Инференс/ансамбль на test: `python ensemble.py` (пишет `ensemble_predictions.csv`)

## Что лежит в репозитории

- Тренировка: `main.py` (универсальный тренер для выбранной модели из `config.py`).
- Конфиг путей/датасета: `config.py` (если папки `dataset/` нет - скачивает через `kagglehub`).
- Модели:
  - `models/simple_cnn/main.py` - SimpleCNN (вход 3 канала).
  - `models/simple_cnn_v2/main.py` - SimpleCNN v2 (вход 4 канала: RGB + high-pass).
- Утилиты:
  - `utils/data_utils.py` - чтение `train_solution.csv` и датасет `PosesDataset`.
  - `utils/transforms.py` - high-pass канал, оценка mean/std для него, нормализация.
  - `utils/train_utils.py` - сиды, эпоха обучения, F1.
  - `utils/model_utils.py` - сборка модели и загрузка весов.
  - `utils/plots.py` - графики обучения.
- Артефакты результатов:
  - `models/**/seed_*/training_history.csv` + `training_history.png` + `pose_cnn.pth` + `last_ckpt.pth`.
  - `archive/**` - сохраненные прогонки/эксперименты (часто дублируют метрики из `models/**`).
- Предсказания на тесте:
  - `predictions.csv` - CSV с колонками `id,label` (10000 строк).

## Окружение

- Python: `3.13.5`
- PyTorch: `2.9.1+cu126`, CUDA доступна: `True`
- Зависимости: `requirements.txt`

## Датасет (локальный `dataset/`)

Ожидаемая структура:

- `dataset/train_solution.csv`
- `dataset/train_images/*.jpg`
- `dataset/test_images/*.jpg`

По текущему `dataset/`:

- `train_solution.csv`: 49999 строк
- `train_images`: 50000 jpg
- строк с существующими картинками: 49999 (пропусков по файлам: 0)
- распределение классов (train):
  - label 0: 41499 (83.0%)
  - label 1: 8500 (17.0%)
- `test_images`: 10000 jpg

Загрузка train-таблицы и фильтрация по существующим файлам: `utils/data_utils.py`.

## Модели (что именно обучается)

### `SimpleCNN` (v1, 3 канала)

Файл: `models/simple_cnn/main.py`.

- Вход: `N x 3 x H x W` (RGB).
- Backbone: 4 блока `Conv -> BN -> ReLU -> MaxPool`, каналы `3 -> 32 -> 64 -> 128 -> 256`.
- Head: GAP -> Linear(256->256) -> ReLU -> Dropout(0.3) -> Linear(256->2).
- Параметров: `455682`.

### `SimpleCNN v2` (4 канала)

Файл: `models/simple_cnn_v2/main.py`.

- Вход: `N x 4 x H x W` (RGB + high-pass канал).
- Backbone/Head такие же по смыслу, но первый блок принимает 4 канала, а `Conv2d(..., bias=False)` в блоках.
- Параметров: `455490`.

High-pass канал строится и добавляется в трансформах: `utils/transforms.py` (`AddHighPassChannel`).
Его mean/std оцениваются по train-таблице и кешируются:

- `models/simple_cnn_v2/hp_stats.json` (в текущей версии: mean=0.0034316, std=0.0380552, image_size=128)


## Пайплайн обучения (общий)

Тренер: `main.py`.

- Сплит: `random_split` 80/20 (seed задается через `torch.Generator().manual_seed(seed)`).
- Размер: `image_size=128`.
- Batch size: `64`.
- Аугментации:
  - train: `RandomResizedCrop(scale=(0.9, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(15)`.
  - val: `Resize(128)`.
- Нормализация:
  - RGB: ImageNet mean/std (`utils/transforms.py`).
  - Для `simple_cnn_v2` добавляется 4-й канал и расширяется mean/std.
- Оптимизатор: Adam `lr=1e-3`.
- Loss: CrossEntropy.
- Метрики: accuracy и macro-F1 (`torchmetrics.F1Score`).
- Ранняя остановка: по отсутствию улучшения `val_f1` в течение `EARLY_STOP_PATIENCE=50` эпох.
- Чекпоинт: `seed_xxx/last_ckpt.pth` сохраняется каждый epoch; лучшие веса для инференса - `seed_xxx/pose_cnn.pth`.
- Мягкая остановка по файлу-флагу: если существует `stop.txt`, тренировка сохранит чекпоинт и выйдет.

Важно про `simple_cnn_v2`: mean/std high-pass сейчас считаются по всему `df` до сплита (и train, и val попадают в оценку статистик).

## Эксперименты и результаты

### SimpleCNN augmented (seed_123)

- База: тот же `SimpleCNN` (3x ConvBlock 32/64/128 -> GAP -> Linear 256 + Dropout 0.3 -> Linear 2), чекпоинт `models/simple_cnn/seed_123/pose_cnn.pth`.
- Данные: `train_solution.csv` после фильтрации по наличию картинок в `dataset/train_images`, сплит `random_split` 80/20 с `SEED=123`, `image_size=128`, `batch_size=64`.
- Аугментации train: `RandomResizedCrop((128,128), scale=(0.8,1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`, `RandomAdjustSharpness(sharpness_factor=2, p=0.3)`, `ToTensor` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`; val/test: `Resize(128)` + `ToTensor` + `Normalize`.
- Обучение: CrossEntropy, Adam (`lr=1e-3`), ранняя остановка по val_f1 (patience=50), максимум 300 эпох; лучший снапшот на эпохе 240.
- Метрики лучшего чекпоинта (epoch 240): train_acc=0.9693, train_f1=0.9445, val_acc=0.9678, val_f1=0.9428, val_loss=0.0965.
- Лидборд: single-model сабмишен - **0.94360**.

### Ансамбль 1 (SimpleCNN)

- База: `SimpleCNN` (3x ConvBlock 32/64/128 -> GAP -> Linear 256 + Dropout 0.3 -> Linear 2).
- Сплит: `random_split` 80/20 с `SEED=123`, `image_size=128`, `batch_size=64`, ранняя остановка (patience=15), оптимизатор Adam (`lr=1e-3`), лосс CrossEntropy, метрика мониторинга - F1 macro.
- Аугментации train: `Resize(128)`, `RandomHorizontalFlip()`, `RandomRotation(15 deg)`, `ToTensor`, `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`. Val/test: только `Resize` + `ToTensor` + `Normalize`.
- Чекпоинты: `models/simple_cnn/seed_76`, `seed_100`, `seed_123` (файлы `pose_cnn.pth`).
- Валидация на том же сплите (9999 образцов): soft voting F1=0.9678, acc=0.9822; hard voting F1=0.9678, acc=0.9804.
- Лидборд: soft ансамбль - 0.93749, hard - 0.93358.
- Seeds: `seed_76` - best val_f1 0.921 (epoch 193, val_acc 0.956); `seed_100` - best val_f1 0.915 (epoch 133, val_acc 0.953); `seed_123` - best val_f1 0.914 (epoch 133, val_acc 0.952).

### Ансамбль 2 (SimpleCNN)

- База: `SimpleCNN` (веса из `archive/ensemble_2/seed_76`, `seed_100`, `seed_123`).
- Сплит: `random_split` 80/20, `image_size=128`, `batch_size=64`, оптимизатор Adam (`lr=1e-3`), ранняя остановка и отбор по лучшему val_f1.
- Аугментации train (отличаются от ансамбля 1): `RandomResizedCrop((128,128), scale=(0.8,1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`, `RandomAdjustSharpness(sharpness_factor=2, p=0.3)`, затем `ToTensor` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`. Val/test: `Resize(128)` + `ToTensor` + `Normalize`.
- Лучшие валид. метрики по сидам: `seed_76` - val_f1 0.9376 (epoch 203, val_acc 0.9644); `seed_100` - val_f1 0.9386 (epoch 245, val_acc 0.9659); `seed_123` - val_f1 0.9428 (epoch 240, val_acc 0.9678).
- Лидборд: soft - 0.95197, hard - 0.95038.

## Сводка прогонов (по `training_history.csv`)

Ниже сводка "best val_f1" по всем найденным `training_history.csv` в `models/` и `archive/`.
Формат: `epochs` - сколько эпох реально записано в history; `best_epoch` - где был максимум `val_f1`.

| Группа (папка) | Seed | epochs | best_val_f1 | best_epoch | last_val_f1 |
|---|---:|---:|---:|---:|---:|
| `models/simple_cnn_v2` | 100 | 180 | 0.952854 | 160 | 0.949532 |
| `models/simple_cnn_v2` | 101 | 150 | 0.951362 | 129 | 0.938868 |
| `models/simple_cnn_v2` | 102 | 188 | 0.957075 | 168 | 0.948225 |
| `models/simple_cnn_v2` | 103 | 181 | 0.953491 | 161 | 0.948767 |
| `models/simple_cnn_v2` | 104 | 192 | 0.958306 | 192 | 0.958306 |
| `archive/simple_cnn_v2` | 100 | 180 | 0.952854 | 160 | 0.949532 |
| `archive/simple_cnn_v2` | 101 | 150 | 0.951362 | 129 | 0.938868 |
| `archive/simple_cnn_v2` | 102 | 188 | 0.957075 | 168 | 0.948225 |
| `archive/simple_cnn_v2` | 103 | 181 | 0.953491 | 161 | 0.948767 |
| `archive/simple_cnn_v2` | 104 | 192 | 0.958306 | 192 | 0.958306 |
| `archive/ensemble_1` | 76 | 205 | 0.921257 | 193 | 0.915005 |
| `archive/ensemble_1` | 100 | 148 | 0.914943 | 133 | 0.911241 |
| `archive/ensemble_1` | 123 | 167 | 0.921880 | 152 | 0.904910 |
| `archive/ensemble_2` | 76 | 223 | 0.937592 | 203 | 0.927412 |
| `archive/ensemble_2` | 100 | 265 | 0.938555 | 245 | 0.913173 |
| `archive/ensemble_2` | 123 | 260 | 0.942760 | 240 | 0.935475 |
| `archive/ensemble_3` | 76 | 223 | 0.937592 | 203 | 0.927412 |
| `archive/ensemble_3` | 100 | 265 | 0.938555 | 245 | 0.913173 |
| `archive/ensemble_3` | 101 | 157 | 0.928942 | 137 | 0.924135 |
| `archive/ensemble_3` | 102 | 218 | 0.933440 | 216 | 0.928115 |
| `archive/ensemble_3` | 123 | 260 | 0.942760 | 240 | 0.935475 |

Лучший зафиксированный `best_val_f1` среди всех найденных history: **0.958306** (SimpleCNN v2, seed 104).

## Предсказания (test)

Файл: `predictions.csv`.

- Строк: 10000
- Дубликаты `id`: 0
- `id` отсортированы по возрастанию: True
- Распределение предсказанных классов:
  - label 0: 8405
  - label 1: 1595
