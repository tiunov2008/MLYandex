# Отчет по проекту (сводка артефактов и результатов)

Дата сборки: 2025-12-12  
Папка проекта: `C:\Users\Admin777\Desktop\Yandex\ML\Project`

## 1) Что лежит в репозитории

- Тренировка: `main.py` (универсальный тренер для выбранной модели из `config.py`).
- Конфиг путей/датасета: `config.py` (если папки `dataset/` нет — скачивает через `kagglehub`).
- Модели:
  - `models/simple_cnn/main.py` — SimpleCNN (вход 3 канала).
  - `models/simple_cnn_v2/main.py` — SimpleCNN v2 (вход 4 канала: RGB + high-pass).
  - `models/convnext_tiny/main.py` — ConvNeXtTiny (вход 3 канала).
- Утилиты:
  - `utils/data_utils.py` — чтение `train_solution.csv` и датасет `PosesDataset`.
  - `utils/transforms.py` — high-pass канал, оценка mean/std для него, нормализация.
  - `utils/train_utils.py` — сиды, эпоха обучения, F1.
  - `utils/model_utils.py` — сборка модели и загрузка весов.
  - `utils/plots.py` — графики обучения.
- Артефакты результатов:
  - `models/**/seed_*/training_history.csv` + `training_history.png` + `pose_cnn.pth` + `last_ckpt.pth`.
  - `archive/**` — сохраненные прогонки/эксперименты (часто дублируют метрики из `models/**`).
- Предсказания на тесте:
  - `predictions.csv` — CSV с колонками `id,label` (10000 строк).

## 2) Окружение

- Python: `3.13.5`
- PyTorch: `2.9.1+cu126`, CUDA доступна: `True`
- Зависимости: `requirements.txt`

## 3) Датасет (локальный `dataset/`)

Структура ожидается такая:

- `dataset/train_solution.csv`
- `dataset/train_images/*.jpg`
- `dataset/test_images/*.jpg`

Фактические числа (по текущему `dataset/`):

- `train_solution.csv`: 49999 строк
- `train_images`: 50000 jpg
- строк с существующими картинками: 49999 (пропусков по файлам: 0)
- распределение классов (train):
  - label 0: 41499 (≈ 83.0%)
  - label 1: 8500  (≈ 17.0%)
- `test_images`: 10000 jpg

Загрузка train-таблицы и фильтрация по существующим файлам: `utils/data_utils.py`.

## 4) Модели (что именно обучается)

### 4.1) `SimpleCNN` (v1, 3 канала)

Файл: `models/simple_cnn/main.py`.

- Вход: `N x 3 x H x W` (RGB).
- Backbone: 4 блока `Conv -> BN -> ReLU -> MaxPool`, каналы `3→32→64→128→256`.
- Head: GAP → Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→2).
- Параметров: `455682`.

### 4.2) `SimpleCNN v2` (4 канала)

Файл: `models/simple_cnn_v2/main.py`.

- Вход: `N x 4 x H x W` (RGB + high-pass канал).
- Backbone/Head такие же по смыслу, но первый блок принимает 4 канала, а `Conv2d(..., bias=False)` в блоках.
- Параметров: `455490`.

High-pass канал строится и добавляется в трансформах: `utils/transforms.py` (`AddHighPassChannel`).
Его mean/std оцениваются по train-таблице и кешируются:

- `models/simple_cnn_v2/hp_stats.json` (в текущей версии: mean=0.0034316, std=0.0380552, image_size=128)

### 4.3) `ConvNeXtTiny`

Файл: `models/convnext_tiny/main.py`.

- Вход: `N x 3 x H x W`.
- Параметров: `27821666`.

## 5) Пайплайн обучения (общий)

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
- Чекпоинт: `seed_xxx/last_ckpt.pth` сохраняется каждый epoch; лучшие веса для инференса — `seed_xxx/pose_cnn.pth`.
- Мягкая остановка по файлу-флагу: если существует `stop.txt`, тренировка сохранит чекпоинт и выйдет.

Важно про `simple_cnn_v2`: mean/std high-pass сейчас считаются по всему `df` до сплита (и train, и val попадают в оценку статистик).

## 6) Метрики прогонов (по `training_history.csv`)

Ниже сводка “best val_f1” по всем найденным `training_history.csv` в `models/` и `archive/`.
Формат: `epochs` — сколько эпох реально записано в history; `best_epoch` — где был максимум `val_f1`.

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

## 7) Предсказания (test)

Файл: `predictions.csv`.

- Строк: 10000
- Дубликаты `id`: 0
- `id` отсортированы по возрастанию: True
- Распределение предсказанных классов:
  - label 0: 8405
  - label 1: 1595

Если нужно, можно восстановить “какой именно моделью” получены предсказания только по косвенным признакам (какие веса были загружены, какой скрипт запускался).

## 8) Как воспроизвести

- Обучение выбранной модели: `python main.py` (модель задается в `config.py` через `MODEL_NAME`, сиды — в `main.py` через `SEEDS`).
- Ансамбль/инференс примером: `python ensemble_example.py` (по умолчанию пишет `ensemble_predictions.csv`).

## 9) Замечания

- `README.md` и некоторые `result.md` в консоли отображаются “кракозябрами” (похоже, файл не в UTF-8 или консоль читает не тем кодеком). Это не влияет на код, но мешает просмотру текста в терминале.

