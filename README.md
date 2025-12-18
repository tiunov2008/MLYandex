# Проект: классификация поз (ML Intensive Yandex)

## Быстрый старт

- Установка: `pip install -r requirements.txt`
- Обучение: `python main.py`
- Инференс/ансамбль на test: `python ensemble.py` (пишет `predictions.csv`)

## Датасет

Если в корне проекта есть папка `dataset/`, ожидается структура:

- `dataset/train_solution.csv`
- `dataset/train_images/*.jpg`
- `dataset/test_images/*.jpg`

Если `dataset/` отсутствует, `config.py` при первом импорте попробует скачать датасет через `kagglehub`.

## Как воспроизвести все обучения

В `archive.py` лежат "замороженные" конфиги запусков (модель + пресет аугментаций + seeds + ключевые гиперпараметры).
Список доступных прогонов - в словаре `RUNS` внутри `archive.py`.

Запуск прогона:

`python archive.py run <run_name>`

Пример:

`python archive.py run ensemble_2`

Чтобы не перетирать существующие артефакты, укажите другой вывод:

`python archive.py run ensemble_2 --output-root runs/repro --run-name ensemble_2`

### Ансамбль/инференс на test (`ensemble.py`)

После обучения (или используя уже сохраненные веса) можно получить предсказания на test и/или сделать ансамбль:

`python ensemble.py`

По умолчанию `ensemble.py` собирает soft-voting ансамбль и сохраняет `predictions.csv`. Настройки (voting/out_path/список моделей) задаются прямо внизу `ensemble.py`.

## CLI обучения (`main.py`)

`main.py` принимает:

- seeds: `--seeds 76 100 123` (или один `--seed 76`)
- модель: `--model simple_cnn|simple_cnn_v2`
- аугментации: `--augmentation none|baseline|ensemble_1|ensemble_2`
- вывод артефактов: `--output-root <path>` и `--run-name <name>`

## Где лежат артефакты

Для каждого seed создается папка `.../seed_<seed>/` со стандартным набором файлов:

- `training_history.csv` и `training_history.png`
- `last_ckpt.pth` (чекпоинт для продолжения)
- `pose_cnn.pth` (лучшие веса по `val_f1`)

## Модели

- `simple_cnn` - простая CNN для RGB (3 канала): несколько Conv-блоков + GAP + MLP-голова.
- `simple_cnn_v2` - та же идея, но вход 4 канала (RGB + high-pass канал из `utils/transforms.py`).

Графики обучения и веса можно посмотреть в `archive/` (исторически сохраненные прогоны; и это же значение по умолчанию для `archive.py run`, если не переопределять `--output-root`).

## Какие модели используются в ансамблях

- `ensemble_1` - `simple_cnn`, seeds: `76, 100, 123`.
- `ensemble_2` - `simple_cnn`, seeds: `76, 100, 123`.
- `ensemble_3` - `simple_cnn`, seeds: `76, 100, 101, 102, 103, 123`.
- Ансамбль в `ensemble.py` - по умолчанию `simple_cnn_v2` (берет веса из `models/simple_cnn_v2`).

## Примечания по воспроизводимости

- Даже при фиксированных seed возможны небольшие расхождения метрик из-за CUDA/версий PyTorch/недетерминизма отдельных операций.
- Файл `stop.txt` в корне проекта - мягкая остановка: тренировка сохранит чекпоинт и завершится.
