# Итоги эксперимента: SimpleCNN

Модель: `models/simple_cnn/main.py` (SimpleCNN с 3 ConvBlock и GAP-классификатором).

Обучение:
- Датасет: `train_images`, разметка `train_solution.csv`, входной размер 128x128.
- Трансформы: resize, random horizontal flip, random rotation, нормализация (ImageNet).
- Оптимизатор: Adam, lr=1e-3, критерий: CrossEntropyLoss.
- Метрики: accuracy, macro-F1.
- Чекпоинт: `models/simple_cnn/last_ckpt.pth` (последняя эпоха на момент остановки).
- Финальные веса для инференса: `models/simple_cnn/pose_cnn.pth` (можно пересохранить из last_ckpt при необходимости).

Результаты:
- Валид. качество на последней эпохе: макро-F1 ≈ 0.937 (train F1 ≈ 0.985).
- Финальный score на лидерборде: **0.93990**.

Инференс:
- Скрипт: `predict.py` — использует `last_ckpt.pth` (падает на `pose_cnn.pth`, если чекпоинта нет).
- Вывод: `predictions.csv` с колонками `id,label`.
