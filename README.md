# SimpleCNN augmented (seed_123)
- База: тот же `SimpleCNN` (3 ConvBlock 32/64/128 → GAP → Linear 256 + Dropout 0.3 → Linear 2), чекпоинт `models/simple_cnn/seed_123/pose_cnn.pth`.
- Данные: `train_solution.csv` после фильтрации по наличию картинок в `dataset/train_images`, сплит `random_split` 80/20 с `SEED=123`, размер `image_size=128`, `batch_size=64`.
- Аугментации train: `RandomResizedCrop((128,128), scale=(0.8,1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`, `RandomAdjustSharpness(sharpness_factor=2, p=0.3)`, `ToTensor` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`; val/test: `Resize(128)` + `ToTensor` + `Normalize`.
- Обучение: CrossEntropy, Adam (`lr=1e-3`), ранняя остановка по val_f1 (patience=50), максимум 300 эпох; лучший снапшот на эпохе 240.
- Метрики лучшего чекпоинта (epoch 240): train_acc=0.9693, train_f1=0.9445, val_acc=0.9678, val_f1=0.9428, val_loss=0.0965.
- Лидборд: single-model сабмишен — **0.94360**.


# Отчет об обучении ансамбля
- База: `SimpleCNN` (3×ConvBlock 32/64/128 → GAP → Linear 256 + Dropout 0.3 → Linear 2), обучалась на `train_solution.csv` после фильтрации по наличию файлов в `dataset/train_images`.
- Сплит: `random_split` 80/20 с `SEED=123`, `image_size=128`, `batch_size=64`, ранняя остановка (patience=15), оптимизатор Adam (`lr=1e-3`), лосс CrossEntropy, метрика мониторинга — F1 macro.
- Аугментации train: `Resize(128)`, `RandomHorizontalFlip()`, `RandomRotation(15°)`, `ToTensor`, `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`. Val/test: только `Resize` + `ToTensor` + `Normalize`.
- Использованные чекпоинты для ансамбля: `models/simple_cnn/seed_76`, `seed_100`, `seed_123` (файлы `pose_cnn.pth`).
- Валидация на том же сплите (9999 образцов): soft voting F1=0.9678, acc=0.9822; hard voting F1=0.9678, acc=0.9804.
- Лидборд: soft ансамбль — 0.93749, hard — 0.93358.
- Seeds: `seed_76` — best val_f1 0.921 (epoch 193, val_acc 0.956); `seed_100` — best val_f1 0.915 (epoch 133, val_acc 0.953); `seed_123` — best val_f1 0.914 (epoch 133, val_acc 0.952).

# Отчет об обучении ансамбля 2
- База: `SimpleCNN` (те же веса из `archive/ensemble_2/seed_76`, `seed_100`, `seed_123`).
- Сплит: `random_split` 80/20, `image_size=128`, `batch_size=64`, оптимизатор Adam (`lr=1e-3`), ранняя остановка и отбор по лучшему val_f1.
- Аугментации train (отличаются от ансамбля 1): `RandomResizedCrop((128,128), scale=(0.8,1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`, `RandomAdjustSharpness(sharpness_factor=2, p=0.3)`, затем `ToTensor` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`. Val/test: `Resize(128)` + `ToTensor` + `Normalize`.
- Лучшие валид. метрики по сидaм: `seed_76` - val_f1 0.9376 (epoch 203, val_acc 0.9644); `seed_100` - val_f1 0.9386 (epoch 245, val_acc 0.9659); `seed_123` - val_f1 0.9428 (epoch 240, val_acc 0.9678).
- Лидборд: soft - 0.95197, hard - 0.95038.

