# Построчные комментарии для разбора кода (учебная разметка).
# Комментарии добавлены автоматически; логика файла не менялась.

from torch import nn  # импортируем torch.nn (слои/модули нейросетей)


class ConvBlock(nn.Module):  # сверточный блок: Conv -> BN -> ReLU -> Pool
    """Conv -> BatchNorm -> ReLU -> MaxPool."""  # краткое описание пайплайна блока

    def __init__(self, in_channels, out_channels):  # инициализируем блок по числу каналов
        super().__init__()  # инициализируем базовый класс nn.Module
        self.conv = nn.Conv2d(  # свёртка 2D по изображению/фичам
            in_channels,  # число входных каналов
            out_channels,  # число выходных каналов
            kernel_size=3,  # ядро 3x3
            padding=1,  # padding=1 сохраняет spatial-размер при stride=1
            stride=1,  # шаг свёртки
            bias=False,  # bias отключаем, т.к. далее идёт BatchNorm
        )  # конец конфигурации Conv2d
        self.norm = nn.BatchNorm2d(out_channels)  # батч-нормализация по каналам
        self.relu = nn.ReLU(inplace=True)  # нелинейность ReLU (inplace экономит память)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # даунсэмплинг в 2 раза по H/W

    def forward(self, x):  # прямой проход: (B, C, H, W) -> (B, C', H/2, W/2)
        x = self.conv(x)  # свёртка: извлекаем локальные признаки
        x = self.norm(x)  # нормализуем активации
        x = self.relu(x)  # применяем нелинейность
        x = self.pool(x)  # уменьшаем spatial-разрешение
        return x  # возвращаем тензор признаков


class SimpleCNN(nn.Module):  # простая CNN для классификации (v2: + high-pass канал)
    def __init__(self, num_classes=2):  # задаём число классов для последнего слоя
        super().__init__()  # инициализируем базовый класс nn.Module
        self.features = nn.Sequential(  # «тело» сети: последовательность ConvBlock
            ConvBlock(4, 32),  # вход: 4 канала (RGB + high-pass) -> 32 канала
            ConvBlock(32, 64),  # 32 -> 64
            ConvBlock(64, 128),  # 64 -> 128
            ConvBlock(128, 256),  # 128 -> 256
        )  # конец блока features
        self.classifier = nn.Sequential(  # «голова» классификатора
            nn.AdaptiveAvgPool2d(1),  # усредняем по spatial до 1x1
            nn.Flatten(),  # (B, 256, 1, 1) -> (B, 256)
            nn.Linear(256, 256),  # полносвязный слой для смешивания признаков
            nn.ReLU(inplace=True),  # нелинейность
            nn.Dropout(0.3),  # регуляризация
            nn.Linear(256, num_classes),  # выход: логиты по классам
        )  # конец блока classifier

    def forward(self, x):  # прямой проход всей модели
        x_feat = self.features(x)  # извлекаем пространственные признаки CNN
        out = self.classifier(x_feat)  # преобразуем признаки в логиты классов
        return out  # возвращаем логиты
