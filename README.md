# README HW2 ветки
ФИО: Ярмухаметов Юрий Алексеевич

Предмет: Обработка и генерация изображений

Задача: Детекция аномалий

[Ноутбук с кодом](notebook_anomaly_detection.ipynb)

## Задача

На первый взгляд задача показалась простой:

1) Создаем пайплайн для загрузки данных (стоит отметить, что данные для теста отличаются от тренировочных, поэтому я добавлял в пайплайн аугментации для тренировочного сета)
2) Обучаем авттоэнкодер
3) Через обученный энкодер прогоняем данные с проливами металла (аномалиями), считаем лосс - MSE и по нему определяем порог для классификации
4) Загружаем тестовую часть без аугментаций, считаем лосс и по трешхолду делаем классификацию
5) Получаем fpr, tpr > 0.9
6) Радуемся :)

Но что-то пошло не так - я не смог достичь требуемых значений по метрикам

## Что было сделано

### Аугментации и предобработка данных

```
target_size = 32
preprocess_augm = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.RandomRotation(1),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.01, contrast=0.01),
    transforms.ToTensor(),
    transforms.Normalize(mean=0, std=1),
])
```
Выстроил вот такой пайплайн: ресайз каждой картинки до 32х32 (пробовал еще 64, 48 и 50 - особо не заметил различий), различные повороты и изменение яркости, контрасности и наконец - нормализация. Пробовал менять значения пр каждому из пунктов (кроме нормализации) 


### Модели

В процессе работы пробовал 3 разные архитектуры (на самом деле вначале пробовал еще две, но для них лосс на трейне и валидации был уж очень большой и никак не сходился):

```
lass Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.conv10 = nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.upsample(x)
        x = self.conv8(x)
        x = self.upsample(x)
        x = self.conv9(x)
        x = self.upsample(x)
        x = self.sigmoid(self.conv10(x))
        return x
```

```
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Bottleneck layer
            nn.Conv2d(64, 16, kernel_size=1, padding=0),  # Reduced channels
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample to 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample to 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

```
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
Эксперименты с этими архитектурами есть в ноутбуке. Так же я пробовал менять lr и optimizer, но это никак не меняло результат, поэтому зафиксировал lr = 0.0004 и optimizer = AdamW (пробовал другие вариации Adam, ASGD). 

Еще добавил планировщик и early stop с ожиданием в 10 эпох.

## Подбор threshold

После прохода обученным энкодером по картинкам с проливом я пробовал брать минимальное значение лосса, среднее и максимальное, от каждого пробовал вручную крутить порог - добавлял и убавлял константы и смотрел на значения tpr, fpr. Даже таким способом не смог достичь метрик (это не стал добавлять в ноутбук)

По итогу лучший результат был достингут с первой моделью и порогом равынм минимальному лоссу на проливе (хотя в какой-то момент вручную удалось докуртить значения до 0.92 и 0.83:


![image](https://github.com/arrogantcont/processing_and_generating_images_course/assets/59160824/c2e7a80e-687b-4d96-afc9-e2a8e511e73e)

## Что можно исправить, чтобы достичь желаемых значений TPR и FPR:

1) добавить динамическую аугментациюю - это должно улучшить обобщающую способность модели и помочь правильно классифицировать тестовые данные, которые немного отличаются от тех, что есть в трейне
2) Изменить архитектуру bottlenck
3) Использовать больше фильтров в свертках 




