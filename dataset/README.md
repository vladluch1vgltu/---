# Структура датасета (формат YOLO)

```
dataset/
├── images/
│   ├── train/    # Обучающие изображения
│   ├── val/      # Валидационные
│   └── test/     # Тестовые
└── labels/
    ├── train/    # Разметка (.txt на каждое изображение)
    ├── val/
    └── test/
```

## Формат разметки

Каждый файл `labels/<split>/<name>.txt` содержит строки:

```
<class_id> <x_center> <y_center> <width> <height>
```

Координаты нормализованы в диапазоне [0, 1].

## Классы (ID)

| ID | Класс |
|----|-------|
| 0 | tank |
| 1 | aircraft |
| 2 | helicopter |
| 3 | bridge |
| 4 | transport |
| 5 | tennis_court |
| 6 | swimming_pool |
| 7 | baseball_diamond |
| 8 | basketball_court |
| 9 | soccer_field |
| 10 | ground_track_field |
| 11 | roundabout |
| 12 | container_crane |

## Поддерживаемые датасеты

- xView, DOTA, SpaceNet, DIOR, HRSC2016

Конвертация DOTA — см. **[DOTA.md](DOTA.md)**:

```bash
python scripts/convert_dataset.py --format dota --input D:/data/DOTA-v2.0 --all-splits
```
