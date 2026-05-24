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
| 1 | armored_vehicle |
| 2 | aircraft |
| 3 | helicopter |
| 4 | airport |
| 5 | bridge |
| 6 | transport |

## Поддерживаемые датасеты

- xView, DOTA, SpaceNet, DIOR, HRSC2016

Конвертация DOTA — см. **[DOTA.md](DOTA.md)**:

```bash
python scripts/convert_dataset.py --format dota --input D:/data/DOTA-v2.0 --all-splits
```
