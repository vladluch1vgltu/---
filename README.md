# Интеллектуальная система распознавания объектов на спутниковых изображениях

Система автоматического обнаружения, локализации и классификации объектов на спутниковых снимках на базе **YOLOv8** (Ultralytics) и **PyTorch**.

## Возможности

- Обнаружение 13 классов (техника, транспорт, спорт и инфраструктура DOTA — см. `dataset.yaml`)
- Предобработка: resize, нормализация, шумоподавление, коррекция яркости/контраста
- Аугментация данных при обучении
- Обработка изображений высокого разрешения (tiling)
- Пакетный инференс
- Экспорт результатов: JPG, JSON, CSV, TXT
- Метрики: Precision, Recall, F1, mAP50, mAP50-95
- REST API и веб-интерфейс
- Экспорт модели в ONNX
- GPU / CUDA

## Структура проекта

```
project/
├── dataset/          # Данные (формат YOLO)
├── models/           # Веса и checkpoint
├── train/            # Модуль обучения
├── inference/        # Модуль инференса
├── preprocess/       # Предобработка
├── evaluate/         # Метрики
├── results/          # Результаты детекции
├── configs/          # Конфигурация
├── api/              # REST API
├── web/              # Веб-интерфейс
├── scripts/          # Утилиты
├── train.py
├── detect.py
├── evaluate.py
├── preprocess.py
├── run_api.py
└── dataset.yaml
```

## Установка

```bash
cd "d:\Студент\Магистратура\Дисертация\Код"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Требуется **Python 3.11+**. Для GPU установите PyTorch с поддержкой CUDA с [pytorch.org](https://pytorch.org).

## Подготовка датасета

### DOTA (рекомендуется)

Подробная инструкция: **[dataset/DOTA.md](dataset/DOTA.md)**

```powershell
# 1. Скачать DOTA-v2.0 с https://captain-whu.github.io/DOTA/dataset.html
# 2. Распаковать в D:\data\DOTA-v2.0 (должны быть train/images, train/labelTxt, val/...)

python scripts/convert_dataset.py --format dota --input "D:/data/DOTA-v2.0" --all-splits
python train.py --data dataset.yaml --epochs 100 --device cuda
```

### Свой датасет

Разместите изображения и разметку YOLO в `dataset/images/{train,val}` и `dataset/labels/{train,val}`.

## Обучение

```bash
python train.py --data dataset.yaml --epochs 100 --device cuda
python train.py --resume
python train.py --export-onnx
```

Параметры задаются в `configs/default.yaml` (transfer learning, early stopping, augmentation).

## Инференс

```bash
python detect.py --source image.tif --output results
python detect.py --source dataset/images/test --batch
python detect.py --source large.tif --tiling
```

## Оценка качества

```bash
python evaluate.py --weights models/best.pt --data dataset.yaml
```

Минимальные требования ТЗ: mAP50 ≥ 0.75, Precision ≥ 0.8, Recall ≥ 0.75 (достигаются после обучения на размеченных данных).

## Предобработка

```bash
python preprocess.py --input dataset/raw --output dataset/processed
python preprocess.py --input image.tif --preview
```

## REST API и веб-интерфейс

```bash
python run_api.py
```

- Документация API: http://localhost:8000/docs
- Веб-интерфейс: http://localhost:8000/ui/index.html

Пример запроса:

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
```

## Конфигурация

Основные параметры — в `configs/default.yaml`:

| Параметр | Описание |
|----------|----------|
| `model.size` | Размер YOLO (n/s/m/l/x) |
| `detection.conf_threshold` | Порог confidence |
| `detection.iou_threshold` | Порог IoU для NMS |
| `preprocessing.tile_size` | Размер тайла для больших снимков |
| `security.max_file_size_mb` | Лимит загрузки |

Добавление нового класса: обновите `classes` в `configs/default.yaml` и `names` в `dataset.yaml`, затем переобучите модель.

## Соответствие ТЗ

| Требование | Реализация |
|------------|------------|
| YOLO (Backbone + Neck + Head) | YOLOv8 через Ultralytics |
| 7 классов + расширение | `configs/default.yaml`, `dataset.yaml` |
| JPG, PNG, TIFF | `preprocess/pipeline.py` |
| Высокое разрешение | Tiling в `inference/detector.py` |
| NMS, confidence/IoU фильтры | Ultralytics + межтайловый NMS |
| Визуализация + JSON/CSV/TXT | `SatelliteDetector._save_outputs` |
| Метрики mAP, P, R, F1 | `evaluate/metrics.py` |
| Transfer learning, checkpoints | `train/trainer.py` |
| ONNX export | `train.py --export-onnx` |
| REST API + Web UI | `api/app.py`, `web/index.html` |
| Безопасность входных файлов | `utils/security.py` |

## Описание для диссертации

Текст для пояснительной записки (этапы алгоритма, метрики, эпохи, параметры): **[docs/ОПИСАНИЕ_ПРОГРАММЫ.md](docs/ОПИСАНИЕ_ПРОГРАММЫ.md)**

## Лицензия

Учебно-исследовательский проект (магистерская диссертация).
