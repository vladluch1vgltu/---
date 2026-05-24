# Подключение датасета DOTA

Пошаговая инструкция для **DOTA v1.0 / v1.5 / v2.0**.

## 1. Скачивание

1. Откройте: https://captain-whu.github.io/DOTA/dataset.html  
2. Зарегистрируйтесь и скачайте **несколько архивов** — картинки и разметка идут **разными ссылками**.

### Почему нет папки `labelTxt`?

Часто скачивают только **Images** (train / val / test) — тогда видны лишь:

```text
DOTA/
├── train/    ← только .png
├── val/
└── test/   ← разметки официально НЕТ (соревнование)
```

**Разметка — отдельный ZIP.** На сайте ищите ссылки вроде:

| Версия | Что скачать кроме картинок |
|--------|----------------------------|
| DOTA-v1.0 | `Annotations` / `labelTxt` для train и val |
| DOTA-v2.0 | Доп. пакет **annotations of DOTA-v2.0**; внутри часто `trainset_reclabelTxt`, `valset_reclabelTxt` |
| Любая | Архив с именем `labelTxt`, `*_ann`, `reclabel` |

Для **DOTA-v2.0** на сайте указано: сначала images v1.0, затем **отдельно** extra images + **annotations v2.0** (код распаковки на странице).

### Нормальные варианты структуры после полной загрузки

**Вариант A** (классический):

```text
train/images/  +  train/labelTxt/
val/images/    +  val/labelTxt/
```

**Вариант B** (DOTA-v2 relabel):

```text
train/images/  +  train/trainset_reclabelTxt/
val/images/    +  val/valset_reclabelTxt/
```

**Вариант C** (вложенные папки с зеркал Baidu/Google Drive):

```text
train/labelTxt-v1.0/labelTxt/*.txt
```

**Папка `test`** — только изображения для отправки на сервер оценки; **публичных `.txt` для test нет**. Обучение: только **train + val**.

### Ваш случай: папка `DOTA\Training`

После распаковки часто получается **не** `labelTxt/`, а так:

```text
DOTA\Training\
├── images\          ← 942 снимка P0000.png ...
├── P0000.txt        ← разметка в корне (или после распаковки labelTxt.zip)
├── P0001.txt
├── labelTxt.zip     ← можно не трогать, если .txt уже распакованы
└── train\           ← часто пустая, не используется
```

Конвертация в проект:

```powershell
cd "d:\Студент\Магистратура\Дисертация\Код"
python scripts/convert_dataset.py --format dota --input "D:\Студент\Магистратура\Дисертация\DOTA\Training" --split train
```

Для **валидации** скачайте отдельно *Validation* с сайта DOTA (images + txt) и положите в `DOTA\Validation`, затем:

```powershell
python scripts/convert_dataset.py --format dota --input "D:\Студент\Магистратура\Дисертация\DOTA\Validation" --split val
```

### Проверка, что разметка есть

```powershell
python scripts/check_dota.py --input "D:\Студент\Магистратура\Дисертация\DOTA\Training"
```

Должна появиться папка с сотнями/тысячами `.txt` рядом с images.

## 2. Конвертация в формат YOLO

Из корня проекта (`Код/`):

```powershell
cd "d:\Студент\Магистратура\Дисертация\Код"

# Весь train + val одной командой
python scripts/convert_dataset.py --format dota --input "D:/data/DOTA-v2.0" --all-splits
```

Либо по отдельности:

```powershell
python scripts/convert_dataset.py --format dota --input "D:/data/DOTA-v2.0/train" --split train
python scripts/convert_dataset.py --format dota --input "D:/data/DOTA-v2.0/val" --split val
```

После конвертации:

```text
dataset/
├── images/train/   ← копии снимков
├── images/val/
├── labels/train/   ← YOLO .txt (class cx cy w h)
└── labels/val/
```

## 3. Соответствие классов DOTA → проект

| Класс в DOTA | Класс в системе |
|--------------|-----------------|
| plane | aircraft |
| helicopter, helipad | helicopter |
| bridge | bridge |
| airport | airport |
| small-vehicle, large-vehicle, ship, harbor | transport |
| storage-tank | tank |
| armored-vehicle (если есть в разметке) | armored_vehicle |

Остальные классы DOTA (tennis-court, swimming-pool, ship как отдельный тип и т.д.) **пропускаются** — в датасете остаются только 7 целевых классов из ТЗ.

> В DOTA мало меток именно «tank» / «armored_vehicle» — основная масса объектов: самолёты, корабли, машины, мосты. Для диссертации это нормально: модель учится на доступных классах; при необходимости добавьте DIOR/xView с танками.

## 4. Проверка

```powershell
# Сколько файлов
(Get-ChildItem dataset\images\train).Count
(Get-ChildItem dataset\labels\train).Count

# Пример разметки
Get-Content dataset\labels\train\P0000.txt -Head 5
```

Строка YOLO выглядит так: `2 0.512 0.341 0.08 0.12` (класс, центр, ширина, высота — всё от 0 до 1).

## 5. Обучение

`dataset.yaml` уже настроен на папку `dataset/`:

```powershell
python train.py --data dataset.yaml --epochs 100 --device cuda
```

При нехватке VRAM уменьшите batch в `configs/default.yaml`:

```yaml
training:
  batch_size: 8   # было 16
```

## 6. Оценка и детекция

```powershell
python evaluate.py --weights models/best.pt --data dataset.yaml --split val
python detect.py --source dataset/images/val --batch --output results
```

## 7. Частые проблемы

| Проблема | Решение |
|----------|---------|
| `0 изображений` после конвертации | Проверьте путь: должны быть `.../train/images` и `.../train/labelTxt` |
| Нет `labelTxt` | Скачайте **Annotations** с сайта DOTA; ищите `trainset_reclabelTxt` / `valset_reclabelTxt` |
| Есть только train/val/test с png | Это только images — нужен второй архив с разметкой |
| Очень мало объектов | Нормально: фильтр оставляет только 7 классов ТЗ |
| CUDA out of memory | `batch_size: 4`, модель `model.size: "s"` в configs/default.yaml |
| Медленный инференс на больших снимках | `python detect.py --source img.tif --tiling` |

## 8. DOTA test (без публичной разметки)

Папку `test/images` можно конвертировать с `--split test` для инференса, но **обучать** на test нельзя — публичных labels нет. Используйте только `train` + `val`.
