#!/usr/bin/env python3
"""
Конвертация DOTA (и др.) в формат YOLO для обучения.

Примеры:
    # Весь датасет (train + val) из корня DOTA-v2.0
    python scripts/convert_dataset.py --format dota --input D:/data/DOTA-v2.0 --all-splits

    # Один сплит
    python scripts/convert_dataset.py --format dota --input D:/data/DOTA-v2.0/train --split train
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Имена классов DOTA v1 / v1.5 / v2 -> классы проекта (без armored_vehicle и airport)
CLASS_MAP = {
    # Техника и транспорт
    "tank": "tank",
    "storage-tank": "tank",
    "storagetank": "tank",
    "plane": "aircraft",
    "aircraft": "aircraft",
    "helicopter": "helicopter",
    "helipad": "helicopter",
    "bridge": "bridge",
    "vehicle": "transport",
    "car": "transport",
    "truck": "transport",
    "small-vehicle": "transport",
    "smallvehicle": "transport",
    "large-vehicle": "transport",
    "largevehicle": "transport",
    "harbor": "transport",
    "ship": "transport",
    # Спорт и инфраструктура (ранее отфильтровывались)
    "baseball-diamond": "baseball_diamond",
    "baseballdiamond": "baseball_diamond",
    "tennis-court": "tennis_court",
    "tenniscourt": "tennis_court",
    "basketball-court": "basketball_court",
    "basketballcourt": "basketball_court",
    "ground-track-field": "ground_track_field",
    "groundtrackfield": "ground_track_field",
    "soccer-ball-field": "soccer_field",
    "soccerballfield": "soccer_field",
    "swimming-pool": "swimming_pool",
    "swimmingpool": "swimming_pool",
    "roundabout": "roundabout",
    "container-crane": "container_crane",
    "containercrane": "container_crane",
}

TARGET_CLASSES = [
    "tank",
    "aircraft",
    "helicopter",
    "bridge",
    "transport",
    "tennis_court",
    "swimming_pool",
    "baseball_diamond",
    "basketball_court",
    "soccer_field",
    "ground_track_field",
    "roundabout",
    "container_crane",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def parse_dota_label(line: str, img_w: int, img_h: int) -> list[float] | None:
    """DOTA: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty -> YOLO bbox."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if ":" in line and not line[0].replace(".", "").replace("-", "").isdigit():
        return None  # метаданные: imagesource:GoogleEarth, gsd:0.21
    parts = line.split()
    if len(parts) < 9:
        return None

    try:
        coords = [float(parts[i]) for i in range(8)]
    except ValueError:
        return None

    cls_name = normalize_class_name(parts[8])
    mapped = CLASS_MAP.get(cls_name)
    if mapped is None or mapped not in TARGET_CLASSES:
        return None

    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_max <= x_min or y_max <= y_min:
        return None

    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    if bw <= 0 or bh <= 0:
        return None

    cls_id = TARGET_CLASSES.index(mapped)
    return [cls_id, cx, cy, bw, bh]


def _label_dir_candidates(base: Path, split: str) -> list[Path]:
    """Все типичные имена папок с разметкой DOTA (зависят от версии и зеркала скачивания)."""
    return [
        base / "labelTxt",
        base / "labels",
        base / "labelTxt-v1.5",  # .txt прямо в папке (ваш Val)
        base / "val" / "labelTxt-v1.5",
        base / "labelTxt-v1.0" / "labelTxt",
        base / "labelTxt-v1.5" / "labelTxt",
        base / "labelTxt-v2.0" / "labelTxt",
        # DOTA-v2.0 переразметка (relabeled) — предпочтительно для v2
        base / "trainset_reclabelTxt" if split == "train" else base / "valset_reclabelTxt",
        base / f"{split}set_reclabelTxt",
        base / f"{split}_labelTxt",
        base / "ann",
        base / "Annotations",
    ]


def _image_dir_candidates(base: Path, split: str) -> list[Path]:
    return [
        base / "images",
        base,  # картинки прямо в train/ (P0000.png)
        base.parent / split / "images" if base.name != split else base / "images",
    ]


def resolve_dota_paths(input_dir: Path, split: str) -> tuple[Path, Path]:
    """
    Поиск папок images и разметки для DOTA.

    Разметка часто в отдельном архиве и называется не labelTxt, а
    trainset_reclabelTxt / labelTxt-v1.0/labelTxt и т.д.
    """
    bases = [input_dir]
    if input_dir.name != split and (input_dir / split).exists():
        bases.append(input_dir / split)
    if (input_dir.parent / split).exists():
        bases.append(input_dir.parent / split)

    pairs: list[tuple[Path, Path]] = []
    for base in bases:
        for img_dir in _image_dir_candidates(base, split):
            for lbl_dir in _label_dir_candidates(base, split):
                pairs.append((img_dir, lbl_dir))
        # Training/images + P*.txt в корне; Val/images + val/labelTxt-v1.5
        pairs.append((base / "images", base))
        pairs.append((input_dir / "images", input_dir / "val" / "labelTxt-v1.5"))
        pairs.append((input_dir / "images", input_dir / "labelTxt-v1.5"))
        # Разметка в соседней папке на уровне DOTA root: train/images + train_labelTxt/
        pairs.append((base / "images", input_dir.parent / f"{split}_labelTxt"))
        pairs.append((base / "images", input_dir.parent / f"DOTA-{split}-labelTxt"))

    seen: set[tuple[str, str]] = set()
    for img_dir, lbl_dir in pairs:
        key = (str(img_dir), str(lbl_dir))
        if key in seen:
            continue
        seen.add(key)
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        if not any(img_dir.glob("*.png")) and not any(img_dir.glob("*.jpg")):
            continue
        if not any(lbl_dir.glob("*.txt")):
            continue
        return img_dir, lbl_dir

    raise FileNotFoundError(
        f"Не найдены images + разметка для split='{split}' в {input_dir}.\n\n"
        "Частые причины:\n"
        "  1) Скачаны только картинки — нужен отдельный архив Annotations / labelTxt\n"
        "  2) Папка test — публичной разметки нет, используйте train и val\n"
        "  3) DOTA-v2: разметка в trainset_reclabelTxt / valset_reclabelTxt\n"
        "См. dataset/DOTA.md — раздел «Где взять разметку».\n"
        "Проверка: python scripts/check_dota.py --input <путь>"
    )


def convert_dota_split(input_dir: Path, output_dir: Path, split: str) -> dict[str, int]:
    import cv2

    img_dir, lbl_dir = resolve_dota_paths(input_dir, split)
    out_img = output_dir / "images" / split
    out_lbl = output_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "objects": 0, "skipped_no_label": 0, "skipped_no_class": 0}

    image_files = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img_path in sorted(image_files):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            stats["skipped_no_label"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines: list[str] = []
        raw_objects = 0
        with open(lbl_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_objects += 1
                bbox = parse_dota_label(line, w, h)
                if bbox is None:
                    continue
                yolo_lines.append(
                    f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}"
                )

        if not yolo_lines:
            stats["skipped_no_class"] += 1
            continue

        dest_img = out_img / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        with open(out_lbl / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        stats["images"] += 1
        stats["objects"] += len(yolo_lines)

    return stats


def convert_dota_all(input_dir: Path, output_dir: Path, splits: list[str]) -> None:
    total_images = 0
    total_objects = 0

    print(f"Вход:  {input_dir.resolve()}")
    print(f"Выход: {output_dir.resolve()}\n")

    for split in splits:
        split_input = input_dir
        # Если передан корень DOTA — для каждого split заходим в подпапку
        if (input_dir / split / "images").exists():
            split_input = input_dir / split

        try:
            stats = convert_dota_split(split_input, output_dir, split)
        except FileNotFoundError as e:
            print(f"[{split}] Пропуск: {e}")
            continue

        print(
            f"[{split}] изображений: {stats['images']}, "
            f"объектов: {stats['objects']}, "
            f"без разметки: {stats['skipped_no_label']}, "
            f"без целевых классов: {stats['skipped_no_class']}"
        )
        total_images += stats["images"]
        total_objects += stats["objects"]

    print(f"\nИтого: {total_images} изображений, {total_objects} объектов")
    print(f"Датасет готов: {output_dir / 'images'}")
    print("\nДальше: python train.py --data dataset.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация DOTA в формат YOLO")
    parser.add_argument("--format", choices=["dota"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Корень DOTA или папка split (train)")
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="train", help="train, val или test")
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Конвертировать train и val (и test, если есть)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Путь не найден: {input_dir}")

    if args.format == "dota":
        if args.all_splits:
            splits = []
            for s in ("train", "val", "test"):
                if (input_dir / s / "images").exists() or (input_dir / "images" / s).exists():
                    splits.append(s)
                elif s == args.split and (input_dir / "images").exists():
                    splits.append(s)
            if not splits:
                splits = ["train", "val"]
            convert_dota_all(input_dir, output_dir, splits)
        else:
            stats = convert_dota_split(input_dir, output_dir, args.split)
            print(f"Конвертировано: {stats['images']} изображений, {stats['objects']} объектов")


if __name__ == "__main__":
    main()
