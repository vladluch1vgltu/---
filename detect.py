#!/usr/bin/env python3
"""
Инференс: обнаружение объектов на спутниковых изображениях.

Использование:
    python detect.py --source image.tif
    python detect.py --source dataset/images/test --batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.detector import SatelliteDetector
from utils.security import InputValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Распознавание объектов на спутниковых изображениях")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Путь к весам модели")
    parser.add_argument("--source", type=str, required=True, help="Изображение или директория")
    parser.add_argument("--output", type=str, default="results", help="Директория для результатов")
    parser.add_argument("--batch", action="store_true", help="Пакетная обработка директории")
    parser.add_argument("--tiling", action="store_true", default=None, help="Принудительно использовать tiling")
    parser.add_argument("--no-tiling", action="store_true", help="Отключить tiling")
    return parser.parse_args()


def collect_images(source: Path, extensions: tuple[str, ...]) -> list[Path]:
    if source.is_file():
        return [source]
    return [p for p in source.rglob("*") if p.suffix.lower() in extensions]


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    validator = InputValidator()
    detector = SatelliteDetector(config_path=args.config, weights=args.weights)
    print(f"Модель: {detector.weights_path}")

    extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    sources = collect_images(source, extensions) if args.batch or source.is_dir() else [source]

    valid_sources: list[Path] = []
    for src in sources:
        ok, msg = validator.validate_file(src)
        if ok:
            valid_sources.append(src)
        else:
            print(f"Пропуск {src}: {msg}")

    if not valid_sources:
        print("Нет валидных изображений для обработки.")
        sys.exit(1)

    use_tiling = None
    if args.tiling:
        use_tiling = True
    elif args.no_tiling:
        use_tiling = False

    print(f"Обработка {len(valid_sources)} изображений...")
    start = time.perf_counter()
    results = detector.detect(
        source=[str(s) for s in valid_sources],
        output_dir=args.output,
        use_tiling=use_tiling,
    )
    elapsed = time.perf_counter() - start

    print(f"\n=== Результаты ({elapsed:.2f} с, {elapsed / len(valid_sources):.2f} с/изобр.) ===")
    for r in results:
        print(f"{r['image_name']}: {r['num_detections']} объектов")
    print(f"Сохранено в: {args.output}")


if __name__ == "__main__":
    main()
