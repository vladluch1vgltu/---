#!/usr/bin/env python3
"""
Предварительная обработка спутниковых изображений.

Использование:
    python preprocess.py --input dataset/raw --output dataset/processed
    python preprocess.py --input image.tif --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from preprocess.pipeline import ImagePreprocessor, DataAugmentor
from utils.security import InputValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Предобработка спутниковых изображений")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Файл или директория")
    parser.add_argument("--output", type=str, default=None, help="Директория вывода")
    parser.add_argument("--preview", action="store_true", help="Показать результат (один файл)")
    parser.add_argument("--augment", action="store_true", help="Применить аугментацию")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validator = InputValidator()
    preprocessor = ImagePreprocessor(config_path=args.config)
    augmentor = DataAugmentor(config_path=args.config) if args.augment else None

    input_path = Path(args.input)

    if input_path.is_dir():
        if not args.output:
            print("Для пакетной обработки укажите --output")
            sys.exit(1)
        augmentor = augmentor or DataAugmentor(config_path=args.config)
        count = augmentor.batch_preprocess(input_path, args.output)
        print(f"Обработано изображений: {count}")
        return

    ok, msg = validator.validate_file(input_path)
    if not ok:
        print(f"Ошибка: {msg}")
        sys.exit(1)

    img = preprocessor.load_image(input_path)
    processed = preprocessor.preprocess(img)
    if augmentor:
        processed = augmentor.augment(processed)

    if args.preview:
        cv2.imshow("Original", img)
        cv2.imshow("Processed", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), processed)
        print(f"Сохранено: {out}")
    else:
        out = Path("results") / f"{input_path.stem}_preprocessed.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), processed)
        print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
