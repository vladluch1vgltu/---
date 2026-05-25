#!/usr/bin/env python3
"""
Обучение модели YOLO на спутниковых изображениях.

Использование:
    python train.py --data dataset.yaml --epochs 100
    python train.py --resume
    python train.py --export-onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение модели распознавания объектов на спутниковых изображениях"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Конфигурация (для GPU-сервера: configs/gpu_l40s.yaml)",
    )
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Путь к dataset.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=None, help="Размер батча")
    parser.add_argument("--imgsz", type=int, default=None, help="Размер стороны входа (например 1024)")
    parser.add_argument("--device", type=str, default=None, help="0, cuda, cpu или auto")
    parser.add_argument("--resume", action="store_true", help="Продолжить обучение с checkpoint")
    parser.add_argument("--export-onnx", action="store_true", help="Экспортировать лучшую модель в ONNX")
    parser.add_argument("--weights", type=str, default=None, help="Путь к весам для экспорта")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ModelTrainer(config_path=args.config)

    if args.export_onnx:
        path = trainer.export_onnx(weights=args.weights)
        print(f"Модель экспортирована в ONNX: {path}")
        return

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.device:
        overrides["device"] = args.device
    if args.imgsz:
        overrides["img_size"] = args.imgsz

    print("Запуск обучения...")
    print(f"Конфиг: {args.config}")
    result = trainer.train(data_yaml=args.data, resume=args.resume, **overrides)

    print("\n=== Результаты обучения ===")
    print(f"Лучшие веса: {result['best_weights']}")
    print(f"Директория: {result['save_dir']}")
    print("\nМетрики:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
