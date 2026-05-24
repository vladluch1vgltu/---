#!/usr/bin/env python3
"""
Оценка качества модели: Precision, Recall, F1, mAP50, mAP50-95.

Использование:
    python evaluate.py --weights models/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate.metrics import ModelEvaluator
from utils.weights import resolve_trained_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка качества модели")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Путь к весам (по умолчанию: models/best.pt или runs/detect/.../best.pt)",
    )
    parser.add_argument("--data", type=str, default="dataset.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = ModelEvaluator(config_path=args.config)
    weights = resolve_trained_weights(args.weights, evaluator.paths["models"])
    print(f"Веса: {weights}")
    report = evaluator.evaluate(weights=weights, data_yaml=args.data, split=args.split)

    print("\n=== Метрики оценки ===")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value}")

    print("\n=== Требования ТЗ ===")
    for check, passed in report["requirements_detail"]["checks"].items():
        status = "OK" if passed else "НЕ ВЫПОЛНЕНО"
        print(f"  {check}: {status}")

    overall = "ВЫПОЛНЕНЫ" if report["requirements_met"] else "НЕ ВЫПОЛНЕНЫ"
    print(f"\nИтого: требования {overall}")
    print(f"Отчёт: results/evaluation_report.json")


if __name__ == "__main__":
    main()
