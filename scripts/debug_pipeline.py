#!/usr/bin/env python3
"""
Пошаговая отладка pipeline перед полным обучением.

Использование:
    python scripts/debug_pipeline.py
    python scripts/debug_pipeline.py --train-epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_dataset() -> bool:
    print("\n[1/5] Проверка датасета...")
    ok = True
    for split in ("train", "val"):
        img_dir = ROOT / "dataset" / "images" / split
        lbl_dir = ROOT / "dataset" / "labels" / split
        imgs = {p.stem for p in img_dir.glob("*.png")}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        matched = len(imgs & lbls)
        print(f"  {split}: images={len(imgs)}, labels={len(lbls)}, pairs={matched}")
        if matched == 0:
            ok = False
        orphan_img = len(imgs - lbls)
        if orphan_img:
            print(f"    предупреждение: {orphan_img} изображений без разметки")
    yaml_path = ROOT / "dataset.yaml"
    if not yaml_path.exists():
        print("  ОШИБКА: нет dataset.yaml")
        ok = False
    return ok


def check_inference_sample() -> bool:
    print("\n[2/5] Тест инференса на одном снимке val...")
    val_imgs = list((ROOT / "dataset" / "images" / "val").glob("*.png"))
    if not val_imgs:
        print("  ОШИБКА: нет val изображений")
        return False
    sample = val_imgs[0]
    from inference.detector import SatelliteDetector

    det = SatelliteDetector(config_path=ROOT / "configs" / "default.yaml")
    results = det.detect(source=sample, output_dir=ROOT / "results" / "debug")
    r = results[0]
    print(f"  {sample.name}: {r['num_detections']} детекций -> results/debug/")
    return True


def check_pretrained_val() -> bool:
    print("\n[3/5] Проверка dataset.yaml на предобученной модели (COCO)...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.val(
        data=str(ROOT / "dataset.yaml"),
        split="val",
        imgsz=640,
        batch=4,
        device="cpu",
        plots=False,
        verbose=False,
    )
    map50 = float(results.box.map50)
    print(f"  mAP50 (COCO, без дообучения): {map50:.4f} — низко это нормально")
    return True


def run_short_train(epochs: int) -> bool:
    print(f"\n[4/5] Короткое обучение ({epochs} эпох) для отладки...")
    from train.trainer import ModelTrainer

    trainer = ModelTrainer(config_path=ROOT / "configs" / "default.yaml")
    result = trainer.train(
        data_yaml=ROOT / "dataset.yaml",
        epochs=epochs,
        batch_size=4,
        device="cpu",
    )
    print(f"  Веса: {result['best_weights']}")
    for k, v in result.get("metrics", {}).items():
        print(f"  {k}: {v:.4f}")
    return Path(result["best_weights"]).exists()


def check_evaluate() -> bool:
    print("\n[5/5] Оценка best.pt...")
    from utils.weights import resolve_trained_weights

    try:
        weights = resolve_trained_weights(None, ROOT / "models")
    except FileNotFoundError:
        print("  пропуск: веса не найдены")
        return False
    from evaluate.metrics import ModelEvaluator

    report = ModelEvaluator(config_path=ROOT / "configs" / "default.yaml").evaluate(
        weights=weights, data_yaml=ROOT / "dataset.yaml", split="val"
    )
    print(f"  mAP50: {report['metrics'].get('map50', 0)}")
    print(f"  отчёт: results/evaluation_report.json")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Отладка pipeline")
    parser.add_argument("--train-epochs", type=int, default=0, help=">0 — запустить короткое обучение")
    parser.add_argument("--skip-val", action="store_true", help="Пропустить val на COCO (долго)")
    args = parser.parse_args()

    print("=== Отладка системы распознавания ===")
    steps = [
        ("dataset", check_dataset),
        ("inference", check_inference_sample),
    ]
    if not args.skip_val:
        steps.append(("pretrained_val", check_pretrained_val))
    if args.train_epochs > 0:
        steps.append(("train", lambda: run_short_train(args.train_epochs)))
        steps.append(("evaluate", check_evaluate))

    failed = []
    for name, fn in steps:
        try:
            if not fn():
                failed.append(name)
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            failed.append(name)

    print("\n=== Итог ===")
    if failed:
        print(f"Проблемы на шагах: {', '.join(failed)}")
        sys.exit(1)
    print("Все проверки пройдены. Для полного обучения: python train.py --data dataset.yaml --epochs 100 --device cpu")


if __name__ == "__main__":
    main()
