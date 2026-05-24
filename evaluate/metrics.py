"""Расчёт метрик качества: Precision, Recall, F1, IoU, mAP50, mAP50-95."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


class ModelEvaluator:
    """Автоматический расчёт метрик после обучения или на тестовой выборке."""

    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.eval_cfg = self.config["evaluation"]
        self.paths = self.config["paths"]

    def evaluate(
        self,
        weights: str | Path,
        data_yaml: str | Path = "dataset.yaml",
        split: str = "val",
    ) -> dict[str, Any]:
        """
        Оценка модели на размеченном датасете.

        Returns:
            Метрики и флаг соответствия минимальным требованиям ТЗ.
        """
        model = YOLO(str(weights))
        results = model.val(
            data=str(data_yaml),
            split=split,
            conf=self.config["detection"]["conf_threshold"],
            iou=self.config["detection"]["iou_threshold"],
        )

        metrics: dict[str, float] = {}
        per_class: dict[str, dict[str, float]] = {}

        if hasattr(results, "box"):
            box = results.box
            metrics["precision"] = round(float(getattr(box, "mp", 0.0)), 4)
            metrics["recall"] = round(float(getattr(box, "mr", 0.0)), 4)
            metrics["map50"] = round(float(getattr(box, "map50", 0.0)), 4)
            metrics["map50_95"] = round(float(getattr(box, "map", 0.0)), 4)

            p, r = metrics["precision"], metrics["recall"]
            metrics["f1"] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)
            metrics["iou"] = metrics["map50"]

            if hasattr(box, "ap50") and box.ap50 is not None:
                class_names = self.config["classes"]
                for i, ap in enumerate(box.ap50):
                    name = class_names[i] if i < len(class_names) else f"class_{i}"
                    per_class[name] = {"ap50": round(float(ap), 4)}

        requirements = self._check_requirements(metrics)
        report = {
            "weights": str(weights),
            "data": str(data_yaml),
            "split": split,
            "metrics": metrics,
            "per_class": per_class,
            "requirements_met": requirements["met"],
            "requirements_detail": requirements,
        }

        out_dir = Path(self.paths["results"])
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _check_requirements(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Проверка минимальных требований ТЗ."""
        checks = {
            "map50": metrics.get("map50", 0) >= self.eval_cfg["min_map50"],
            "precision": metrics.get("precision", 0) >= self.eval_cfg["min_precision"],
            "recall": metrics.get("recall", 0) >= self.eval_cfg["min_recall"],
        }
        return {
            "met": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "min_map50": self.eval_cfg["min_map50"],
                "min_precision": self.eval_cfg["min_precision"],
                "min_recall": self.eval_cfg["min_recall"],
            },
        }
