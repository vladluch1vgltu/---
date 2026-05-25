"""Модуль обучения модели YOLO на спутниковых изображениях."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from utils.weights import find_best_weights, publish_best_weights, resolve_pretrained_weights


class ModelTrainer:
    """Обучение модели с поддержкой transfer learning, fine-tuning и early stopping."""

    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.model_cfg = self.config["model"]
        self.train_cfg = self.config["training"]
        self.aug_cfg = self.config["augmentation"]
        self.paths = self.config["paths"]

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_weights(self) -> str:
        if not self.model_cfg.get("pretrained", True):
            arch = self.model_cfg.get("architecture", "yolov8")
            size = self.model_cfg.get("size", "m")
            return f"{arch}{size}.yaml"
        return resolve_pretrained_weights(
            self.model_cfg.get("weights"),
            size=self.model_cfg.get("size", "m"),
            architecture=self.model_cfg.get("architecture", "yolov8"),
        )

    def train(
        self,
        data_yaml: str | Path = "dataset.yaml",
        resume: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        """
        Запуск обучения модели.

        Args:
            data_yaml: Путь к dataset.yaml в формате YOLO.
            resume: Продолжить с последнего checkpoint.
            **overrides: Переопределение параметров обучения.

        Returns:
            Словарь с метриками и путями к артефактам.
        """
        weights = self._resolve_weights()
        model = YOLO(weights)

        train_args = {
            "data": str(data_yaml),
            "epochs": overrides.get("epochs", self.train_cfg["epochs"]),
            "batch": overrides.get("batch_size", self.train_cfg["batch_size"]),
            "imgsz": overrides.get("img_size", self.config["preprocessing"]["img_size"]),
            "patience": overrides.get("patience", self.train_cfg["patience"]),
            "optimizer": self.train_cfg["optimizer"],
            "lr0": self.train_cfg["lr0"],
            "lrf": self.train_cfg["lrf"],
            "warmup_epochs": self.train_cfg["warmup_epochs"],
            "device": overrides.get("device", self.train_cfg["device"]),
            "workers": self.train_cfg["workers"],
            "save_period": self.train_cfg["save_period"],
            "amp": self.train_cfg["amp"],
            "cache": self.train_cfg.get("cache", False),
            "close_mosaic": self.train_cfg.get("close_mosaic", 10),
            "cos_lr": self.train_cfg.get("cos_lr", False),
            "project": self.paths["models"],
            "name": "checkpoints",
            "exist_ok": True,
            "pretrained": self.model_cfg.get("pretrained", True),
            "resume": resume,
            # Аугментация
            "degrees": self.aug_cfg["degrees"],
            "flipud": self.aug_cfg["flipud"],
            "fliplr": self.aug_cfg["fliplr"],
            "scale": self.aug_cfg["scale"],
            "hsv_h": self.aug_cfg["hsv_h"],
            "hsv_s": self.aug_cfg["hsv_s"],
            "hsv_v": self.aug_cfg["hsv_v"],
            "mosaic": self.aug_cfg["mosaic"] if self.aug_cfg["enabled"] else 0.0,
            "mixup": self.aug_cfg["mixup"] if self.aug_cfg["enabled"] else 0.0,
        }

        if not self.model_cfg.get("pretrained", True):
            train_args["pretrained"] = False

        model.train(**train_args)

        models_dir = Path(self.paths["models"])
        best_weights = find_best_weights(models_dir)
        if best_weights is None and getattr(model, "trainer", None) is not None:
            fallback = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if fallback.is_file():
                best_weights = fallback

        target = models_dir / "best.pt"
        if best_weights is not None:
            publish_best_weights(best_weights, models_dir)

        metrics = self._extract_metrics(model, data_yaml)
        save_dir = (
            best_weights.parent.parent
            if best_weights is not None
            else models_dir / "checkpoints"
        )
        return {
            "best_weights": str(target if target.exists() else best_weights),
            "metrics": metrics,
            "save_dir": str(save_dir),
        }

    def _extract_metrics(self, model: YOLO, data_yaml: str | Path) -> dict[str, float]:
        """Расчёт метрик после обучения."""
        val_results = model.val(data=str(data_yaml))
        metrics: dict[str, float] = {}
        if hasattr(val_results, "box"):
            box = val_results.box
            metrics["precision"] = float(getattr(box, "mp", 0.0))
            metrics["recall"] = float(getattr(box, "mr", 0.0))
            metrics["map50"] = float(getattr(box, "map50", 0.0))
            metrics["map50_95"] = float(getattr(box, "map", 0.0))
            if metrics["precision"] + metrics["recall"] > 0:
                p, r = metrics["precision"], metrics["recall"]
                metrics["f1"] = 2 * p * r / (p + r)
            else:
                metrics["f1"] = 0.0
        return metrics

    def export_onnx(self, weights: str | Path | None = None, opset: int = 12) -> str:
        """Экспорт модели в ONNX."""
        from utils.weights import resolve_trained_weights

        w = resolve_trained_weights(weights, self.paths["models"])
        model = YOLO(str(w))
        export_path = model.export(format="onnx", opset=opset)
        return str(export_path)
