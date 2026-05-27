"""Модуль инференса: обнаружение объектов на спутниковых изображениях."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from preprocess.pipeline import ImagePreprocessor
from utils.weights import resolve_trained_weights


class SatelliteDetector:
    """Детектор объектов с поддержкой больших изображений (tiling) и экспорта результатов."""

    def __init__(
        self,
        config_path: str | Path = "configs/default.yaml",
        weights: str | Path | None = None,
    ):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.det_cfg = self.config["detection"]
        self.inf_cfg = self.config["inference"]
        self.pre_cfg = self.config["preprocessing"]
        self.classes = self.config["classes"]
        self.paths = self.config["paths"]

        w = resolve_trained_weights(weights, self.paths["models"])
        self.model = YOLO(str(w))
        self.weights_path = w
        self.preprocessor = ImagePreprocessor(self.config)
        self.device = self._resolve_device(self.inf_cfg.get("device", "auto"))

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def detect(
        self,
        source: str | Path | list[str | Path],
        output_dir: str | Path | None = None,
        use_tiling: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Обнаружение объектов на одном или нескольких изображениях.

        Returns:
            Список результатов по каждому изображению.
        """
        sources = [source] if isinstance(source, (str, Path)) else list(source)
        output_dir = Path(output_dir or self.paths["results"])
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results: list[dict[str, Any]] = []

        for src in sources:
            src_path = Path(src)
            img = self.preprocessor.load_image(src_path)
            h, w = img.shape[:2]
            tile_thresh = self.pre_cfg.get("tile_size", 1024) * 2

            if use_tiling is None:
                use_tiling = max(h, w) > tile_thresh

            if use_tiling:
                detections = self._detect_tiled(img, src_path.stem)
            else:
                # YOLO сам делает letterbox по imgsz; preprocess() (denoise, 640²) ломает качество.
                detections = self._run_model(
                    img,
                    offset=(0, 0),
                    imgsz=self.pre_cfg.get("img_size", 640),
                )

            result = self._build_result(src_path, detections, img)
            self._save_outputs(result, img, output_dir)
            all_results.append(result)

        return all_results

    def _detect_tiled(self, img: np.ndarray, name: str) -> list[dict[str, Any]]:
        """Скользящее окно для изображений высокого разрешения."""
        tile_size = self.pre_cfg.get("tile_size", 1024)
        overlap = self.pre_cfg.get("tile_overlap", 128)
        stride = tile_size - overlap
        h, w = img.shape[:2]

        all_dets: list[dict[str, Any]] = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)
                tile = img[y1:y2, x1:x2]
                dets = self._run_model(
                    tile,
                    offset=(x1, y1),
                    imgsz=self.pre_cfg.get("tile_size", 1024),
                )
                all_dets.extend(dets)

        return self._merge_tile_detections(all_dets)

    def _run_model(
        self,
        img: np.ndarray,
        offset: tuple[int, int] = (0, 0),
        imgsz: int | None = None,
    ) -> list[dict[str, Any]]:
        """Запуск YOLO на фрагменте (как yolo predict: исходный кадр + imgsz)."""
        size = imgsz or self.pre_cfg.get("img_size", 640)
        results = self.model.predict(
            source=img,
            imgsz=size,
            conf=self.det_cfg["conf_threshold"],
            iou=self.det_cfg["iou_threshold"],
            max_det=self.det_cfg["max_det"],
            device=self.device,
            half=self.inf_cfg.get("half", False),
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        ox, oy = offset

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "bbox": [
                        float(xyxy[0] + ox),
                        float(xyxy[1] + oy),
                        float(xyxy[2] + ox),
                        float(xyxy[3] + oy),
                    ],
                    "class_id": cls_id,
                    "class_name": self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}",
                    "confidence": conf,
                })
        return detections

    def _merge_tile_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """NMS между тайлами."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        classes = np.array([d["class_id"] for d in detections])

        indices = self._nms_numpy(boxes, scores, classes, self.det_cfg["iou_threshold"])
        return [detections[i] for i in indices]

    @staticmethod
    def _nms_numpy(
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        iou_thresh: float,
    ) -> list[int]:
        """Класс-специфичный NMS."""
        keep: list[int] = []
        for cls in np.unique(classes):
            idx = np.where(classes == cls)[0]
            cls_boxes = boxes[idx]
            cls_scores = scores[idx]
            order = cls_scores.argsort()[::-1]
            while order.size > 0:
                i = order[0]
                keep.append(int(idx[i]))
                if order.size == 1:
                    break
                ious = SatelliteDetector._iou_batch(cls_boxes[i], cls_boxes[order[1:]])
                remaining = np.where(ious <= iou_thresh)[0]
                order = order[remaining + 1]
        return keep

    @staticmethod
    def _iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6)

    def _build_result(
        self,
        src_path: Path,
        detections: list[dict[str, Any]],
        img: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "image": str(src_path),
            "image_name": src_path.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "num_detections": len(detections),
            "detections": detections,
        }

    def _save_outputs(
        self,
        result: dict[str, Any],
        img: np.ndarray,
        output_dir: Path,
    ) -> None:
        stem = Path(result["image_name"]).stem

        if self.inf_cfg.get("save_json", True):
            with open(output_dir / f"{stem}_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        if self.inf_cfg.get("save_csv", True) and result["detections"]:
            rows = [
                {
                    "image": result["image_name"],
                    "class": d["class_name"],
                    "confidence": d["confidence"],
                    "x1": d["bbox"][0],
                    "y1": d["bbox"][1],
                    "x2": d["bbox"][2],
                    "y2": d["bbox"][3],
                }
                for d in result["detections"]
            ]
            pd.DataFrame(rows).to_csv(output_dir / f"{stem}_results.csv", index=False)

        if self.inf_cfg.get("save_txt", True):
            with open(output_dir / f"{stem}_results.txt", "w", encoding="utf-8") as f:
                for d in result["detections"]:
                    f.write(
                        f"{d['class_name']} {d['confidence']:.4f} "
                        f"{d['bbox'][0]:.1f} {d['bbox'][1]:.1f} "
                        f"{d['bbox'][2]:.1f} {d['bbox'][3]:.1f}\n"
                    )

        if self.inf_cfg.get("save_img", True):
            vis = self.visualize(img, result["detections"])
            cv2.imwrite(str(output_dir / f"{stem}_detected.jpg"), vis)

    def visualize(self, img: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Отрисовка bounding box, класса и confidence."""
        vis = img.copy()
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0),
        ]
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            color = colors[cls_id % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return vis
