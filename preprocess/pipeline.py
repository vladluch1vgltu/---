"""Предварительная обработка и аугментация спутниковых изображений."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


class ImagePreprocessor:
    """Resize, нормализация, шумоподавление, коррекция яркости/контраста."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path = "configs/default.yaml"):
        if config is None:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.cfg = config["preprocessing"]

    def load_image(self, path: str | Path) -> np.ndarray:
        """Загрузка JPG, PNG, TIFF."""
        path = Path(path)
        if path.suffix.lower() in {".tif", ".tiff"}:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {path}")
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            return img
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {path}")
        return img

    def preprocess(self, img: np.ndarray, target_size: int | None = None) -> np.ndarray:
        """Полный pipeline предобработки."""
        out = img.copy()

        if self.cfg.get("denoise", True):
            out = self.denoise(out)

        if self.cfg.get("brightness_contrast", True):
            out = self.adjust_brightness_contrast(out)

        size = target_size or self.cfg.get("img_size", 640)
        if self.cfg.get("normalize", True):
            out = self.resize_and_normalize(out, size)
        else:
            out = cv2.resize(out, (size, size))

        return out

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """Подавление шума (Non-local Means)."""
        return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    @staticmethod
    def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.1, beta: int = 5) -> np.ndarray:
        """Коррекция яркости (alpha) и контраста (beta)."""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def resize_and_normalize(img: np.ndarray, size: int) -> np.ndarray:
        """Изменение размера с сохранением пропорций и паддинг."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas

    def to_tensor_format(self, img: np.ndarray) -> np.ndarray:
        """Подготовка к входному тензору: BGR -> RGB, [0,1], CHW."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))


class DataAugmentor:
    """Аугментация для обучения: поворот, отражение, масштаб, яркость, шум, размытие."""

    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)["augmentation"]

    def augment(self, img: np.ndarray, seed: int | None = None) -> np.ndarray:
        if not self.cfg.get("enabled", True):
            return img
        rng = np.random.default_rng(seed)
        out = img.copy()

        if rng.random() < 0.5:
            out = cv2.flip(out, 1)

        angle = self.cfg.get("degrees", 15.0)
        if angle > 0:
            rot = rng.uniform(-angle, angle)
            h, w = out.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
            out = cv2.warpAffine(out, M, (w, h))

        scale = self.cfg.get("scale", 0.5)
        if scale > 0 and rng.random() < 0.3:
            factor = rng.uniform(1 - scale * 0.5, 1 + scale * 0.5)
            nh, nw = int(h * factor), int(w * factor)
            out = cv2.resize(out, (nw, nh))
            out = cv2.resize(out, (w, h))

        if rng.random() < self.cfg.get("blur", 0.01) * 10:
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)

        if rng.random() < self.cfg.get("noise", 0.02) * 10:
            noise = rng.integers(-15, 16, out.shape, dtype=np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return out

    def batch_preprocess(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config_path: str | Path = "configs/default.yaml",
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ) -> int:
        """Пакетная предобработка изображений."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = ImagePreprocessor(config_path=config_path)

        count = 0
        for path in input_dir.rglob("*"):
            if path.suffix.lower() not in extensions:
                continue
            img = preprocessor.load_image(path)
            processed = preprocessor.preprocess(img)
            rel = path.relative_to(input_dir)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path.with_suffix(".jpg")), processed)
            count += 1
        return count
