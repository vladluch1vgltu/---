"""Проверка входных файлов и защита от некорректных данных."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class InputValidator:
    """Валидация загружаемых изображений согласно требованиям безопасности."""

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            from utils.config import load_config
            config = load_config()
        self.cfg = config.get("security", {})

    def validate_file(self, path: str | Path) -> tuple[bool, str]:
        """
        Проверка файла перед обработкой.

        Returns:
            (успех, сообщение об ошибке)
        """
        path = Path(path)

        if not path.exists():
            return False, "Файл не существует"

        if not path.is_file():
            return False, "Указанный путь не является файлом"

        ext = path.suffix.lower()
        allowed = [e.lower() for e in self.cfg.get("allowed_extensions", [".jpg", ".png"])]
        if ext not in allowed:
            return False, f"Недопустимый формат: {ext}. Разрешены: {', '.join(allowed)}"

        max_mb = self.cfg.get("max_file_size_mb", 100)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            return False, f"Размер файла ({size_mb:.1f} MB) превышает лимит {max_mb} MB"

        try:
            img = cv2.imread(str(path))
            if img is None and ext in {".tif", ".tiff"}:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Не удалось декодировать изображение"

            h, w = img.shape[:2]
            max_dim = self.cfg.get("max_image_dimension", 16384)
            if max(h, w) > max_dim:
                return False, f"Размер изображения ({w}x{h}) превышает лимит {max_dim}px"

            if not np.isfinite(img).all():
                return False, "Изображение содержит некорректные значения пикселей"

        except Exception as e:
            return False, f"Ошибка чтения изображения: {e}"

        return True, "OK"

    def validate_upload_bytes(self, data: bytes, filename: str) -> tuple[bool, str]:
        """Валидация загруженных через API байтов."""
        ext = Path(filename).suffix.lower()
        allowed = [e.lower() for e in self.cfg.get("allowed_extensions", [])]
        if ext not in allowed:
            return False, f"Недопустимый формат: {ext}"

        max_bytes = self.cfg.get("max_file_size_mb", 100) * 1024 * 1024
        if len(data) > max_bytes:
            return False, "Размер файла превышает допустимый лимит"

        if len(data) < 100:
            return False, "Файл слишком маленький для изображения"

        return True, "OK"
