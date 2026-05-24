"""Загрузка конфигурации системы."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
