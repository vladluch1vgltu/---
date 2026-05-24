"""Поиск и публикация весов обученной модели (Ultralytics сохраняет в runs/detect/)."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path


def training_weights_dir(models_dir: str | Path = "models") -> Path:
    """Каталог, куда Ultralytics пишет best.pt при project=models, name=checkpoints."""
    return Path("runs/detect") / models_dir / "checkpoints" / "weights"


def candidate_best_paths(models_dir: str | Path = "models") -> list[Path]:
    """Возможные расположения best.pt (от более предпочтительного к запасным)."""
    models_dir = Path(models_dir)
    return [
        models_dir / "best.pt",
        training_weights_dir(models_dir) / "best.pt",
        models_dir / "checkpoints" / "weights" / "best.pt",
    ]


def find_best_weights(models_dir: str | Path = "models") -> Path | None:
    """Найти обученные веса best.pt или None."""
    for path in candidate_best_paths(models_dir):
        if path.is_file():
            return path
    return None


def publish_best_weights(
    source: str | Path,
    models_dir: str | Path = "models",
) -> Path:
    """Скопировать best.pt в models/best.pt для detect/evaluate/API."""
    source = Path(source)
    target = Path(models_dir) / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def resolve_trained_weights(
    explicit: str | Path | None = None,
    models_dir: str | Path = "models",
    *,
    allow_pretrained_fallback: bool = False,
    pretrained_name: str = "yolov8m.pt",
) -> Path:
    """
    Путь к весам для инференса/оценки.

    Приоритет: явный аргумент → models/best.pt → runs/detect/.../best.pt.
  """
    models_dir = Path(models_dir)

    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Веса не найдены: {path}")

    found = find_best_weights(models_dir)
    if found is not None:
        return found

    if allow_pretrained_fallback:
        for candidate in (models_dir / pretrained_name, Path(pretrained_name)):
            if candidate.is_file():
                warnings.warn(
                    f"Обученные веса не найдены, используется предобученная COCO-модель: {candidate}. "
                    f"Обучите модель (python train.py) или укажите --weights.",
                    stacklevel=2,
                )
                return candidate

    searched = "\n  ".join(str(p) for p in candidate_best_paths(models_dir))
    raise FileNotFoundError(
        "Обученные веса не найдены. Ожидались файлы:\n  "
        f"{searched}\n"
        "Запустите обучение: python train.py --data dataset.yaml"
    )


def resolve_pretrained_weights(
    custom: str | Path | None,
    size: str = "m",
    architecture: str = "yolov8",
) -> str:
    """Веса для старта обучения (transfer learning)."""
    if custom:
        path = Path(custom)
        if path.is_file():
            return str(path)
    name = f"{architecture}{size}.pt"
    for candidate in (Path(name), Path("models") / name):
        if candidate.is_file():
            return str(candidate)
    return name
