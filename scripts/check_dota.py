#!/usr/bin/env python3
"""Проверка структуры скачанного DOTA и поиск папок с разметкой."""

from __future__ import annotations

import argparse
from pathlib import Path


def scan(path: Path, max_depth: int = 4) -> None:
    print(f"\nСканирование: {path.resolve()}\n")
    if not path.exists():
        print("  Путь не существует!")
        return

    txt_dirs: list[tuple[Path, int]] = []
    img_dirs: list[tuple[Path, int]] = []

    for p in path.rglob("*"):
        if not p.is_dir():
            continue
        depth = len(p.relative_to(path).parts)
        if depth > max_depth:
            continue
        n_txt = len(list(p.glob("*.txt")))
        n_img = len(list(p.glob("*.png"))) + len(list(p.glob("*.jpg")))
        name_lower = p.name.lower()
        if n_txt >= 5 and ("label" in name_lower or "ann" in name_lower or n_txt > 50):
            txt_dirs.append((p, n_txt))
        if n_img >= 5 and ("image" in name_lower or n_img > 50 or p.name in ("train", "val")):
            img_dirs.append((p, n_img))

    print("=== Папки с .txt (кандидаты в разметку) ===")
    for d, n in sorted(txt_dirs, key=lambda x: -x[1])[:15]:
        print(f"  {n:5d} txt  {d}")

    print("\n=== Папки с .png/.jpg (кандидаты в images) ===")
    for d, n in sorted(img_dirs, key=lambda x: -x[1])[:15]:
        print(f"  {n:5d} img  {d}")

    # test — без разметки это норма
    test_only = path / "test"
    if test_only.exists() and not any((test_only / "labelTxt").exists() for _ in [0]):
        has_test_imgs = any(test_only.rglob("*.png"))
        if has_test_imgs:
            print("\n⚠ Папка test/ обычно БЕЗ публичной разметки — для обучения нужны train + val.")

    print("\n=== Что скачать, если txt-папок нет ===")
    print("  https://captain-whu.github.io/DOTA/dataset.html")
    print("  Отдельные ссылки: Annotations / labelTxt (не только Images).")
    print("  Для DOTA-v2.0: trainset_reclabelTxt и valset_reclabelTxt (или полный пакет v2).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка структуры DOTA")
    parser.add_argument("--input", type=str, required=True, help="Корень DOTA или папка train")
    args = parser.parse_args()
    scan(Path(args.input))


if __name__ == "__main__":
    main()
