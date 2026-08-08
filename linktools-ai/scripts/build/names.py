#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-word filename policy."""

import re
from pathlib import Path


def _semantic_stem(path: Path) -> str:
    return path.stem.removeprefix("_")


def check_names(source_root: "str | Path") -> "tuple[str, ...]":
    root = Path(source_root)
    errors: list[str] = []
    modules = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts and path.stem != "__init__"]
    for path in modules:
        if "__pycache__" in path.parts:
            continue
        if not re.fullmatch(r"_?[a-z][a-z0-9]*", path.stem):
            errors.append(str(path))
    by_stem: dict[str, list[Path]] = {}
    for path in modules:
        by_stem.setdefault(_semantic_stem(path), []).append(path)
    for stem, paths in sorted(by_stem.items()):
        if len(paths) > 1:
            errors.append(f"duplicate module basename: {stem}: {', '.join(str(path) for path in sorted(paths))}")
    package_paths = {
        f"{path.relative_to(root).as_posix()}/"
        for path in root.rglob("*")
        if path.is_dir() and path.name != "__pycache__" and (path / "__init__.py").is_file()
    }
    for path in modules:
        collisions = tuple(
            package_path
            for package_path in package_paths
            if package_path.rstrip("/").rsplit("/", 1)[-1] == _semantic_stem(path)
        )
        if collisions:
            errors.append(f"module/package stem collision: {path}")
    for path in root.rglob("*"):
        if not path.is_dir() or path.name == "__pycache__":
            continue
        if not re.fullmatch(r"[a-z][a-z0-9]*", path.name):
            errors.append(str(path))
    return tuple(errors)


__all__ = ["check_names"]
