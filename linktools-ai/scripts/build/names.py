#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-word filename and package-boundary policy."""

import re
from pathlib import Path


def _semantic_stem(path: Path) -> str:
    return path.stem.removeprefix("_")


def _namespace_root(root: Path) -> Path:
    current = root
    while (current.parent / "__init__.py").is_file():
        current = current.parent
    return current


def check_names(source_root: "str | Path") -> "tuple[str, ...]":
    root = _namespace_root(Path(source_root))
    errors: list[str] = []
    modules = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts and path.stem != "__init__"]
    for path in modules:
        if "__pycache__" in path.parts:
            continue
        if not re.fullmatch(r"_?[a-z][a-z0-9]*", path.stem):
            errors.append(str(path))
    package_paths = tuple(
        path.relative_to(root).parts
        for path in root.rglob("*")
        if path.is_dir() and path.name != "__pycache__" and (path / "__init__.py").is_file()
    )
    for path in modules:
        module_parts = path.relative_to(root).with_suffix("").parts
        parent_parts = module_parts[:-1]
        stem = _semantic_stem(path)
        if any(
            package_path[-1] == stem
            and (
                package_path == (*parent_parts, stem)
                or parent_parts[: len(package_path)] == package_path
            )
            for package_path in package_paths
        ):
            errors.append(f"module/package stem collision: {path}")
    for path in modules:
        module_parts = path.relative_to(root).with_suffix("").parts
        parent_parts = module_parts[:-1]
        stem = _semantic_stem(path)
        conflicts = tuple(
            other
            for other in modules
            if other != path
            and _semantic_stem(other) == stem
            and len(other.relative_to(root).parts) < len(module_parts)
            and parent_parts[: len(other.relative_to(root).parts) - 1] == other.relative_to(root).with_suffix("").parts[:-1]
        )
        if conflicts:
            names = ", ".join(str(item) for item in sorted(conflicts))
            errors.append(f"nested module basename collision: {path}: {names}")
    for path in root.rglob("*"):
        if not path.is_dir() or path.name == "__pycache__":
            continue
        if not re.fullmatch(r"[a-z][a-z0-9]*", path.name):
            errors.append(str(path))
    return tuple(errors)


__all__ = ["check_names"]
