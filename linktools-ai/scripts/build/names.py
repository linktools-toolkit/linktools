#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-word filename policy."""

import json
import re
from pathlib import Path


def check_names(source_root: "str | Path", exception_file: "str | Path | None" = None) -> "tuple[str, ...]":
    root = Path(source_root)
    raw_exceptions = json.loads(Path(exception_file).read_text(encoding="utf-8")) if exception_file and Path(exception_file).exists() else {}
    exception_names = set(raw_exceptions.get("names", ())) if isinstance(raw_exceptions, dict) else set()
    errors: list[str] = []
    modules = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts and path.stem != "__init__"]
    for path in modules:
        if "__pycache__" in path.parts:
            continue
        if path.stem == "__init__" or path.stem in exception_names:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9]*", path.stem):
            errors.append(str(path))
    by_stem: dict[str, list[Path]] = {}
    for path in modules:
        by_stem.setdefault(path.stem, []).append(path)
    frozen = {"asset", "storage"}
    for stem, paths in sorted(by_stem.items()):
        if len(paths) > 1 and not all(path.relative_to(root).parts[0] in frozen for path in paths):
            errors.append(f"duplicate module basename: {stem}: {', '.join(str(path) for path in sorted(paths))}")
    package_names = {
        path.name
        for path in root.rglob("*")
        if path.is_dir() and path.name != "__pycache__" and (path / "__init__.py").is_file()
    }
    for path in modules:
        if path.relative_to(root).parts[0] not in frozen and path.stem in package_names:
            errors.append(f"module/package stem collision: {path}")
    for path in root.rglob("*"):
        if not path.is_dir() or path.name == "__pycache__":
            continue
        if not re.fullmatch(r"[a-z][a-z0-9]*", path.name) and path.name not in exception_names:
            errors.append(str(path))
    return tuple(errors)


__all__ = ["check_names"]
