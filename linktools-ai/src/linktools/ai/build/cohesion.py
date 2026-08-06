#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small static cohesion checks for module naming and ownership."""

from pathlib import Path


def check_files(source_root: "str | Path") -> "tuple[str, ...]":
    errors = []
    for path in Path(source_root).rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if any(token in path.name.lower() for token in ("helper", "utils", "manager", "common")):
            errors.append(str(path))
    return tuple(errors)


__all__ = ["check_files"]
