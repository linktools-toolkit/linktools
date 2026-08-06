#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-word filename policy."""

import json
import re
from pathlib import Path


def check_names(source_root: "str | Path", exception_file: "str | Path | None" = None) -> "tuple[str, ...]":
    root = Path(source_root)
    exceptions = json.loads(Path(exception_file).read_text(encoding="utf-8")) if exception_file and Path(exception_file).exists() else {}
    errors = []
    for path in root.rglob("*.py"):
        if path.stem == "__init__" or path.stem in exceptions:
            continue
        if not re.fullmatch(r"[a-z0-9]+", path.stem):
            errors.append(str(path))
    return tuple(errors)


__all__ = ["check_names"]
