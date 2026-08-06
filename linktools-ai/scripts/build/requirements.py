#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Logical dependency group loading."""

from pathlib import Path
from typing import cast

from linktools.ai.core.json import JsonValue

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


def load_requirements(path: "str | Path") -> "dict[str, JsonValue]":
    if _yaml is None:
        raise RuntimeError("PyYAML is required to read requirements.yml")
    value = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cast("dict[str, JsonValue]", value) if isinstance(value, dict) else {}


__all__ = ["load_requirements"]
