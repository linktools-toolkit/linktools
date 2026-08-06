#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Logical dependency group loading."""

from pathlib import Path


def load_requirements(path: "str | Path") -> "dict[str, object]":
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read requirements.yml") from error
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


__all__ = ["load_requirements"]
