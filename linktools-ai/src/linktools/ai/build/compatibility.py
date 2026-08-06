#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""P0/P1 public symbol compatibility manifest."""

import json
from pathlib import Path


def build_manifest(path: "str | Path") -> "dict[str, object]":
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest(manifest: "dict[str, object]") -> "tuple[str, ...]":
    errors = []
    for item in manifest.get("symbols", []):
        if item.get("priority") in {"P0", "P1"} and item.get("status") == "REMOVED":
            errors.append(str(item.get("old_symbol")))
        if item.get("priority") in {"P0", "P1"} and not item.get("canonical_symbol"):
            errors.append(str(item.get("old_symbol")))
    return tuple(errors)


__all__ = ["build_manifest", "validate_manifest"]
