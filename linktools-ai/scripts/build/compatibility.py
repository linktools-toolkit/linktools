
"""P0/P1 public symbol compatibility manifest."""

import json
from pathlib import Path
from typing import cast

from linktools.ai.core import JsonValue


def build_manifest(path: "str | Path") -> "dict[str, JsonValue]":
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return cast("dict[str, JsonValue]", value) if isinstance(value, dict) else {}


def validate_manifest(manifest: "dict[str, JsonValue]") -> "tuple[str, ...]":
    errors = []
    symbols = manifest.get("symbols", [])
    if not isinstance(symbols, list):
        return ("symbols must be a list",)
    for raw_item in symbols:
        if not isinstance(raw_item, dict):
            errors.append("invalid symbol entry")
            continue
        item = cast("dict[str, JsonValue]", raw_item)
        priority = item.get("priority")
        status = item.get("status")
        old_symbol = str(item.get("old_symbol", ""))
        if priority in {"P0", "P1"} and status == "REMOVED":
            errors.append(old_symbol)
        if priority in {"P0", "P1"} and not item.get("canonical_symbol"):
            errors.append(old_symbol)
    return tuple(errors)


__all__ = ["build_manifest", "validate_manifest"]
