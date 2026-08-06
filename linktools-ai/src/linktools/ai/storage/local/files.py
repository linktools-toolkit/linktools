#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Atomic local JSON and byte-file operations."""


import json
from pathlib import Path
from ...foundation.json import normalize_json
from ..filesystem.atomic import atomic_write_bytes as atomic_replace_bytes

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...foundation.json import JsonValue

def atomic_write_bytes(path: "str | Path", content: bytes) -> None:
    atomic_replace_bytes(Path(path), content)


def atomic_write_json(path: "str | Path", value: "JsonValue") -> None:
    atomic_write_bytes(
        path,
        json.dumps(
            normalize_json(value),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
    )


def read_bytes(path: "str | Path") -> bytes:
    return Path(path).read_bytes()


def read_json(path: "str | Path") -> "JsonValue":
    return normalize_json(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = ["atomic_write_bytes", "atomic_write_json", "read_bytes", "read_json"]
