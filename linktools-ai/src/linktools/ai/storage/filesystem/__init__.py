#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared filesystem primitives for persistence implementations."""


from .atomic import atomic_write_bytes

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

def validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def atomic_write(path: "Path", content: bytes) -> None:
    atomic_write_bytes(path, content)

__all__ = ["atomic_write", "atomic_write_bytes", "validate_id_segment"]
