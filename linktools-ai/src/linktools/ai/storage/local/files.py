"""Atomic local JSON and byte-file operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ...json import JsonValue, decode_json, encode_json


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: JsonValue) -> None:
    atomic_write_bytes(path, encode_json(value).encode("utf-8"))


def read_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def read_json(path: str | Path) -> JsonValue:
    return decode_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["atomic_write_bytes", "atomic_write_json", "read_bytes", "read_json"]
