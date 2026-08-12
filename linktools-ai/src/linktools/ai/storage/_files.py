#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root-contained atomic file primitives."""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..core import JsonValue
from ..errors import AIError, ErrorCode, InvalidStoragePathError


def validate_root_path(root: Path, relative: str) -> Path:
    if not relative or "\x00" in relative or "\\" in relative:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
    return candidate


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes_atomic(path: Path, value: bytes, *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
        if fsync:
            sync_directory(path.parent)
    except BaseException as error:
        Path(temporary).unlink(missing_ok=True)
        if replaced:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        raise


def read_json(path: Path) -> 'dict[str, JsonValue]':
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def write_json_atomic(path: Path, value: 'dict[str, JsonValue]', *, fsync: bool = False) -> None:
    write_bytes_atomic(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        fsync=fsync,
    )


def unlink_if_exists(path: Path) -> None:
    path.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    """Flush a directory when the platform exposes directory descriptors."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            return
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: "str | Path", value: bytes) -> None:
    write_bytes_atomic(Path(path), value)


def atomic_write_json(path: "str | Path", value: JsonValue) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


_STORAGE_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class StorageId:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StorageId":
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > 255
            or raw in {".", ".."}
            or _STORAGE_ID.fullmatch(raw) is None
        ):
            raise InvalidStoragePathError(f"invalid storage id: {raw!r}")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class StoragePath:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StoragePath":
        parts = raw.split("/") if isinstance(raw, str) else []
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > 512
            or any(_STORAGE_ID.fullmatch(part) is None for part in parts)
        ):
            raise InvalidStoragePathError(f"invalid storage path: {raw!r}")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class Sha256Digest:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "Sha256Digest":
        if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{64}", raw) is None:
            raise InvalidStoragePathError(f"invalid sha256 digest: {raw!r}")
        return cls(raw)


def safe_child(
    root: "str | Path",
    *validated_parts: "StorageId | StoragePath | Sha256Digest | str",
) -> Path:
    root_path = Path(root).resolve(strict=False)
    values = tuple(
        part.value if isinstance(part, (StorageId, StoragePath, Sha256Digest)) else part
        for part in validated_parts
    )
    candidate = root_path.joinpath(*values).resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise InvalidStoragePathError(f"path escapes storage root: {candidate}") from error
    return candidate


__all__ = [
    "Sha256Digest",
    "StorageId",
    "StoragePath",
    "atomic_write_bytes",
    "atomic_write_json",
    "read_bytes",
    "read_json",
    "safe_child",
    "sync_directory",
    "unlink_if_exists",
    "validate_root_path",
    "write_bytes_atomic",
    "write_json_atomic",
]
