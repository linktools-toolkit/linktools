"""Validation and containment checks for local storage identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...errors import InvalidStoragePathError

_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class StorageId:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StorageId":
        if not isinstance(raw, str) or not 0 < len(raw) <= 255 or raw in {".", ".."} or _ID.fullmatch(raw) is None:
            raise InvalidStoragePathError(f"invalid storage id: {raw!r}")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class StoragePath:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StoragePath":
        if not isinstance(raw, str) or not raw or len(raw) > 512:
            raise InvalidStoragePathError(f"invalid storage path: {raw!r}")
        parts = raw.split("/")
        if any(_ID.fullmatch(part) is None or part in {".", ".."} for part in parts):
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


def safe_child(root: str | Path, *validated_parts: StorageId | StoragePath | Sha256Digest | str) -> Path:
    root_path = Path(root).resolve(strict=False)
    candidate = root_path.joinpath(*(part.value if hasattr(part, "value") else part for part in validated_parts)).resolve(strict=False)
    if not candidate.is_relative_to(root_path):
        raise InvalidStoragePathError(f"path escapes storage root: {candidate}")
    return candidate


__all__ = ["StoragePath", "Sha256Digest", "StorageId", "safe_child"]
