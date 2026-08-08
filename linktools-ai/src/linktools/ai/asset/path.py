#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable AssetRoot and strict key path validation."""

import hashlib
from pathlib import Path

from ..core.errors import ErrorCode, AIError
from .model import AssetKey, AssetRoot

_RESERVED_ROOT_NAMES = frozenset({".asset-revision", ".history", ".txn"})


def file_root(locator: str) -> AssetRoot:
    path = Path(locator).resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


def asset_path(root: AssetRoot, key: AssetKey) -> Path:
    if root.scheme != "file" or not key.kind or not key.id:
        raise AIError(ErrorCode.ASSET_PATH_ABSOLUTE)
    if key.kind in _RESERVED_ROOT_NAMES:
        raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT)
    for index, value in enumerate((key.kind, key.id)):
        if (
            not value
            or len(value.encode("utf-8")) > 512
            or "\x00" in value
            or "\\" in value
            or (index == 0 and "/" in value)
            or value in {".", ".."}
            or any(part in {".", ".."} for part in Path(value).parts)
        ):
            raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT)
        if Path(value).is_absolute() or (len(value) > 1 and value[1] == ":"):
            raise AIError(ErrorCode.ASSET_PATH_ABSOLUTE)
    root_path = Path(root.locator).resolve()
    relative = Path(key.kind) / key.id
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT) from exc
    return candidate


__all__ = ["asset_path", "file_root"]
