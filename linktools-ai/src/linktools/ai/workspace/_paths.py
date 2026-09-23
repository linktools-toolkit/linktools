#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Workspace paths and path contracts."""

from pathlib import Path

_STORAGE_DIR_NAME = ".linktools"


def workspace_storage_root(root: Path) -> Path:
    return root / _STORAGE_DIR_NAME


def workspace_rules_root(root: Path) -> Path:
    return workspace_storage_root(root) / "rules"


def workspace_locks_root(root: Path) -> Path:
    return workspace_storage_root(root) / "locks"


def workspace_storage_name() -> str:
    return _STORAGE_DIR_NAME


def is_workspace_storage_path(path: str) -> bool:
    return path == _STORAGE_DIR_NAME or path.startswith(_STORAGE_DIR_NAME + "/")


def validate_workspace_path(path: str) -> str:
    """Validate a canonical Workspace-relative POSIX path."""
    if not isinstance(path, str) or not path:
        raise ValueError("workspace path must be a non-empty string")
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("workspace path must be valid UTF-8") from error
    if "\\" in path or "\x00" in path or "//" in path or path.startswith("/"):
        raise ValueError("workspace path must be canonical relative POSIX")
    if path == ".":
        return path
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("workspace path contains a non-canonical component")
    return path


__all__ = [
    "is_workspace_storage_path",
    "validate_workspace_path",
    "workspace_locks_root",
    "workspace_rules_root",
    "workspace_storage_root",
    "workspace_storage_name",
]
