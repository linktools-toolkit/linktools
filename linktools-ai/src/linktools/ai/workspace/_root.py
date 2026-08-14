#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace discovery and identity."""

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import cast

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from ..core import JsonValue, Principal, PrincipalKind, canonical_sha256
from ..errors import AIError, ErrorCode

_STORAGE_DIR_NAME = ".linktools"


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    max_skill_depth: int = 8
    max_concurrency: int = 4
    timeout_seconds: float = 300
    max_calls: int = 100

    def validate(self) -> None:
        if self.max_skill_depth < 0 or self.max_concurrency < 1 or self.timeout_seconds <= 0 or self.max_calls < 1:
            raise ValueError("workspace policy limits must be positive")


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    config: "dict[str, JsonValue]"
    workspace_id: str

    @property
    def storage_root(self) -> Path:
        return self.root / _STORAGE_DIR_NAME

    @classmethod
    def discover(
        cls,
        start: "str | Path",
        *,
        root: "str | Path | None" = None,
    ) -> "Workspace":
        candidate = Path(root).expanduser().resolve() if root is not None else Path(start).expanduser().resolve()
        if root is None and candidate.is_file():
            candidate = candidate.parent
        if root is None:
            for parent in (candidate, *candidate.parents):
                config_file = parent / _STORAGE_DIR_NAME / "config.yaml"
                if config_file.exists():
                    return cls._build(parent, config_file)
        config_file = candidate / _STORAGE_DIR_NAME / "config.yaml"
        return cls._build(candidate, config_file if config_file.exists() else None)

    @classmethod
    def load(cls, root: "str | Path") -> "Workspace":
        candidate = Path(root).expanduser().resolve()
        config_file = candidate / _STORAGE_DIR_NAME / "config.yaml"
        return cls._build(candidate, config_file if config_file.exists() else None)

    @classmethod
    def _build(cls, root: Path, config_file: "Path | None") -> "Workspace":
        normalized_root = _normalized_root(root)
        return cls(
            root=root,
            config=load_config(config_file) if config_file else {},
            workspace_id=canonical_sha256(["workspace", normalized_root]),
        )


def trusted_workspace_principal(workspace_id: str, principal_id: str = "workspace") -> Principal:
    if not workspace_id.strip() or not principal_id.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return Principal(principal_id, workspace_id, PrincipalKind.LOCAL_TRUSTED.value)


def load_config(path: Path) -> "dict[str, JsonValue]":
    if not path.exists() or _yaml is None:
        return {}
    value = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast("dict[str, JsonValue]", value) if isinstance(value, dict) else {}


def _normalized_root(root: Path) -> str:
    return unicodedata.normalize("NFC", root.as_posix())


__all__ = ["Workspace", "WorkspacePolicy", "load_config", "trusted_workspace_principal"]
