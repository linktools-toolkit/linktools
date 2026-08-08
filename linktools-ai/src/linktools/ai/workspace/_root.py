#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace discovery and identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast
import unicodedata

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from ..core import Principal
from ..core.errors import ErrorCode, AIError
from ..core.ids import canonical_sha256
from ..core.json import JsonValue


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
    storage_root: Path
    config_path: "Path | None"
    config: "dict[str, JsonValue]"
    workspace_id: str

    @classmethod
    def discover(
        cls,
        start: "str | Path",
        *,
        root: "str | Path | None" = None,
        storage_root: "str | Path | None" = None,
    ) -> "Workspace":
        candidate = Path(root).expanduser().resolve() if root is not None else Path(start).expanduser().resolve()
        if root is None and candidate.is_file():
            candidate = candidate.parent
        if root is None:
            for parent in (candidate, *candidate.parents):
                config_path = parent / ".linktools" / "config.yaml"
                if config_path.exists():
                    return cls._build(parent, config_path, storage_root)
        config_path = candidate / ".linktools" / "config.yaml"
        return cls._build(candidate, config_path if config_path.exists() else None, storage_root)

    @classmethod
    def load(cls, root: "str | Path", *, storage_root: "str | Path | None" = None) -> "Workspace":
        candidate = Path(root).expanduser().resolve()
        config_path = candidate / ".linktools" / "config.yaml"
        return cls._build(candidate, config_path if config_path.exists() else None, storage_root)

    @classmethod
    def _build(cls, root: Path, config_path: "Path | None", storage_root: "str | Path | None") -> "Workspace":
        normalized_root = _normalized_root(root)
        resolved_storage_root = root if storage_root is None else Path(storage_root).expanduser().resolve()
        return cls(
            root=root,
            storage_root=resolved_storage_root,
            config_path=config_path,
            config=load_config(config_path) if config_path else {},
            workspace_id=canonical_sha256(["workspace", normalized_root]),
        )


def trusted_workspace_principal(workspace_id: str, principal_id: str = "workspace") -> Principal:
    if not workspace_id.strip() or not principal_id.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return Principal(principal_id, workspace_id, "LOCAL_TRUSTED")


def load_config(path: Path) -> "dict[str, JsonValue]":
    if not path.exists() or _yaml is None:
        return {}
    value = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast("dict[str, JsonValue]", value) if isinstance(value, dict) else {}


def _normalized_root(root: Path) -> str:
    return unicodedata.normalize("NFC", root.as_posix())


__all__ = ["Workspace", "WorkspacePolicy", "load_config", "trusted_workspace_principal"]
