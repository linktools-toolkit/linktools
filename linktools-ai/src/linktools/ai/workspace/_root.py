#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace paths, discovery, configuration, and immutable policy."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import yaml as _yaml
from linktools.core import environ

from ..core import JsonValue, normalize_json_value
from ..errors import AIError, ErrorCode
from ..spec import (
    DEFAULT_REPOSITORY_INSTRUCTION_BYTES,
    DEFAULT_REPOSITORY_INSTRUCTION_DOCUMENTS,
)

_logger = environ.get_logger("ai.workspace")


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


PermissionDecision = Literal["allow", "ask", "deny"]
_PERMISSION_DECISION_RANK: Mapping[PermissionDecision, int] = {
    "allow": 0,
    "ask": 1,
    "deny": 2,
}
_TOOL_PERMISSION_CLASSES = frozenset(
    {
        "filesystem.read",
        "filesystem.write",
        "shell",
    }
)


@dataclass(frozen=True, slots=True)
class ToolPermissionRule:
    decision: PermissionDecision
    tool_name: "str | None" = None
    tool_class: "str | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, str):
            raise TypeError("tool permission decision must be a string")
        if self.decision not in _PERMISSION_DECISION_RANK:
            raise ValueError("tool permission decision is invalid")
        if (self.tool_name is None) == (self.tool_class is None):
            raise ValueError("tool permission rule requires exactly one selector")
        if self.tool_name is not None:
            if not isinstance(self.tool_name, str):
                raise TypeError("tool permission tool name must be a string")
            if (
                not self.tool_name
                or self.tool_name != self.tool_name.strip()
                or "*" in self.tool_name
            ):
                raise ValueError("tool permission tool name is invalid")
        if self.tool_class is not None:
            if not isinstance(self.tool_class, str):
                raise TypeError("tool permission class must be a string")
            if self.tool_class not in _TOOL_PERMISSION_CLASSES:
                raise ValueError("tool permission class is invalid")


@dataclass(frozen=True, slots=True)
class ToolPermissionPolicy:
    rules: tuple[ToolPermissionRule, ...] = ()
    default: PermissionDecision = "allow"

    def __post_init__(self) -> None:
        if not isinstance(self.default, str):
            raise TypeError("default tool permission decision must be a string")
        if self.default not in _PERMISSION_DECISION_RANK:
            raise ValueError("default tool permission decision is invalid")
        if not isinstance(self.rules, tuple) or any(
            not isinstance(rule, ToolPermissionRule) for rule in self.rules
        ):
            raise TypeError("tool permission rules must be a tuple of ToolPermissionRule")

    @property
    def requires_approval(self) -> bool:
        return self.default == "ask" or any(rule.decision == "ask" for rule in self.rules)

    def decide(
        self,
        *,
        tool_name: str,
        tool_class: "str | None",
    ) -> PermissionDecision:
        if not isinstance(tool_name, str):
            raise TypeError("tool permission tool name must be a string")
        if not tool_name or tool_name != tool_name.strip() or "*" in tool_name:
            raise ValueError("tool permission tool name is invalid")
        if tool_class is not None:
            if not isinstance(tool_class, str):
                raise TypeError("tool permission class must be a string or None")
            if tool_class not in _TOOL_PERMISSION_CLASSES:
                raise ValueError("tool permission class is invalid")
        matched = tuple(
            rule
            for rule in self.rules
            if (
                rule.tool_name is not None
                and rule.tool_name == tool_name
            )
            or (
                rule.tool_class is not None
                and rule.tool_class == tool_class
            )
        )
        if not matched:
            return self.default
        return max(
            matched,
            key=lambda rule: _PERMISSION_DECISION_RANK[rule.decision],
        ).decision


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    tool_permissions: ToolPermissionPolicy = field(
        default_factory=ToolPermissionPolicy
    )
    max_repository_instruction_documents: int = DEFAULT_REPOSITORY_INSTRUCTION_DOCUMENTS
    max_repository_instruction_bytes: int = DEFAULT_REPOSITORY_INSTRUCTION_BYTES

    def validate(self) -> None:
        if not isinstance(self.tool_permissions, ToolPermissionPolicy):
            raise TypeError("workspace tool_permissions must be ToolPermissionPolicy")
        limits = (
            self.max_repository_instruction_documents,
            self.max_repository_instruction_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in limits
        ):
            raise ValueError("workspace instruction limits must be positive integers")


@dataclass(frozen=True, slots=True)
class Workspace:
    STORAGE_DIR_NAME: ClassVar[str] = ".linktools"

    root: Path
    config: "dict[str, JsonValue]"
    policy: WorkspacePolicy = field(default_factory=WorkspacePolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("workspace root must be a Path")
        try:
            root = self.root.expanduser().resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError("workspace root is invalid") from error
        object.__setattr__(self, "root", root)

    @property
    def storage_root(self) -> Path:
        return self.root / self.STORAGE_DIR_NAME

    @property
    def locks_root(self) -> Path:
        return self.storage_root / "locks"

    def is_storage_path(self, path: str) -> bool:
        return path == self.STORAGE_DIR_NAME or path.startswith(self.STORAGE_DIR_NAME + "/")

    def validate_path(self, path: str) -> str:
        """Validate a canonical Workspace-relative POSIX path."""
        return validate_workspace_path(path)

    @classmethod
    def discover(
        cls,
        start: "str | Path",
        *,
        root: "str | Path | None" = None,
        policy: "WorkspacePolicy | None" = None,
    ) -> "Workspace":
        selected_policy = _select_policy(policy)
        candidate = (
            Path(root).expanduser().resolve()
            if root is not None
            else Path(start).expanduser().resolve()
        )
        if root is None and candidate.is_file():
            candidate = candidate.parent
        if root is None:
            for parent in (candidate, *candidate.parents):
                config_file = parent / cls.STORAGE_DIR_NAME / "config.yaml"
                if config_file.exists():
                    return cls._build(
                        parent,
                        config_file,
                        selected_policy,
                    )
        config_file = candidate / cls.STORAGE_DIR_NAME / "config.yaml"
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            selected_policy,
        )

    @classmethod
    def load(
        cls,
        root: "str | Path",
        *,
        policy: "WorkspacePolicy | None" = None,
    ) -> "Workspace":
        candidate = Path(root).expanduser().resolve()
        config_file = candidate / cls.STORAGE_DIR_NAME / "config.yaml"
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            _select_policy(policy),
        )

    @classmethod
    def initialize(
        cls,
        root: "str | Path",
        *,
        policy: "WorkspacePolicy | None" = None,
    ) -> "Workspace":
        """Create the workspace storage directory without persisting identity."""
        candidate = Path(root).expanduser().resolve()
        config_dir = candidate / cls.STORAGE_DIR_NAME
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        if config_file.exists():
            load_config(config_file)
        _logger.info("workspace initialized: root=%s", candidate)
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            _select_policy(policy),
        )

    @classmethod
    def _build(
        cls,
        root: Path,
        config_file: "Path | None",
        policy: WorkspacePolicy,
    ) -> "Workspace":
        config = load_config(config_file) if config_file else {}
        return cls(
            root=root,
            config=config,
            policy=policy,
        )


def load_config(path: Path) -> "dict[str, JsonValue]":
    if not path.exists():
        return {}
    try:
        raw = _yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return {}
        value = normalize_json_value(raw)
        if not isinstance(value, dict):
            raise TypeError("workspace config root must be a mapping")
        return value
    except (_yaml.YAMLError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID) from error


def _select_policy(policy: "WorkspacePolicy | None") -> WorkspacePolicy:
    selected = WorkspacePolicy() if policy is None else policy
    if not isinstance(selected, WorkspacePolicy):
        raise TypeError("policy must be WorkspacePolicy or None")
    selected.validate()
    return selected


__all__ = [
    "PermissionDecision",
    "ToolPermissionRule",
    "Workspace",
    "WorkspacePolicy",
    "ToolPermissionPolicy",
    "load_config",
    "validate_workspace_path",
]
