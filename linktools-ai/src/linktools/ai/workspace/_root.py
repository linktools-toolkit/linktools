#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace discovery, identity, and immutable policy."""

import os
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml as _yaml
from filelock import FileLock
from linktools.core import environ

from ..core import JsonValue, Principal, PrincipalKind, normalize_json_value
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ._sandbox import Sandbox

_STORAGE_DIR_NAME = ".linktools"
_logger = environ.get_logger("ai.workspace")


def normalize_workspace_path(path: str) -> str:
    """Validate one canonical workspace-relative POSIX path."""
    if not isinstance(path, str) or not path:
        raise ValueError("workspace path must be a non-empty string")
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
class WorkspaceToolPermissionPolicy:
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
    tool_permissions: WorkspaceToolPermissionPolicy = field(
        default_factory=WorkspaceToolPermissionPolicy
    )
    max_repository_instruction_documents: int = 128
    max_repository_instruction_bytes: int = 256 * 1024
    max_preloaded_skill_bytes: int = 256 * 1024
    max_binary_input_parts: int = 32
    max_binary_input_bytes: int = 64 * 1024 * 1024

    def validate(self) -> None:
        if not isinstance(self.tool_permissions, WorkspaceToolPermissionPolicy):
            raise TypeError("workspace tool_permissions must be WorkspaceToolPermissionPolicy")
        limits = (
            self.max_repository_instruction_documents,
            self.max_repository_instruction_bytes,
            self.max_preloaded_skill_bytes,
            self.max_binary_input_parts,
            self.max_binary_input_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in limits
        ):
            raise ValueError("workspace instruction limits must be positive integers")


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    config: "dict[str, JsonValue]"
    workspace_id: str
    policy: WorkspacePolicy = field(default_factory=WorkspacePolicy)
    sandbox: "Sandbox | None" = field(default=None, repr=False, compare=False)

    @property
    def storage_root(self) -> Path:
        return self.root / _STORAGE_DIR_NAME

    @classmethod
    def discover(
        cls,
        start: "str | Path",
        *,
        root: "str | Path | None" = None,
        workspace_id: "str | None" = None,
        policy: "WorkspacePolicy | None" = None,
        sandbox: "Sandbox | None" = None,
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
                config_file = parent / _STORAGE_DIR_NAME / "config.yaml"
                if config_file.exists():
                    return cls._build(
                        parent,
                        config_file,
                        selected_policy,
                        sandbox,
                        workspace_id,
                    )
        config_file = candidate / _STORAGE_DIR_NAME / "config.yaml"
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            selected_policy,
            sandbox,
            workspace_id,
        )

    @classmethod
    def load(
        cls,
        root: "str | Path",
        *,
        workspace_id: "str | None" = None,
        policy: "WorkspacePolicy | None" = None,
        sandbox: "Sandbox | None" = None,
    ) -> "Workspace":
        candidate = Path(root).expanduser().resolve()
        config_file = candidate / _STORAGE_DIR_NAME / "config.yaml"
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            _select_policy(policy),
            sandbox,
            workspace_id,
        )

    @classmethod
    def initialize(
        cls,
        root: "str | Path",
        *,
        workspace_id: "str | None" = None,
        policy: "WorkspacePolicy | None" = None,
        sandbox: "Sandbox | None" = None,
    ) -> "Workspace":
        """Create the workspace identity once, then load it without path-derived state."""
        candidate = Path(root).expanduser().resolve()
        config_dir = candidate / _STORAGE_DIR_NAME
        config_file = config_dir / "config.yaml"
        config_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(config_file) + ".lock"):
            if config_file.exists():
                config = load_config(config_file)
                resolved = _configured_workspace_id(config)
                if resolved is None:
                    resolved = _validate_workspace_id(
                        workspace_id
                        if workspace_id is not None
                        else uuid.uuid4().hex
                    )
                    config = dict(config)
                    config["workspace_id"] = resolved
                    _write_config_atomically(config_file, config)
                elif workspace_id is not None and workspace_id != resolved:
                    raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID)
            else:
                resolved = _validate_workspace_id(
                    workspace_id if workspace_id is not None else uuid.uuid4().hex
                )
                config = {"workspace_id": resolved}
                _write_config_atomically(config_file, config)
                _logger.info("workspace initialized: id=%s", resolved)
        return cls._build(
            candidate,
            config_file,
            _select_policy(policy),
            sandbox,
            workspace_id,
        )

    @classmethod
    def _build(
        cls,
        root: Path,
        config_file: "Path | None",
        policy: WorkspacePolicy,
        sandbox: "Sandbox | None",
        workspace_id: "str | None",
    ) -> "Workspace":
        config = load_config(config_file) if config_file else {}
        configured_workspace_id = _configured_workspace_id(config)
        if workspace_id is not None:
            resolved_workspace_id = _validate_workspace_id(workspace_id)
            if (
                configured_workspace_id is not None
                and configured_workspace_id != resolved_workspace_id
            ):
                raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID)
        elif configured_workspace_id is None:
            raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID)
        else:
            resolved_workspace_id = configured_workspace_id
        return cls(
            root=root,
            config=config,
            workspace_id=resolved_workspace_id,
            policy=policy,
            sandbox=sandbox,
        )


def trusted_workspace_principal(
    workspace_id: str,
    principal_id: str = "workspace",
) -> Principal:
    if not workspace_id.strip() or not principal_id.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return Principal(principal_id, workspace_id, PrincipalKind.LOCAL_TRUSTED.value)


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


def _validate_workspace_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID)
    return value


def _configured_workspace_id(config: Mapping[str, JsonValue]) -> "str | None":
    value = config.get("workspace_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AIError(ErrorCode.WORKSPACE_CONFIG_INVALID)
    return value


def _write_config_atomically(path: Path, config: Mapping[str, JsonValue]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            _yaml.safe_dump(dict(config), handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "PermissionDecision",
    "ToolPermissionRule",
    "Workspace",
    "WorkspacePolicy",
    "WorkspaceToolPermissionPolicy",
    "load_config",
    "normalize_workspace_path",
    "trusted_workspace_principal",
]
