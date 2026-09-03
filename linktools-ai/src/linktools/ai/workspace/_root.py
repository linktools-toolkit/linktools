#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace discovery, identity, and immutable policy."""

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml as _yaml

from ..core import JsonValue, Principal, PrincipalKind, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from ._sandbox import Sandbox

_STORAGE_DIR_NAME = ".linktools"

PermissionDecision = Literal["allow", "ask", "deny"]
_PERMISSION_DECISION_RANK: Mapping[PermissionDecision, int] = {
    "allow": 0,
    "ask": 1,
    "deny": 2,
}
_TOOL_PERMISSION_CLASSES = frozenset(
    {
        "control",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "memory.read",
        "memory.write",
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
    max_skill_depth: int = 8
    max_concurrency: int = 4
    timeout_seconds: float = 300
    max_calls: int = 100
    tool_permissions: WorkspaceToolPermissionPolicy = field(
        default_factory=WorkspaceToolPermissionPolicy
    )
    max_repository_instruction_documents: int = 128
    max_repository_instruction_bytes: int = 256 * 1024
    max_preloaded_skill_bytes: int = 256 * 1024

    def validate(self) -> None:
        if (
            self.max_skill_depth < 0
            or self.max_concurrency < 1
            or self.timeout_seconds <= 0
            or self.max_calls < 1
        ):
            raise ValueError("workspace policy limits must be positive")
        if not isinstance(self.tool_permissions, WorkspaceToolPermissionPolicy):
            raise TypeError("workspace tool_permissions must be WorkspaceToolPermissionPolicy")
        limits = (
            self.max_repository_instruction_documents,
            self.max_repository_instruction_bytes,
            self.max_preloaded_skill_bytes,
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
                    return cls._build(parent, config_file, selected_policy, sandbox)
        config_file = candidate / _STORAGE_DIR_NAME / "config.yaml"
        return cls._build(
            candidate,
            config_file if config_file.exists() else None,
            selected_policy,
            sandbox,
        )

    @classmethod
    def load(
        cls,
        root: "str | Path",
        *,
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
        )

    @classmethod
    def _build(
        cls,
        root: Path,
        config_file: "Path | None",
        policy: WorkspacePolicy,
        sandbox: "Sandbox | None",
    ) -> "Workspace":
        normalized_root = _normalized_root(root)
        return cls(
            root=root,
            config=load_config(config_file) if config_file else {},
            workspace_id=canonical_sha256(["workspace", normalized_root]),
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


def _normalized_root(root: Path) -> str:
    return unicodedata.normalize("NFC", root.as_posix())


__all__ = [
    "PermissionDecision",
    "ToolPermissionRule",
    "Workspace",
    "WorkspacePolicy",
    "WorkspaceToolPermissionPolicy",
    "load_config",
    "trusted_workspace_principal",
]
