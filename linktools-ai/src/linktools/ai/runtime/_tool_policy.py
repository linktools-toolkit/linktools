#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime tool selection, provenance, replay, and failure policy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext, ToolDefinition

from ..capability import (
    SKILL_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
)
from ..errors import AIError, ErrorCode
from ._tool import ToolOperationDecision
from .state._contracts import ToolOperationRecord

MEMORY_TOOL_NAMES = ("delete_memory", "read_memory", "search_memory", "write_memory")
MEMORY_READ_TOOL_NAMES = ("read_memory", "search_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
PYDANTIC_CONTROL_TOOL_KINDS = frozenset({"capability-load", "tool-search"})
PLAN_SAFE_METADATA_KEY = "linktools.ai.plan_safe"
_REPLAY_SAFE_METADATA_KEY = "linktools.ai.replay_safe"
_MODEL_USAGE_INPUT_METADATA_KEY = "linktools.ai.model_usage.input_tokens"
_MODEL_USAGE_OUTPUT_METADATA_KEY = "linktools.ai.model_usage.output_tokens"
_MODEL_USAGE_CACHE_READ_METADATA_KEY = "linktools.ai.model_usage.cache_read_tokens"
_MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"
_OBSERVATION_ID_METADATA_KEY = "linktools.ai.observation_id"
_DURATION_NS_METADATA_KEY = "linktools.ai.duration_ns"
_MODEL_TOOL_ERROR_MAX_CHARS = 4096
_MODEL_TOOL_ERROR_HEAD_CHARS = 1024
_MODEL_TOOL_ERROR_TRUNCATION_MARKER = "...[truncated]..."
_MODEL_EFFECT_UNKNOWN_MESSAGE = "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
_MODEL_RETRY_PREFIX = "TOOL_RETRY_REQUIRED"
_MODEL_FAILED_PREFIX = "TOOL_EXECUTION_FAILED"
_TRUSTED_TOOL_CLASSES = frozenset(
    {
        "control",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "memory.read",
        "memory.write",
    }
)
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
_SKILL_CAPABILITY_ID = "linktools-skill"
_MEMORY_CAPABILITY_ID = "linktools-memory"
_PLANNING_CAPABILITY_ID = "linktools-planning"
_SUBAGENT_CAPABILITY_ID = "linktools-subagent"


@dataclass(frozen=True, slots=True)
class _ToolExecutionPolicy:
    replay_safe: bool
    effect_free: bool


@dataclass
class _ToolCallState:
    decision: ToolOperationDecision
    policy: _ToolExecutionPolicy
    handler_entered: bool = False
    handler_observed: bool = False
    heartbeat_observed: bool = False
    operation_terminalized: bool = False
    preserve_started: bool = False
    cached_failure: bool = False
    terminal_event_recorded: bool = False
    suppress_cancel_metric: bool = False
    metric_started_ns: int | None = None
    heartbeat_task: asyncio.Task[None] | None = None


class ToolOperationBridge(Protocol):
    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision: ...

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision: ...

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool: ...

    async def fail(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> bool: ...

    async def unknown(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> None: ...

    async def existing_call_ids(
        self,
        tool_call_ids: Sequence[str],
    ) -> frozenset[str]: ...

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]: ...

    async def effective_args(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]: ...


class _MissingToolOperationBridge:
    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args, replay_safe
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        del decision
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def unknown(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> None:
        del decision, error
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def existing_call_ids(
        self,
        tool_call_ids: Sequence[str],
    ) -> frozenset[str]:
        del tool_call_ids
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def effective_args(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del ctx, call, tool_def
        return args


def _model_usage_metadata(response: ModelResponse) -> dict[str, str]:
    usage = response.usage
    return {
        _MODEL_USAGE_INPUT_METADATA_KEY: _model_usage_token(usage.input_tokens),
        _MODEL_USAGE_OUTPUT_METADATA_KEY: _model_usage_token(usage.output_tokens),
        _MODEL_USAGE_CACHE_READ_METADATA_KEY: _model_usage_token(
            usage.cache_read_tokens
        ),
        _MODEL_USAGE_CACHE_WRITE_METADATA_KEY: _model_usage_token(
            usage.cache_write_tokens
        ),
    }


def _model_usage_token(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AIError(ErrorCode.MODEL_RESPONSE_INVALID)
    return str(value)


def _validate_trusted_tool_classes(value: tuple[tuple[str, str], ...]) -> None:
    seen: set[str] = set()
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[0]
            or item[0] in seen
            or item[1] not in _TRUSTED_TOOL_CLASSES
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        seen.add(item[0])


def _validate_trusted_mcp_selectors(value: tuple[str, ...]) -> None:
    if len(set(value)) != len(value):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    for selector in value:
        if (
            not selector.startswith("mcp__")
            or selector.endswith("__")
            or "*" in selector
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def tool_name_allowed(name: str, allow_tools: tuple[str, ...]) -> bool:
    return "*" in allow_tools or name in allow_tools


def _trusted_tool_capability(name: str, tool_class: str) -> str | None:
    if tool_class == "control":
        if name in SKILL_TOOL_NAMES:
            return _SKILL_CAPABILITY_ID
        if name in PLANNING_TOOL_NAMES:
            return _PLANNING_CAPABILITY_ID
        if name in SUBAGENT_TOOL_NAMES:
            return _SUBAGENT_CAPABILITY_ID
        return None
    if tool_class in {"filesystem.read", "filesystem.write"}:
        if name not in WORKSPACE_FILESYSTEM_TOOL_NAMES:
            return None
        is_read = name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        if is_read != (tool_class == "filesystem.read"):
            return None
        return _WORKSPACE_SANDBOX_CAPABILITY_ID
    if tool_class == "shell":
        return (
            _WORKSPACE_SANDBOX_CAPABILITY_ID
            if name in WORKSPACE_SHELL_TOOL_NAMES
            else None
        )
    if tool_class in {"memory.read", "memory.write"}:
        if name not in MEMORY_TOOL_NAMES:
            return None
        is_read = name in MEMORY_READ_TOOL_NAMES
        if is_read != (tool_class == "memory.read"):
            return None
        return _MEMORY_CAPABILITY_ID
    return None


def _tool_execution_policy(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: tuple[tuple[str, str], ...],
) -> _ToolExecutionPolicy:
    if tool_def.tool_kind in PYDANTIC_CONTROL_TOOL_KINDS:
        return _ToolExecutionPolicy(True, True)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class is not None:
        expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
        if expected_capability is None or tool_def.capability_id != expected_capability:
            raise AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={"tool_name": tool_def.name},
            )
        if tool_class in {"filesystem.read", "memory.read"}:
            return _ToolExecutionPolicy(True, True)
        if tool_class == "memory.write":
            return _ToolExecutionPolicy(True, False)
        if tool_class == "filesystem.write":
            return _ToolExecutionPolicy(False, False)
        if tool_class == "shell":
            return _ToolExecutionPolicy(
                tool_def.name == "check_command",
                tool_def.name == "check_command",
            )
        if tool_class == "control":
            if tool_def.name in SKILL_TOOL_NAMES:
                return _ToolExecutionPolicy(True, True)
            if tool_def.name in PLANNING_TOOL_NAMES or tool_def.name == "delegate_task":
                return _ToolExecutionPolicy(True, False)
            if tool_def.name == "list_subagents":
                return _ToolExecutionPolicy(True, True)
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    metadata = (tool_def.metadata or {}).get(_REPLAY_SAFE_METADATA_KEY, False)
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return _ToolExecutionPolicy(metadata, False)


def _durable_failure_error(
    error: Exception,
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> Exception:
    if isinstance(error, (ToolRetryError, ToolFailedError, AIError)):
        return error
    if isinstance(error, (ValidationError, ModelRetry)):
        return ToolRetryError(
            RetryPromptPart.from_error(
                error,
                tool_name=tool_def.name,
                tool_call_id=call.tool_call_id,
            )
        )
    if isinstance(error, ToolFailed):
        return ToolFailedError(
            ToolReturnPart(
                tool_def.name,
                error.message,
                tool_call_id=call.tool_call_id,
                outcome="failed",
            )
        )
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _model_tool_error(
    error: BaseException,
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> BaseException:
    if isinstance(error, ValidationError):
        content = RetryPromptPart.from_error(
            error,
            tool_name=tool_def.name,
            tool_call_id=call.tool_call_id,
        ).content
        return ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                _model_tool_error_content(content, "correct the call and retry"),
            )
        )
    if isinstance(error, ModelRetry):
        return ModelRetry(_format_model_tool_error(_MODEL_RETRY_PREFIX, error.message))
    if isinstance(error, ToolRetryError):
        return ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                _model_tool_error_content(
                    error.tool_retry.content,
                    "correct the call and retry",
                ),
            )
        )
    if isinstance(error, ToolFailed):
        return ToolFailed(_format_model_tool_error(_MODEL_FAILED_PREFIX, error.message))
    if isinstance(error, ToolFailedError):
        return ToolFailed(
            _format_model_tool_error(
                _MODEL_FAILED_PREFIX,
                _model_tool_error_content(
                    error.tool_failed.content,
                    "adapt and continue",
                ),
            )
        )
    return error


def _model_tool_error_content(content: object, fallback: str) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return fallback


def _format_model_tool_error(prefix: str, message: str) -> str:
    value = f"{prefix}: {message}"
    if len(value) <= _MODEL_TOOL_ERROR_MAX_CHARS:
        return value
    tail_chars = (
        _MODEL_TOOL_ERROR_MAX_CHARS
        - _MODEL_TOOL_ERROR_HEAD_CHARS
        - len(_MODEL_TOOL_ERROR_TRUNCATION_MARKER)
    )
    return (
        value[:_MODEL_TOOL_ERROR_HEAD_CHARS]
        + _MODEL_TOOL_ERROR_TRUNCATION_MARKER
        + value[-tail_chars:]
    )


def _bypasses_tool_error_hook(error: BaseException) -> bool:
    return isinstance(
        error,
        (
            ModelRetry,
            ToolRetryError,
            ToolFailed,
            ToolFailedError,
            SkipToolExecution,
            CallDeferred,
            ApprovalRequired,
        ),
    )


def tool_is_control(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: tuple[tuple[str, str], ...],
) -> bool:
    if tool_def.tool_kind in PYDANTIC_CONTROL_TOOL_KINDS:
        return True
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class != "control":
        return False
    expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
    return (
        expected_capability is not None
        and tool_def.capability_id == expected_capability
    )


def tool_allowed_in_planning(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: tuple[tuple[str, str], ...],
    trusted_mcp_selectors: tuple[str, ...],
) -> bool:
    if tool_def.tool_kind in PYDANTIC_CONTROL_TOOL_KINDS:
        return True
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class is not None:
        expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
        if expected_capability is None:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        if tool_def.capability_id == expected_capability:
            return tool_class in {"control", "filesystem.read", "memory.read"}
    if tool_def.capability_id in trusted_mcp_selectors:
        if not tool_def.name.startswith(f"{tool_def.capability_id}__"):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        return False
    if any(
        tool_def.name.startswith(f"{selector}__") for selector in trusted_mcp_selectors
    ):
        return False
    metadata = (tool_def.metadata or {}).get(PLAN_SAFE_METADATA_KEY)
    if metadata is None:
        return False
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return metadata


def select_runtime_tool_names(
    *,
    ordinary_tool_policy: tuple[str, ...],
    memory_scope: str | None,
    subagent_available: bool = False,
    planning: bool = False,
) -> tuple[str, ...]:
    names: set[str] = set()
    if memory_scope is not None:
        names.update(
            name
            for name in MEMORY_TOOL_NAMES
            if tool_name_allowed(name, ordinary_tool_policy)
        )
    if planning:
        names.update(PLANNING_TOOL_NAMES)
    if subagent_available:
        names.update(SUBAGENT_TOOL_NAMES)
    return tuple(sorted(names))


__all__ = [
    "MEMORY_READ_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "PLANNING_TOOL_NAMES",
    "PLAN_SAFE_METADATA_KEY",
    "SUBAGENT_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "ToolOperationBridge",
    "ToolOperationDecision",
    "select_runtime_tool_names",
    "tool_allowed_in_planning",
    "tool_is_control",
    "tool_name_allowed",
]
