#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools-only capability semantics not provided by Harness."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any, Protocol

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.capabilities import AbstractCapability
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
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ..capability import (
    SKILL_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
)
from ..core import (
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
)
from ..errors import AIError, ErrorCode
from ..workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructionResolver,
    RepositoryInstructions,
    WorkspacePolicy,
    normalize_workspace_path,
)
from ._compaction import ExternalModelRequestObserver, RuntimeCompaction
from ._journal import ModelRequestJournal
from ._memory import (
    MemoryOperation,
    MemoryStore,
    memory_operation_fingerprint,
    normalize_memory_file,
)
from ._metric_id import _tool_observation_id
from ._tool import ToolOperationDecision
from .state import ToolOperationRecord

_logger = environ.get_logger("ai.runtime.capabilities")

MEMORY_TOOL_NAMES = ("delete_memory", "read_memory", "search_memory", "write_memory")
MEMORY_READ_TOOL_NAMES = ("read_memory", "search_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
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
_REPOSITORY_MARKER_HEADER = "[linktools.repository-instructions.v1]"
_REPOSITORY_MARKER_ACTION = (
    "Apply the payload as newly applicable repository instructions for this failed "
    "filesystem target, then reconsider the failed tool call."
)
_WORKSPACE_SCOPED_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "file_info",
        "create_directory",
        "list_directory",
        "search_files",
        "find_files",
    }
)
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

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool: ...

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None: ...

    async def existing_call_ids(
        self,
        tool_call_ids: Sequence[str],
    ) -> frozenset[str]: ...

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]: ...


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

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
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


class _WorkspaceToolGate(AbstractCapability[None]):
    def __init__(
        self,
        *,
        execution_id: str,
        workspace_root: Path,
        repository_instruction_history: tuple[ModelMessage, ...],
        repository_instruction_marker_authority: frozenset[tuple[str, str]],
        repository_instructions: RepositoryInstructions | None,
        instruction_resolver: RepositoryInstructionResolver,
        policy: WorkspacePolicy,
        trusted_tool_classes: tuple[tuple[str, str], ...],
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(workspace_root, Path):
            raise TypeError("workspace_root must be Path")
        if not isinstance(policy, WorkspacePolicy):
            raise TypeError("policy must be WorkspacePolicy")
        if not isinstance(repository_instruction_history, tuple) or any(
            not isinstance(message, (ModelRequest, ModelResponse))
            for message in repository_instruction_history
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(repository_instruction_marker_authority, frozenset):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for value in repository_instruction_marker_authority:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_trusted_tool_classes(trusted_tool_classes)
        trusted = dict(trusted_tool_classes)
        if len(trusted) != len(trusted_tool_classes):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._execution_id = execution_id
        self._workspace_root = workspace_root
        self._instruction_resolver = instruction_resolver
        self._policy = policy
        self._trusted_tool_class_by_name = trusted
        self._repository_instructions_enabled = repository_instructions is not None
        self._exposure_map: dict[str, RepositoryInstructionDocument] = {}
        if repository_instructions is not None:
            self._exposure_map.update(
                (document.source, document) for document in repository_instructions.documents
            )
        self._marker_authority = repository_instruction_marker_authority
        self._refresh_required = False
        if self._repository_instructions_enabled:
            self._restore_exposure_map(repository_instruction_history)
            self._validate_active_limits()

    def get_instructions(self) -> Callable[[RunContext[None]], str]:
        def active_repository_instructions(_ctx: RunContext[None]) -> str:
            if not self._repository_instructions_enabled:
                return ""
            return RepositoryInstructions(tuple(self._exposure_map.values())).render()

        return active_repository_instructions

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        self._refresh_required = False
        return request_context

    async def before_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_args = _normalize_workspace_tool_args(tool_def.name, args)
        del call
        if self._refresh_required:
            raise ToolFailed(
                "Repository instructions changed; reconsider this tool call on the next model step."
            )
        tool_class = self._trusted_tool_class_by_name.get(tool_def.name)
        decision = self._policy.tool_permissions.decide(
            tool_name=tool_def.name,
            tool_class=tool_class,
        )
        if decision == "deny":
            raise ToolFailed("Tool execution is denied by the current workspace policy.")
        if (
            self._repository_instructions_enabled
            and tool_def.name in _WORKSPACE_SCOPED_TOOL_NAMES
            and tool_class in {"filesystem.read", "filesystem.write"}
        ):
            target = normalized_args["path"]
            subset = await self._instruction_resolver.resolve(
                _repository_instruction_lookup_target(self._workspace_root, target),
                exclude_sources=frozenset(self._exposure_map),
            )
            if any(document.source in self._exposure_map for document in subset.documents):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if subset.documents:
                candidate = RepositoryInstructions(
                    (*tuple(self._exposure_map.values()), *subset.documents)
                )
                self._validate_bundle_limits(candidate)
                marker = _repository_instruction_marker(self._execution_id, subset)
                if len(marker.encode("utf-8")) > self._policy.max_repository_instruction_bytes:
                    raise AIError(ErrorCode.PROMPT_TOO_LARGE)
                self._exposure_map = {
                    document.source: document for document in candidate.documents
                }
                self._refresh_required = True
                raise ToolFailed(marker)
        if decision == "ask" and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return normalized_args

    def _restore_exposure_map(self, messages: tuple[ModelMessage, ...]) -> None:
        calls: dict[tuple[str, str], list[ToolCallPart]] = {}
        for message in messages:
            run_id = message.run_id
            if isinstance(message, ModelResponse):
                if not isinstance(run_id, str) or not run_id:
                    continue
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        calls.setdefault((run_id, part.tool_call_id), []).append(part)
                continue
            if not isinstance(message, ModelRequest):
                continue
            if not isinstance(run_id, str) or not run_id:
                continue
            for part in message.parts:
                if not isinstance(part, ToolReturnPart) or part.outcome != "failed":
                    continue
                authority = (run_id, part.tool_call_id)
                if authority not in self._marker_authority:
                    continue
                content = part.content
                if not isinstance(content, str) or not content.startswith(_REPOSITORY_MARKER_HEADER):
                    continue
                current_execution = _marker_execution_id(content)
                if current_execution != self._execution_id:
                    continue
                paired = calls.get(authority, ())
                if len(paired) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                subset = self._parse_repository_instruction_marker(run_id, paired[0], part)
                if subset is None:
                    continue
                for document in subset.documents:
                    existing = self._exposure_map.get(document.source)
                    if existing is None:
                        self._exposure_map[document.source] = document
                    elif existing != document:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _parse_repository_instruction_marker(
        self,
        run_id: str,
        call: ToolCallPart,
        result: ToolReturnPart,
    ) -> RepositoryInstructions | None:
        if (run_id, result.tool_call_id) not in self._marker_authority:
            return None
        content = result.content
        if result.outcome != "failed" or not isinstance(content, str):
            return None
        if not content.startswith(_REPOSITORY_MARKER_HEADER):
            return None
        execution_id = _marker_execution_id(content)
        if execution_id != self._execution_id:
            return None
        try:
            lines = content.split("\n")
            if len(lines) != 5:
                raise ValueError("repository marker line count is invalid")
            if lines[0] != _REPOSITORY_MARKER_HEADER:
                raise ValueError("repository marker header is invalid")
            if lines[1] != f"execution_id={self._execution_id}":
                raise ValueError("repository marker execution is invalid")
            digest_line = lines[2]
            if not digest_line.startswith("digest="):
                raise ValueError("repository marker digest is missing")
            digest = digest_line.removeprefix("digest=")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("repository marker digest is invalid")
            if lines[3] != f"action={_REPOSITORY_MARKER_ACTION}":
                raise ValueError("repository marker action is invalid")
            if not lines[4].startswith("payload="):
                raise ValueError("repository marker payload is missing")
            payload_text = lines[4].removeprefix("payload=")
            raw = json.loads(payload_text)
            normalized = normalize_json_value(raw)
            if not isinstance(normalized, Mapping):
                raise ValueError("repository marker payload is not a mapping")
            if canonical_json_bytes(normalized).decode("utf-8") != payload_text:
                raise ValueError("repository marker payload is not canonical")
            instructions = RepositoryInstructions.from_payload(normalized)
            if instructions.digest != digest:
                raise ValueError("repository marker digest mismatch")
            tool_class = self._trusted_tool_class_by_name.get(call.tool_name)
            if (
                call.tool_name not in _WORKSPACE_SCOPED_TOOL_NAMES
                or tool_class not in {"filesystem.read", "filesystem.write"}
                or call.tool_call_id != result.tool_call_id
                or call.tool_name != result.tool_name
            ):
                raise ValueError("repository marker tool provenance is invalid")
            arguments = call.args_as_dict()
            target = arguments.get("path")
            if not isinstance(target, str):
                raise ValueError("repository marker target is invalid")
            target_scope = _logical_target_scope(
                self._workspace_root,
                _repository_instruction_target(target),
            )
            if any(
                not _scope_applies_to_target(document.scope, target_scope)
                for document in instructions.documents
            ):
                raise ValueError("repository marker scope is invalid")
            return instructions
        except AIError as error:
            if error.code in {
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT,
            }:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise
        except (json.JSONDecodeError, TypeError, ValueError, UnicodeError, OSError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_active_limits(self) -> None:
        self._validate_bundle_limits(
            RepositoryInstructions(tuple(self._exposure_map.values()))
        )

    def _validate_bundle_limits(self, instructions: RepositoryInstructions) -> None:
        if len(instructions.documents) > self._policy.max_repository_instruction_documents:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)
        if len(instructions.render().encode("utf-8")) > self._policy.max_repository_instruction_bytes:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)


def _normalize_workspace_tool_args(
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    if tool_name not in _WORKSPACE_SCOPED_TOOL_NAMES:
        return args
    target = args.get("path")
    if not isinstance(target, str):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    normalized = normalize_workspace_path(target)
    if normalized == target:
        return args
    result = dict(args)
    result["path"] = normalized
    return result


def _repository_instruction_marker(
    execution_id: str,
    subset: RepositoryInstructions,
) -> str:
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return (
        _REPOSITORY_MARKER_HEADER
        + "\n"
        + f"execution_id={execution_id}\n"
        + f"digest={subset.digest}\n"
        + f"action={_REPOSITORY_MARKER_ACTION}\n"
        + "payload="
        + canonical_json_bytes(subset.to_payload()).decode("utf-8")
    )


def _marker_execution_id(content: str) -> str | None:
    if not isinstance(content, str):
        return None
    lines = content.split("\n", 2)
    if len(lines) < 2 or lines[0] != _REPOSITORY_MARKER_HEADER:
        return None
    execution_line = lines[1]
    prefix = "execution_id="
    if not execution_line.startswith(prefix):
        return None
    execution_id = execution_line.removeprefix(prefix)
    if not execution_id or execution_id != execution_id.strip():
        return None
    return execution_id


def _repository_instruction_target(target: str) -> str:
    return "." if target == "" else target


def _repository_instruction_lookup_target(root: Path, target: str) -> str:
    lookup_target = _repository_instruction_target(target)
    try:
        _logical_target_scope(root, lookup_target)
    except (OSError, ValueError) as error:
        raise ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                "workspace path is invalid or outside the workspace; use a path within the workspace root and retry",
            )
        ) from error
    return lookup_target


def _logical_target_scope(root: Path, target: str) -> str:
    raw = os.fspath(target)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("repository marker target is invalid")
    root_value = os.fspath(root)
    if os.path.isabs(raw):
        normalized = Path(os.path.abspath(os.path.normpath(raw)))
    else:
        normalized = Path(
            os.path.abspath(os.path.normpath(os.path.join(root_value, raw)))
        )
    try:
        relative = normalized.relative_to(root)
    except (ValueError, OSError) as error:
        raise ValueError("repository marker target is outside workspace") from error
    scope = relative.as_posix()
    if scope in {"", "."}:
        return "."
    if "\\" in scope or "\x00" in scope:
        raise ValueError("repository marker target scope is invalid")
    parts = scope.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("repository marker target scope is invalid")
    return scope


def _scope_applies_to_target(scope: str, target: str) -> bool:
    if scope == ".":
        return True
    scope_parts = scope.split("/")
    target_parts = [] if target == "." else target.split("/")
    return (
        len(target_parts) >= len(scope_parts)
        and target_parts[: len(scope_parts)] == scope_parts
    )


class _SelectedMemory(AbstractCapability[None]):
    def __init__(
        self,
        store: MemoryStore,
        *,
        selected_tool_names: tuple[str, ...],
        id: str,
        operation_identity_run_id: str | None = None,
    ) -> None:
        self.id = id
        self._store = store
        self._selected_tool_names = selected_tool_names
        self._operation_identity_run_id = operation_identity_run_id

    def get_instructions(self) -> str:
        return _memory_guidance(self._selected_tool_names)

    def get_toolset(self) -> AbstractToolset[None]:
        toolset = FunctionToolset(id="memory")
        if "read_memory" in self._selected_tool_names:
            toolset.add_function(self._read_memory, name="read_memory")
        if "search_memory" in self._selected_tool_names:
            toolset.add_function(self._search_memory, name="search_memory")
        if "write_memory" in self._selected_tool_names:
            toolset.add_function(self._write_memory, name="write_memory")
        if "delete_memory" in self._selected_tool_names:
            toolset.add_function(self._delete_memory, name="delete_memory")
        return toolset

    async def _read_memory(self, ctx: RunContext[None], file: str) -> str:
        del ctx
        normalized = _memory_file_argument(file)
        result = await self._store.read(normalized, max_chars=65_536)
        if result is None:
            raise ModelRetry(
                f"There is no memory file named {normalized!r}; search memory first."
            )
        suffix = "\n[truncated]" if result.truncated else ""
        return result.content + suffix

    async def _search_memory(
        self,
        ctx: RunContext[None],
        query: str,
    ) -> dict[str, object]:
        del ctx
        result = await self._store.search(query, limit=10)
        return {
            "matches": [
                {"file": match.file, "snippet": match.snippet, "score": match.score}
                for match in result.matches
            ],
            "scanned": result.scanned,
            "truncated": result.truncated,
        }

    async def _write_memory(
        self,
        ctx: RunContext[None],
        content: str,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> dict[str, object]:
        normalized = _memory_file_argument(file)
        if old_text is None and not content.strip():
            raise ModelRetry("Nothing to write; provide content to append.")
        operation = _memory_operation(
            ctx,
            "write",
            normalized,
            content,
            old_text,
            operation_identity_run_id=self._operation_identity_run_id,
        )
        replay = await self._store.get_operation(operation)
        if replay is not None:
            return {
                "file": replay.file,
                "version": replay.version,
                "status": replay.status,
            }
        current = await self._store.read(normalized, max_chars=65_536)
        if current is not None and current.truncated:
            raise ModelRetry("The memory file is too large to edit safely.")
        value = "" if current is None else current.content
        if old_text is None:
            next_content = content
            append = True
            status = "created" if current is None else "appended"
        else:
            if not old_text or value.count(old_text) != 1:
                raise ModelRetry("old_text must match exactly one existing passage.")
            next_content = value.replace(old_text, content)
            append = False
            status = "updated"
        mutation = await self._store.write(
            normalized,
            content if append else next_content,
            expected_version=None if current is None else current.version,
            operation=operation,
            append=append,
        )
        return {
            "file": normalized,
            "version": mutation.version,
            "status": status if not mutation.replayed else mutation.status,
        }

    async def _delete_memory(
        self,
        ctx: RunContext[None],
        file: str,
    ) -> dict[str, JsonValue]:
        normalized = _memory_file_argument(file)
        if normalized == "MEMORY.md":
            raise ModelRetry("MEMORY.md is the main notebook; edit it instead.")
        operation = _memory_operation(
            ctx,
            "delete",
            normalized,
            operation_identity_run_id=self._operation_identity_run_id,
        )
        replay = await self._store.get_operation(operation)
        if replay is not None:
            return {
                "file": replay.file,
                "version": replay.version,
                "status": replay.status,
            }
        current = await self._store.read(normalized, max_chars=1)
        mutation = await self._store.delete(
            normalized,
            expected_version=None if current is None else current.version,
            operation=operation,
        )
        return {
            "file": normalized,
            "version": mutation.version,
            "status": mutation.status,
        }


class _CompactionCapability(AbstractCapability[None]):
    def __init__(
        self,
        target_tokens: int | None,
        *,
        trusted_workspace_read: bool,
        journal: ModelRequestJournal | None,
        observer: ExternalModelRequestObserver | None,
        projection_sink: (
            Callable[[Sequence[ModelMessage], Sequence[ModelMessage] | None], None]
            | None
        ),
    ) -> None:
        if not isinstance(trusted_workspace_read, bool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._compaction = RuntimeCompaction(
            target_tokens,
            trusted_workspace_read=trusted_workspace_read,
            journal=journal,
            observer=observer,
            projection_sink=projection_sink,
        )

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self._compaction.before_model_request(ctx, request_context)


def _memory_guidance(selected_tools: tuple[str, ...]) -> str:
    actions = ", ".join(f"`{name}`" for name in selected_tools)
    return f"Use only these memory tools when needed: {actions}."


def _memory_file_argument(file: str) -> str:
    if not isinstance(file, str):
        raise ModelRetry("memory file must be a string")
    try:
        return normalize_memory_file(file)
    except AIError as error:
        raise ModelRetry("memory file name is invalid") from error


def _memory_operation(
    ctx: RunContext[None],
    kind: str,
    file: str,
    content: str | None = None,
    old_text: str | None = None,
    *,
    operation_identity_run_id: str | None = None,
) -> MemoryOperation:
    call_id = _tool_call_identity(
        ctx,
        operation_identity_run_id=operation_identity_run_id,
    )
    append = kind == "write" and old_text is None
    fingerprint = memory_operation_fingerprint(
        kind,
        file,
        content,
        old_text,
        append,
    )
    return MemoryOperation(
        call_id,
        fingerprint,
        kind,
        file,
        content,
        old_text,
        append,
    )


def _tool_call_identity(
    ctx: RunContext[None],
    *,
    operation_identity_run_id: str | None,
) -> str:
    run_id = operation_identity_run_id or ctx.run_id
    call_id = ctx.tool_call_id
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(call_id, str)
        or not call_id
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return canonical_sha256({"run_id": run_id, "tool_call_id": call_id})


def _model_usage_metadata(response: ModelResponse) -> dict[str, str]:
    usage = response.usage
    return {
        _MODEL_USAGE_INPUT_METADATA_KEY: _model_usage_token(usage.input_tokens),
        _MODEL_USAGE_OUTPUT_METADATA_KEY: _model_usage_token(usage.output_tokens),
        _MODEL_USAGE_CACHE_READ_METADATA_KEY: _model_usage_token(usage.cache_read_tokens),
        _MODEL_USAGE_CACHE_WRITE_METADATA_KEY: _model_usage_token(usage.cache_write_tokens),
    }


def _model_usage_token(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AIError(ErrorCode.MODEL_RESPONSE_INVALID)
    return str(value)


def _validate_compaction_target(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


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
        if not selector.startswith("mcp__") or selector.endswith("__") or "*" in selector:
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
        return _WORKSPACE_SANDBOX_CAPABILITY_ID if name in WORKSPACE_SHELL_TOOL_NAMES else None
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
    _validate_trusted_tool_classes(trusted_tool_classes)
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
    _validate_trusted_tool_classes(trusted_tool_classes)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class != "control":
        return False
    expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
    return expected_capability is not None and tool_def.capability_id == expected_capability


def tool_allowed_in_planning(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: tuple[tuple[str, str], ...],
    trusted_mcp_selectors: tuple[str, ...],
) -> bool:
    _validate_trusted_tool_classes(trusted_tool_classes)
    _validate_trusted_mcp_selectors(trusted_mcp_selectors)
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
    if any(tool_def.name.startswith(f"{selector}__") for selector in trusted_mcp_selectors):
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
