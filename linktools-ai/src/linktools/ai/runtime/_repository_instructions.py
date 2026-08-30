#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime integration for repository instructions and workspace tool policy."""

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import ApprovalRequired, ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition

from ..core import canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode
from ..workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructionResolver,
    RepositoryInstructions,
    WorkspacePolicy,
)
from ._capabilities import _validate_trusted_tool_classes
from ._observation import _ObservationalMiddlewareCapability


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

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(_ObservationalMiddlewareCapability,))

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
            target = args.get("path")
            if not isinstance(target, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            subset = await self._instruction_resolver.resolve(
                target,
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
        return args

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
            target_scope = _logical_target_scope(self._workspace_root, target)
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
