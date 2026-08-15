#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped Pydantic AI infrastructure capabilities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, WrapToolExecuteHandler
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability
from pydantic_ai.exceptions import ModelRetry, ToolRetryError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import Memory, SearchableMemoryStore
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore

from ..capability import SKILL_TOOL_NAMES
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.agent.capabilities")

MEMORY_TOOL_NAMES = ("delete_memory", "read_memory", "search_memory", "write_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
SUBAGENT_TOOL_NAMES = ("delegate_task",)
WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    "create_directory", "edit_file", "file_info", "find_files", "list_directory", "read_file", "search_files", "write_file",
)
WORKSPACE_SHELL_TOOL_NAMES = ("check_command", "run_command", "start_command", "stop_command")


class _RetryAwareStepPersistence(StepPersistence[None]):
    async def wrap_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        except (ModelRetry, ToolRetryError) as error:
            _logger.debug(
                "tool effect marked failed: run=%s tool=%s call=%s",
                self.run_id or ctx.run_id,
                tool_def.name,
                call.tool_call_id,
            )
            await super().on_tool_execute_error(ctx, call=call, tool_def=tool_def, args=args, error=error)
            raise


@dataclass(frozen=True, slots=True)
class AgentRunScope:
    root: Path
    agent_name: str
    conversation_id: "str | None"
    step_run_id: str
    segment_sequence: "int | None"
    memory_scope: "str | None"
    step_store: StepStore
    memory_store: "SearchableMemoryStore | None"
    platform_tool_names: "tuple[str, ...]" = ()
    context_target_tokens: "int | None" = None
    parent_step_run_id: "str | None" = None
    subagent_delegate: "SubagentDelegate | None" = None


class SubagentDelegate(Protocol):
    async def __call__(self, agent_id: str, user_prompt: str, *, tool_call_id: str) -> JsonValue: ...


async def compose_platform_capabilities(
    scope: AgentRunScope,
    *,
    model_factory: Callable[["str | Model | None"], "str | Model"],
    parent_model: "str | Model",
) -> "tuple[PydanticAgentCapability[None], ...]":
    del model_factory, parent_model
    _validate_compaction_target(scope.context_target_tokens)
    capabilities: list[PydanticAgentCapability[None]] = [
        _RetryAwareStepPersistence(
            store=scope.step_store,
            agent_name=scope.agent_name,
            run_id=scope.step_run_id,
            parent_run_id=scope.parent_step_run_id,
            metadata={
                "capability_scope": "parent",
                "agent_name": scope.agent_name,
                **({} if scope.segment_sequence is None else {"segment_sequence": str(scope.segment_sequence)}),
            },
        )
    ]
    selected = frozenset(scope.platform_tool_names)
    selected_memory = tuple(name for name in MEMORY_TOOL_NAMES if name in selected)
    if selected_memory:
        if scope.memory_store is None or scope.memory_scope is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            _SelectedMemory(
                store=scope.memory_store,
                namespace=scope.memory_scope,
                agent_name="memory",
                inject_memory=False,
                guidance=_memory_guidance(selected_memory),
                selected_tool_names=selected_memory,
            )
        )
    if any(name in selected for name in PLANNING_TOOL_NAMES):
        capabilities.append(Planning())
    if "delegate_task" in selected:
        if scope.subagent_delegate is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(_SubagentCapability(scope.subagent_delegate))
    capabilities.append(_build_compaction(scope.context_target_tokens))
    _logger.debug(
        "platform capabilities composed: agent=%s step=%s tools=%s count=%s memory_scope_digest=%s",
        scope.agent_name,
        scope.step_run_id,
        scope.platform_tool_names,
        len(capabilities),
        None if scope.memory_scope is None else canonical_sha256(scope.memory_scope),
    )
    return tuple(capabilities)


@dataclass
class _SelectedMemory(Memory[None]):
    selected_tool_names: "tuple[str, ...]" = ()

    def get_toolset(self) -> "AbstractToolset[None] | None":
        toolset = super().get_toolset()
        return None if toolset is None else toolset.filtered(lambda _ctx, definition: definition.name in self.selected_tool_names)


def tool_name_allowed(name: str, allow_tools: "tuple[str, ...]") -> bool:
    return "*" in allow_tools or name in allow_tools


def select_platform_tool_names(*, allow_tools: "tuple[str, ...]", memory_scope: "str | None", subagent_available: bool = False) -> "tuple[str, ...]":
    candidates = list(PLANNING_TOOL_NAMES)
    if memory_scope is not None:
        candidates.extend(MEMORY_TOOL_NAMES)
    if subagent_available:
        candidates.extend(SUBAGENT_TOOL_NAMES)
    return tuple(sorted(name for name in candidates if tool_name_allowed(name, allow_tools)))


class _SubagentCapability(AbstractCapability[None]):
    def __init__(self, delegate: SubagentDelegate) -> None:
        self._delegate = delegate

    def get_toolset(self) -> "FunctionToolset[None]":
        toolset = FunctionToolset[None](id="linktools-subagent")

        @toolset.tool
        async def delegate_task(ctx: RunContext[None], agent_id: str, task: str) -> "dict[str, Any]":
            values = (agent_id.strip(), task.strip())
            if not all(values):
                raise ModelRetry("agent_id and task are required")
            if not ctx.tool_call_id:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            result = await self._delegate(values[0], values[1], tool_call_id=ctx.tool_call_id)
            if not isinstance(result, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return result

        return toolset


def _memory_guidance(selected_tools: "tuple[str, ...]") -> str:
    actions = ", ".join(f"`{name}`" for name in selected_tools)
    return f"Use only these memory tools when needed: {actions}."


def _build_compaction(context_target_tokens: "int | None") -> PydanticAgentCapability[None]:
    deduplicate = DeduplicateFileReads(file_key=_workspace_file_key)
    if context_target_tokens is None:
        return deduplicate
    return TieredCompaction(
        tiers=[deduplicate, ClearToolResults(max_tokens=1, keep_pairs=3), SummarizingCompaction(max_messages=1, keep_messages=20)],
        target_tokens=context_target_tokens,
    )


def _validate_compaction_target(context_target_tokens: "int | None") -> None:
    if context_target_tokens is not None and (
        not isinstance(context_target_tokens, int)
        or isinstance(context_target_tokens, bool)
        or context_target_tokens <= 0
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _workspace_file_key(part: ToolCallPart) -> "str | None":
    if part.tool_name != "read_file":
        return None
    try:
        arguments = part.args_as_dict()
    except (TypeError, ValueError):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


__all__ = [
    "MEMORY_TOOL_NAMES",
    "PLANNING_TOOL_NAMES",
    "SKILL_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "AgentRunScope",
    "SubagentDelegate",
    "compose_platform_capabilities",
    "select_platform_tool_names",
    "tool_name_allowed",
]
