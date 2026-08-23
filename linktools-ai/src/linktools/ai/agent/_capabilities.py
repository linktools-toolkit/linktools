#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped Pydantic AI infrastructure capabilities."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, WrapToolExecuteHandler
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability
from pydantic_ai.exceptions import ModelRetry, ToolRetryError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model, ModelRequestContext
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
from ..core import JsonValue
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.agent.capabilities")

MEMORY_TOOL_NAMES = ("delete_memory", "read_memory", "search_memory", "write_memory")
MEMORY_READ_TOOL_NAMES = ("read_memory", "search_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
SUBAGENT_TOOL_NAMES = ("delegate_task",)
WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
    "write_file",
)
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = (
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
)
WORKSPACE_SHELL_TOOL_NAMES = ("check_command", "run_command", "start_command", "stop_command")
PLAN_SAFE_METADATA_KEY = "linktools.ai.plan_safe"
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
_WORKSPACE_FILESYSTEM_CAPABILITY_ID = "workspace-filesystem"
_WORKSPACE_SHELL_CAPABILITY_ID = "workspace-shell"
_SKILL_CAPABILITY_ID = "linktools-skill"
_MEMORY_CAPABILITY_ID = "linktools-memory"
_PLANNING_CAPABILITY_ID = "linktools-planning"
_SUBAGENT_CAPABILITY_ID = "linktools-subagent"


@dataclass(frozen=True, slots=True)
class ToolOperationDecision:
    operation_id: str
    owner: str
    fence: int
    replay_safe: bool
    cached_result: JsonValue = None
    has_cached_result: bool = False
    cached_error: "BaseException | None" = None


@dataclass
class _ToolCallState:
    decision: ToolOperationDecision
    handler_entered: bool = False
    operation_terminalized: bool = False
    preserve_started: bool = False
    cached_failure: bool = False
    effect_terminalized: bool = False
    heartbeat_task: "asyncio.Task[None] | None" = None


class ToolOperationBridge(Protocol):
    async def begin(
        self,
        ctx: "RunContext[None]",
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> ToolOperationDecision: ...

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision: ...

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool: ...

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool: ...

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None: ...


class _MissingToolOperationBridge:
    async def begin(
        self,
        ctx: "RunContext[None]",
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
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


class _RuntimeStepPersistence(StepPersistence[None]):
    def __init__(
        self,
        *,
        tool_operations: ToolOperationBridge,
        planning: bool = False,
        trusted_tool_classes: "tuple[tuple[str, str], ...]" = (),
        trusted_mcp_selectors: "tuple[str, ...]" = (),
        **kwargs: Any,
    ) -> None:
        if not isinstance(planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _validate_trusted_tool_classes(trusted_tool_classes)
        _validate_trusted_mcp_selectors(trusted_mcp_selectors)
        super().__init__(**kwargs)
        self._tool_operations = tool_operations
        self._planning = planning
        self._trusted_tool_classes = trusted_tool_classes
        self._trusted_mcp_selectors = trusted_mcp_selectors
        self._calls: dict[tuple[str, str], _ToolCallState] = {}

    async def before_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if self._planning and not tool_allowed_in_planning(
            tool_def,
            trusted_tool_classes=self._trusted_tool_classes,
            trusted_mcp_selectors=self._trusted_mcp_selectors,
        ):
            raise AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={"tool_name": tool_def.name, "planning": True},
            )
        decision = await self._tool_operations.begin(ctx, call, tool_def, args)
        key = self._decision_key(ctx, call)
        state = _ToolCallState(
            decision=decision,
            operation_terminalized=decision.has_cached_result or decision.cached_error is not None,
            cached_failure=decision.cached_error is not None,
        )
        self._calls[key] = state
        try:
            return await super().before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        except BaseException:
            self._calls.pop(key, None)
            raise

    async def wrap_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        key = self._decision_key(ctx, call)
        state = self._calls[key]
        if state.decision.cached_error is not None:
            try:
                await self._record_failed_effect(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=state.decision.cached_error,
                    state=state,
                )
            except (ModelRetry, ToolRetryError):
                self._calls.pop(key, None)
                raise
            raise AssertionError("cached failure effect hook must raise")
        if state.decision.has_cached_result:
            return state.decision.cached_result

        async def tracked_handler(validated_args: dict[str, Any]) -> Any:
            state.handler_entered = True
            return await handler(validated_args)

        handler_task = asyncio.create_task(
            super().wrap_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                handler=tracked_handler,
            ),
            name=f"tool-handler-{call.tool_call_id}",
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat(state, handler_task),
            name=f"tool-heartbeat-{call.tool_call_id}",
        )
        state.heartbeat_task = heartbeat_task
        keep_call_state = True
        try:
            done, _ = await asyncio.wait(
                (handler_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_error = heartbeat_task.exception()
                if heartbeat_error is not None:
                    handler_task.cancel()
                    await asyncio.gather(handler_task, return_exceptions=True)
                    if state.handler_entered and not state.decision.replay_safe:
                        await self._mark_unknown(state, heartbeat_error)
                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from heartbeat_error
                    state.preserve_started = True
                    raise heartbeat_error
            result = await handler_task
            try:
                cancelled = await self._tool_operations.complete(
                    state.decision,
                    result,
                )
            except BaseException:
                state.preserve_started = True
                raise
            state.operation_terminalized = True
            if cancelled:
                state.preserve_started = False
                keep_call_state = False
                self._calls.pop(key, None)
                raise asyncio.CancelledError
            return result
        except (ModelRetry, ToolRetryError) as error:
            if state.decision.replay_safe:
                cancelled = await self._tool_operations.fail(
                    state.decision,
                    error,
                )
                state.operation_terminalized = True
                if cancelled:
                    state.preserve_started = False
                    keep_call_state = False
                    self._calls.pop(key, None)
                    raise asyncio.CancelledError
                _logger.debug(
                    "tool effect marked failed: run=%s tool=%s call=%s",
                    self.run_id or ctx.run_id,
                    tool_def.name,
                    call.tool_call_id,
                )
                try:
                    await self._record_failed_effect(
                        ctx,
                        call=call,
                        tool_def=tool_def,
                        args=args,
                        error=error,
                        state=state,
                    )
                except (ModelRetry, ToolRetryError):
                    keep_call_state = False
                    self._calls.pop(key, None)
                    raise
                raise AssertionError("retry effect hook must raise")
            await self._mark_unknown(state, error)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
        except asyncio.CancelledError as error:
            if state.operation_terminalized:
                state.preserve_started = False
                keep_call_state = False
                self._calls.pop(key, None)
                raise
            state.preserve_started = True
            if state.handler_entered and not state.decision.replay_safe:
                await self._mark_unknown(state, error)
            keep_call_state = False
            self._calls.pop(key, None)
            raise
        except Exception as error:
            if not state.decision.replay_safe and state.handler_entered:
                await self._mark_unknown(state, error)
            elif not state.handler_entered:
                state.preserve_started = True
            raise
        finally:
            await self._stop_heartbeat(state)
            if not handler_task.done():
                handler_task.cancel()
                await asyncio.gather(handler_task, return_exceptions=True)
            if not keep_call_state:
                self._calls.pop(key, None)

    async def after_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        key = self._decision_key(ctx, call)
        state = self._calls.get(key)
        if state is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if state.preserve_started or not state.operation_terminalized or state.cached_failure:
            self._calls.pop(key, None)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        try:
            return await super().after_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
            )
        finally:
            self._calls.pop(key, None)

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        key = self._decision_key(ctx, call)
        state = self._calls.get(key)
        if state is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
        try:
            if state.operation_terminalized or state.cached_failure or state.preserve_started:
                raise error
            if not state.decision.replay_safe:
                await self._mark_unknown(state, error)
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
            cancelled = await self._tool_operations.fail(state.decision, error)
            state.operation_terminalized = True
            if cancelled:
                self._calls.pop(key, None)
                raise asyncio.CancelledError
            return await self._record_failed_effect(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=error,
                state=state,
            )
        finally:
            await self._stop_heartbeat(state)
            self._calls.pop(key, None)

    async def _heartbeat(
        self,
        state: _ToolCallState,
        handler_task: "asyncio.Task[Any]",
    ) -> None:
        while not handler_task.done():
            await asyncio.sleep(15)
            if handler_task.done():
                return
            state.decision = await self._tool_operations.renew(state.decision)

    async def _stop_heartbeat(self, state: _ToolCallState) -> None:
        task = state.heartbeat_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        state.heartbeat_task = None

    async def _record_failed_effect(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: BaseException,
        state: _ToolCallState,
    ) -> Any:
        if state.effect_terminalized:
            raise error
        try:
            result = await super().on_tool_execute_error(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=error,
            )
        except (ModelRetry, ToolRetryError):
            state.effect_terminalized = True
            raise
        state.effect_terminalized = True
        return result

    async def _mark_unknown(self, state: _ToolCallState, error: BaseException) -> None:
        state.preserve_started = True
        state.operation_terminalized = True
        await self._tool_operations.unknown(state.decision, error)

    def _decision_key(self, ctx: "RunContext[None]", call: ToolCallPart) -> tuple[str, str]:
        return self._effective_run_id(ctx), call.tool_call_id


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
    history_id: "str | None" = None
    platform_tool_names: "tuple[str, ...]" = ()
    planning: bool = False
    trusted_tool_classes: "tuple[tuple[str, str], ...]" = ()
    trusted_mcp_selectors: "tuple[str, ...]" = ()
    context_target_tokens: "int | None" = None
    parent_step_run_id: "str | None" = None
    subagent_delegate: "SubagentDelegate | None" = None
    tool_operations: "ToolOperationBridge | None" = None


class SubagentDelegate(Protocol):
    async def __call__(
        self,
        agent_id: str,
        user_prompt: str,
        *,
        tool_call_id: str,
    ) -> "dict[str, JsonValue]": ...


async def compose_platform_capabilities(
    scope: AgentRunScope,
    *,
    model_factory: Callable[["str | Model | None"], "str | Model"],
    parent_model: "str | Model",
) -> "tuple[PydanticAgentCapability[None], ...]":
    del model_factory, parent_model
    _validate_compaction_target(scope.context_target_tokens)
    _validate_trusted_tool_classes(scope.trusted_tool_classes)
    _validate_trusted_mcp_selectors(scope.trusted_mcp_selectors)
    capabilities: list[PydanticAgentCapability[None]] = []
    capabilities.append(
        _RuntimeStepPersistence(
            store=scope.step_store,
            agent_name=scope.agent_name,
            run_id=scope.step_run_id,
            parent_run_id=scope.parent_step_run_id,
            metadata={
                "capability_scope": "parent",
                "agent_name": scope.agent_name,
                **(
                    {}
                    if scope.history_id is None
                    else {"history_id": scope.history_id}
                ),
                **(
                    {}
                    if scope.segment_sequence is None
                    else {"segment_sequence": str(scope.segment_sequence)}
                ),
            },
            tool_operations=scope.tool_operations or _MissingToolOperationBridge(),
            planning=scope.planning,
            trusted_tool_classes=scope.trusted_tool_classes,
            trusted_mcp_selectors=scope.trusted_mcp_selectors,
        )
    )
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
                id=_MEMORY_CAPABILITY_ID,
            )
        )
    if scope.context_target_tokens is not None:
        capabilities.append(
            _CompactionCapability(
                scope.context_target_tokens,
                step_store=scope.step_store,
                conversation_id=scope.conversation_id,
                step_run_id=scope.step_run_id,
            )
        )
    if any(name in selected for name in PLANNING_TOOL_NAMES):
        capabilities.append(Planning(id=_PLANNING_CAPABILITY_ID))
    if "delegate_task" in selected:
        if scope.subagent_delegate is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(_SubagentCapability(scope.subagent_delegate))
    return tuple(capabilities)


@dataclass
class _SelectedMemory(Memory[None]):
    selected_tool_names: "tuple[str, ...]" = ()

    def get_toolset(self) -> "AbstractToolset[None] | None":
        toolset = super().get_toolset()
        return None if toolset is None else toolset.filtered(
            lambda _ctx, definition: definition.name in self.selected_tool_names
        )


class _CompactionCapability(AbstractCapability[None]):
    def __init__(
        self,
        target_tokens: int,
        *,
        step_store: StepStore,
        conversation_id: "str | None",
        step_run_id: str,
    ) -> None:
        self._target_tokens = target_tokens
        self._step_store = step_store
        self._conversation_id = conversation_id
        self._step_run_id = step_run_id
        self._compaction = _build_compaction(target_tokens)

    async def before_run(self, ctx: RunContext[None]) -> None:
        if ctx.message_history:
            return
        if self._conversation_id is None:
            return
        try:
            run = await self._step_store.get_run(run_id=self._step_run_id)
            if run is None or not run.messages:
                return
            ctx.message_history[:] = list(run.messages)
        except BaseException:
            ctx.message_history[:] = []
            raise

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self._compaction.before_model_request(ctx, request_context)


class _SubagentCapability(AbstractCapability[None]):
    def __init__(self, delegate: SubagentDelegate) -> None:
        self.id = _SUBAGENT_CAPABILITY_ID
        self._delegate = delegate

    def get_toolset(self) -> "FunctionToolset[None]":
        toolset = FunctionToolset[None](id=_SUBAGENT_CAPABILITY_ID)

        @toolset.tool
        async def delegate_task(ctx: RunContext[None], agent_id: str, task: str) -> "dict[str, Any]":
            values = (agent_id.strip(), task.strip())
            if not all(values):
                raise ModelRetry("agent_id and task are required")
            if not ctx.tool_call_id:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            result = await self._delegate(values[0], values[1], tool_call_id=ctx.tool_call_id)
            if not isinstance(result, dict):
                raise AIError(ErrorCode.INTERNAL_ERROR)
            return result

        return toolset


def _memory_guidance(selected_tools: "tuple[str, ...]") -> str:
    actions = ", ".join(f"`{name}`" for name in selected_tools)
    return f"Use only these memory tools when needed: {actions}."


def _build_compaction(
    context_target_tokens: "int | None",
) -> AbstractCapability[None]:
    deduplicate = DeduplicateFileReads(file_key=_workspace_file_key)
    if context_target_tokens is None:
        return deduplicate
    return TieredCompaction(
        tiers=[
            deduplicate,
            ClearToolResults(max_tokens=1, keep_pairs=3),
            SummarizingCompaction(max_messages=1, keep_messages=20),
        ],
        target_tokens=context_target_tokens,
    )


def _workspace_file_key(part: ToolCallPart) -> "str | None":
    if part.tool_name != "read_file":
        return None
    try:
        arguments = part.args_as_dict()
    except (TypeError, ValueError):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


def _validate_compaction_target(value: "int | None") -> None:
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


def tool_name_allowed(name: str, allow_tools: "tuple[str, ...]") -> bool:
    return "*" in allow_tools or name in allow_tools


def _trusted_tool_capability(name: str, tool_class: str) -> "str | None":
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
        return _WORKSPACE_FILESYSTEM_CAPABILITY_ID
    if tool_class == "shell":
        return _WORKSPACE_SHELL_CAPABILITY_ID if name in WORKSPACE_SHELL_TOOL_NAMES else None
    if tool_class in {"memory.read", "memory.write"}:
        if name not in MEMORY_TOOL_NAMES:
            return None
        is_read = name in MEMORY_READ_TOOL_NAMES
        if is_read != (tool_class == "memory.read"):
            return None
        return _MEMORY_CAPABILITY_ID
    return None


def tool_is_control(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
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
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
    trusted_mcp_selectors: "tuple[str, ...]",
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


def select_platform_tool_names(
    *,
    allow_tools: "tuple[str, ...]",
    memory_scope: "str | None",
    subagent_available: bool = False,
    planning: bool = False,
    workspace_filesystem: bool = False,
    workspace_filesystem_write: bool = False,
    workspace_shell: bool = False,
) -> "tuple[str, ...]":
    names: set[str] = set()
    if memory_scope is not None:
        names.update(MEMORY_TOOL_NAMES)
    if subagent_available:
        names.update(SUBAGENT_TOOL_NAMES)
    if planning:
        names.update(PLANNING_TOOL_NAMES)
    if workspace_filesystem:
        names.update(WORKSPACE_FILESYSTEM_TOOL_NAMES if workspace_filesystem_write else WORKSPACE_FILESYSTEM_READ_TOOL_NAMES)
    if workspace_shell:
        names.update(WORKSPACE_SHELL_TOOL_NAMES)
    names = {name for name in names if tool_name_allowed(name, allow_tools)}
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
    "AgentRunScope",
    "SubagentDelegate",
    "ToolOperationBridge",
    "ToolOperationDecision",
    "compose_platform_capabilities",
    "select_platform_tool_names",
    "tool_allowed_in_planning",
    "tool_is_control",
    "tool_name_allowed",
]
