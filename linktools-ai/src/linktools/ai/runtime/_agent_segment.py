#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Agent segment execution primitives."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from ..agent import AgentBinding
from ..capability import AgentContext, SubagentDelegate
from ..core import ExecutionMode, JsonValue, PromptLimits, ThinkingValue, UsageMetrics
from ..errors import ErrorCode, ErrorDiagnostics
from ..workspace import RepositoryInstructions, Workspace
from ._agent_executor import (
    AgentExecutor,
    AgentExecutionResult,
    DurableBoundary,
    LiveDelta,
    _RunScope,
)
from ._input import CanonicalUserInput
from ._memory import MemoryStore
from ._plan import RuntimePlanStore
from ._tool import ToolOperationBridge
from ._tool_boundary import RepositoryInstructionBoundary
from .state import RuntimeDomain
from .state._step_contracts import StepStore
from .state._steps import ExecutionTerminalSealPlan


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    code: ErrorCode
    safe_details: Mapping[str, JsonValue]
    diagnostics: ErrorDiagnostics | None = None
    category: str | None = None
    retryable: bool | None = None
    operation_id: str | None = None


class _StepLifecycle(Protocol):
    async def materialize_conversation(self, *, step_run_id: str) -> None: ...
    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: "str | None" = None,
    ) -> None: ...
    async def materialize_recovery_snapshot(
        self, *, step_run_id: str, require_complete: bool
    ) -> None: ...
    async def verify_terminal_attempts(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        required_step_run_id: str | None,
    ) -> None: ...
    async def release_staging_many(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None: ...
    async def flush_execution_projection(
        self, step_run_id: str, *, execution_id: str
    ) -> None: ...
    async def wait_projection_flight(self, step_run_id: str) -> None: ...
    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan: ...
    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _AgentSegmentInput:
    """The immutable inputs needed to materialize one agent segment."""

    binding: AgentBinding
    context: AgentContext[object]
    workspace: "Workspace | None"
    limits: PromptLimits
    mcp_cwd: str
    user_prompt: CanonicalUserInput | None
    history: list[ModelMessage]
    conversation_id: str
    step_store: StepStore
    step_run_id: str
    segment_sequence: int
    history_id: str | None
    memory_store: MemoryStore | None
    plan_store_resolver: Callable[..., RuntimePlanStore] | None
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    parent_step_run_id: str | None
    subagent_available: bool
    subagent_descriptions: Mapping[str, str | None]
    subagent_delegate: SubagentDelegate | None
    event_sink: Callable[[LiveDelta | DurableBoundary], Awaitable[None]]
    usage_sink: Callable[[UsageMetrics], None]
    tool_operations: ToolOperationBridge | None
    background_tasks: set[asyncio.Task[object]]
    replace_history_system_prompt: bool
    repository_instructions: RepositoryInstructions | None
    repository_instruction_boundary: RepositoryInstructionBoundary | None
    deferred_tool_results: DeferredToolResults | None


@dataclass(frozen=True, slots=True)
class _SegmentCompleted:
    result: AgentExecutionResult


@dataclass(frozen=True, slots=True)
class _SegmentDeferred:
    requests: DeferredToolRequests


@dataclass(frozen=True, slots=True)
class _SegmentFailed:
    error: Exception


@dataclass(frozen=True, slots=True)
class _SegmentCancelled:
    pass


class _AgentSegmentRunner:
    """Run exactly one AgentExecutor segment without changing runtime state."""

    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor

    async def run(
        self,
        segment: _AgentSegmentInput,
    ) -> "_SegmentCompleted | _SegmentDeferred | _SegmentFailed | _SegmentCancelled":
        scope = _RunScope(
            binding=segment.binding,
            context=segment.context,
            workspace=segment.workspace,
            limits=segment.limits,
            mcp_cwd=segment.mcp_cwd,
            user_prompt=segment.user_prompt,
            history=segment.history,
            conversation_id=segment.conversation_id,
            step_store=segment.step_store,
            step_run_id=segment.step_run_id,
            segment_sequence=segment.segment_sequence,
            history_id=segment.history_id,
            memory_store=segment.memory_store,
            plan_store_resolver=segment.plan_store_resolver,
            mode=segment.mode,
            planning=segment.planning,
            thinking=segment.thinking,
            parent_step_run_id=segment.parent_step_run_id,
            subagent_available=segment.subagent_available,
            subagent_descriptions=segment.subagent_descriptions,
            subagent_delegate=segment.subagent_delegate,
            event_sink=segment.event_sink,
            usage_sink=segment.usage_sink,
            tool_operations=segment.tool_operations,
            background_tasks=segment.background_tasks,
            replace_history_system_prompt=segment.replace_history_system_prompt,
            repository_instructions=segment.repository_instructions,
            repository_instruction_boundary=segment.repository_instruction_boundary,
            deferred_tool_results=segment.deferred_tool_results,
        )
        try:
            result = await self._executor.execute(scope)
        except asyncio.CancelledError:
            return _SegmentCancelled()
        except Exception as error:
            return _SegmentFailed(error)
        if isinstance(result, DeferredToolRequests):
            return _SegmentDeferred(result)
        return _SegmentCompleted(result)


async def _step_messages(
    store: StepStore,
    run_id: str,
    *,
    include_interrupted: bool = False,
) -> list[ModelMessage]:
    snapshot = await store.latest_snapshot(
        run_id=run_id,
        include_interrupted=include_interrupted,
    )
    if snapshot is None:
        raise LookupError(run_id)
    return list(snapshot.messages)
