#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local AgentRun execution primitives."""

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
from ..spec import RepositoryInstructions
from ..workspace import Workspace
from ._agent_executor import (
    AgentExecutor,
    AgentExecutionResult,
    DurableBoundary,
    LiveDelta,
    _AgentRunScope,
)
from ._input import CanonicalUserInput
from ._memory import MemoryStore
from ._plan import RuntimePlanStore
from ._tool import ToolOperationBridge
from ._tool_boundary import RepositoryInstructionBoundary
from .state import RuntimeDomain
from .state._contracts import LoadedModelContext
from .state._step_contracts import AgentRunStore
from .state._steps import ExecutionTerminalSealPlan


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    code: ErrorCode
    safe_details: Mapping[str, JsonValue]
    diagnostics: ErrorDiagnostics | None = None
    category: str | None = None
    retryable: bool | None = None
    operation_id: str | None = None


class _AgentRunLifecycle(Protocol):
    async def materialize_conversation(self, *, agent_run_id: str) -> None: ...
    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        agent_run_id: str,
        execution_id: "str | None" = None,
    ) -> None: ...
    async def materialize_recovery_snapshot(
        self, *, agent_run_id: str, require_complete: bool
    ) -> None: ...
    async def verify_terminal_attempts(
        self,
        *,
        candidate_agent_run_ids: tuple[str, ...],
        required_agent_run_id: str | None,
    ) -> None: ...
    async def release_staging_many(
        self,
        *,
        candidate_agent_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None: ...
    async def flush_execution_projection(
        self, agent_run_id: str, *, execution_id: str
    ) -> None: ...
    async def wait_projection_flight(self, agent_run_id: str) -> None: ...
    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        agent_run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan: ...
    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def reconcile_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _AgentRunInput:
    """The immutable inputs needed to materialize one AgentRun."""

    binding: AgentBinding
    context: AgentContext[object]
    workspace: "Workspace | None"
    limits: PromptLimits
    execution_cwd: "str | None"
    user_prompt: CanonicalUserInput | None
    initial_attachments: tuple[Mapping[str, JsonValue], ...]
    history: list[ModelMessage]
    initial_context: LoadedModelContext
    agent_conversation_id: str
    run_store: AgentRunStore
    agent_run_id: str
    agent_run_sequence: int
    history_id: str | None
    memory_store: MemoryStore | None
    plan_store_resolver: Callable[..., RuntimePlanStore] | None
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    parent_agent_run_id: str | None
    subagent_available: bool
    subagent_descriptions: Mapping[str, str | None]
    subagent_delegate: SubagentDelegate | None
    event_sink: Callable[[LiveDelta | DurableBoundary], Awaitable[None]]
    usage_sink: Callable[[UsageMetrics], None]
    tool_operations: ToolOperationBridge | None
    replace_history_system_prompt: bool
    repository_instructions: RepositoryInstructions | None
    repository_instruction_boundary: RepositoryInstructionBoundary | None
    deferred_tool_results: DeferredToolResults | None


@dataclass(frozen=True, slots=True)
class _AgentRunCompleted:
    result: AgentExecutionResult


@dataclass(frozen=True, slots=True)
class _AgentRunDeferred:
    requests: DeferredToolRequests


@dataclass(frozen=True, slots=True)
class _AgentRunFailed:
    error: Exception


@dataclass(frozen=True, slots=True)
class _AgentRunCancelled:
    pass


class _AgentRunRunner:
    """Run exactly one AgentRun without changing runtime state."""

    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor

    async def run(
        self,
        agent_run_input: _AgentRunInput,
    ) -> "_AgentRunCompleted | _AgentRunDeferred | _AgentRunFailed | _AgentRunCancelled":
        scope = _AgentRunScope(
            binding=agent_run_input.binding,
            context=agent_run_input.context,
            workspace=agent_run_input.workspace,
            limits=agent_run_input.limits,
            execution_cwd=agent_run_input.execution_cwd,
            user_prompt=agent_run_input.user_prompt,
            history=agent_run_input.history,
            initial_context=agent_run_input.initial_context,
            initial_attachments=agent_run_input.initial_attachments,
            agent_conversation_id=agent_run_input.agent_conversation_id,
            run_store=agent_run_input.run_store,
            agent_run_id=agent_run_input.agent_run_id,
            agent_run_sequence=agent_run_input.agent_run_sequence,
            history_id=agent_run_input.history_id,
            memory_store=agent_run_input.memory_store,
            plan_store_resolver=agent_run_input.plan_store_resolver,
            mode=agent_run_input.mode,
            planning=agent_run_input.planning,
            thinking=agent_run_input.thinking,
            parent_agent_run_id=agent_run_input.parent_agent_run_id,
            subagent_available=agent_run_input.subagent_available,
            subagent_descriptions=agent_run_input.subagent_descriptions,
            subagent_delegate=agent_run_input.subagent_delegate,
            event_sink=agent_run_input.event_sink,
            usage_sink=agent_run_input.usage_sink,
            tool_operations=agent_run_input.tool_operations,
            replace_history_system_prompt=agent_run_input.replace_history_system_prompt,
            repository_instructions=agent_run_input.repository_instructions,
            repository_instruction_boundary=agent_run_input.repository_instruction_boundary,
            deferred_tool_results=agent_run_input.deferred_tool_results,
        )
        try:
            result = await self._executor.execute(scope)
        except asyncio.CancelledError:
            return _AgentRunCancelled()
        except Exception as error:
            return _AgentRunFailed(error)
        if isinstance(result, DeferredToolRequests):
            return _AgentRunDeferred(result)
        return _AgentRunCompleted(result)


async def _agent_run_messages(
    store: AgentRunStore,
    agent_run_id: str,
    *,
    include_interrupted: bool = False,
) -> list[ModelMessage]:
    snapshot = await store.latest_snapshot(
        agent_run_id=agent_run_id,
        include_interrupted=include_interrupted,
    )
    if snapshot is None:
        raise LookupError(agent_run_id)
    return list(
        snapshot.messages
        if snapshot.context_messages is None
        else snapshot.context_messages
    )
