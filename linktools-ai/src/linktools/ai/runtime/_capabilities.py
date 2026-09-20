#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from linktools.core import environ
from pydantic_ai import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..core import PromptLimits
from ..errors import AIError, ErrorCode
from ._compaction import (
    ExternalModelRequestCapture,
    RuntimeCompaction,
    RuntimeCompactionPolicy,
)
from ._capture import RuntimeCaptureStore
from ._harness import HarnessPlanStoreAdapter
from ._harness_memory import (
    build_harness_memory,
    select_harness_memory_tools,
)
from ._harness_planning import build_harness_planning
from ._memory import MemoryStore
from ._metric_capability import RuntimeModelObservationCapability
from ._plan import RuntimePlanStore
from .state._step_contracts import ContinuableSnapshot, RunRecord, StepStore

if TYPE_CHECKING:
    from ._journal import ModelRequestJournal

_MEMORY_CAPABILITY_ID = "linktools.ai.memory"
_logger = environ.get_logger("ai.runtime.capabilities")


@dataclass(kw_only=True, eq=False)
class _RuntimeStepPersistence(AbstractCapability[None]):
    """Persist Runtime-owned step events, raw occurrences, and recovery snapshots."""

    capture: RuntimeCaptureStore = field(repr=False, compare=False)
    agent_name: str
    run_id: str
    parent_run_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    deferred_pause_sink: Callable[[int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _last_observed_step_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _last_snapshot_transcript_count: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _last_snapshot_context: tuple[ModelMessage, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _live_messages: Sequence[ModelMessage] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.capture, RuntimeCaptureStore):
            raise TypeError("capture must be RuntimeCaptureStore")
        if not self.run_id or self.capture.step_run_id != self.run_id:
            raise ValueError("Runtime persistence run id is invalid")

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_run(self, ctx: PydanticRunContext[None]) -> None:
        self._live_messages = ctx.messages
        await self.capture.register_run(
            RunRecord(
                run_id=self.run_id,
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                metadata=dict(self.metadata),
                started_at=datetime.now(timezone.utc),
            )
        )
        await self.capture.record_event("run_started", ctx.run_step)

    async def before_model_request(
        self,
        ctx: PydanticRunContext[None],
        request_context: object,
    ) -> object:
        self._live_messages = ctx.messages
        if ctx.messages and isinstance(ctx.messages[-1], ModelRequest):
            self.capture.append_transcript_message(ctx.messages[-1])
        return request_context

    async def after_node_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        self._live_messages = ctx.messages
        self._last_observed_step_index = ctx.run_step
        if isinstance(node, ModelRequestNode):
            response = None
            if isinstance(result, CallToolsNode):
                response = result.model_response
            elif ctx.messages and isinstance(ctx.messages[-1], ModelResponse):
                response = ctx.messages[-1]
            if response is not None:
                self.capture.append_transcript_message(response)
        if isinstance(node, CallToolsNode):
            pending = result.request if isinstance(result, ModelRequestNode) else None
            await self._save_snapshot(
                ctx,
                messages=ctx.messages,
                pending=pending,
                state="complete",
            )
        return result

    async def after_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        self._live_messages = result.all_messages()
        interrupted = isinstance(result.output, DeferredToolRequests)
        if interrupted:
            if self._last_observed_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self.deferred_pause_sink is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self.deferred_pause_sink(self._last_observed_step_index)
            self.capture.mark_interrupted()
        await self._save_snapshot(
            ctx,
            messages=result.all_messages(),
            state="interrupted" if interrupted else "complete",
        )
        await self.capture.record_event(
            "run_interrupted" if interrupted else "run_completed",
            ctx.run_step,
        )
        return result

    async def on_run_error(
        self,
        ctx: PydanticRunContext[None],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        messages = self._live_messages or ctx.messages
        await self._save_snapshot(
            ctx,
            messages=messages,
            state="interrupted",
        )
        await self.capture.record_event(
            "run_failed",
            ctx.run_step,
            error=repr(error),
        )
        raise error

    async def before_tool_execute(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        await self.capture.record_event(
            "tool_call_started",
            ctx.run_step,
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
        )
        return args

    async def after_tool_execute(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        del args
        await self.capture.record_event(
            "tool_call_completed",
            ctx.run_step,
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
        )
        return result

    async def on_tool_execute_error(
        self,
        ctx: PydanticRunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        del args
        await self.capture.record_event(
            "tool_call_failed",
            ctx.run_step,
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
            error=repr(error),
        )
        raise error

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        self.capture.remember_context_projection(source, projected)

    async def _save_snapshot(
        self,
        ctx: PydanticRunContext[None],
        *,
        messages: Sequence[ModelMessage],
        pending: ModelMessage | None = None,
        state: str,
    ) -> None:
        context_messages, pending_index = self.capture.snapshot_context(
            messages,
            pending=pending,
        )
        frozen_context = tuple(context_messages)
        raw = self.capture.transcript_messages()
        if (
            self._last_snapshot_context == frozen_context
            and self._last_snapshot_transcript_count == len(raw)
            and pending_index is None
        ):
            return
        await self.capture.save_snapshot(
            ContinuableSnapshot(
                run_id=self.run_id,
                step_index=ctx.run_step,
                messages=list(raw),
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                state=state,
                context_messages=context_messages,
                transcript_message_count_before=self._last_snapshot_transcript_count,
                pending_request_index=pending_index,
            )
        )
        self._last_snapshot_transcript_count = len(raw)
        self._last_snapshot_context = frozen_context


async def compose_platform_capabilities(
    *,
    agent_name: str,
    step_run_id: str,
    execution_id: str | None = None,
    segment_sequence: int | None,
    history_id: str | None,
    memory_scope: str | None,
    step_store: StepStore,
    memory_store: MemoryStore | None,
    ordinary_tool_policy: tuple[str, ...],
    compaction_policy: RuntimeCompactionPolicy,
    limits: PromptLimits,
    planning: bool,
    context_target_tokens: int | None,
    parent_step_run_id: str | None,
    plan_store_resolver: Callable[[PydanticRunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    model_journal: "ModelRequestJournal | None" = None,
    model_observation_enabled: bool = False,
    model_request_observer: "ExternalModelRequestCapture | None" = None,
    harness_store: HarnessStepStoreAdapter | None = None,
) -> tuple[AbstractCapability[None], ...]:
    capabilities: list[AbstractCapability[None]] = []
    persistence = _RuntimeStepPersistence(
        id="linktools.ai.step-persistence",
        store=harness_store or HarnessStepStoreAdapter(
            step_store,
            execution_id=execution_id,
        ),
        agent_name=agent_name,
        run_id=step_run_id,
        parent_run_id=parent_step_run_id,
        metadata={
            "capability_scope": "parent",
            "agent_name": agent_name,
            **({} if history_id is None else {"history_id": history_id}),
            **(
                {}
                if segment_sequence is None
                else {"segment_sequence": str(segment_sequence)}
            ),
        },
        deferred_pause_sink=deferred_pause_sink,
        model_journal=model_journal,
        model_observation_enabled=model_observation_enabled,
    )
    capabilities.append(persistence)
    selected_memory = select_harness_memory_tools(ordinary_tool_policy)
    if memory_scope is not None and selected_memory:
        if memory_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        memory_capability = build_harness_memory(
            memory_store,
            allow_tools=ordinary_tool_policy,
            capability_id=_MEMORY_CAPABILITY_ID,
        )
        capabilities.append(memory_capability)
    if planning:
        if plan_store_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        def resolve_plan_store(
            ctx: PydanticRunContext[None],
        ) -> HarnessPlanStoreAdapter:
            store = plan_store_resolver(ctx)
            if store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return HarnessPlanStoreAdapter(store)

        planning_capability = build_harness_planning(resolve_plan_store)
        capabilities.append(planning_capability)
    capabilities.append(
        RuntimeCompaction(
            context_target_tokens,
            limits=limits,
            policy=compaction_policy,
            journal=model_journal,
            request_observer=model_request_observer,
            projection_sink=persistence.remember_context_projection,
        )
    )
    _logger.debug(
        "platform capabilities composed: agent=%s step=%s memory_tools=%s "
        "planning=%s compaction_policy=per-run",
        agent_name,
        step_run_id,
        selected_memory,
        planning,
    )
    return tuple(capabilities)


__all__ = [
    "compose_platform_capabilities",
]
