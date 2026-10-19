#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..core import PromptLimits, DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..errors import AIError, ErrorCode
from .persistence import CapabilityPersistence
from ._agent_run_recorder import AgentRunRecorder
from ._budget import RunBudgetContext
from ._compaction import ExternalModelRequestRecorder, CompactionCapability, CompactionPolicy
from linktools.ai.capability import ConversationCapability
from ._harness import HarnessPlanStoreAdapter
from ._harness_memory import build_harness_memory, select_harness_memory_tools
from ._harness_planning import build_harness_planning
from ._memory import MemoryStore
from linktools.ai.capability import CheckpointCapability, TraceCapability
from ._plan import RuntimePlanStore
from ._run_hooks import AgentRunCapture
from ._conversation_service import ConversationService
from ._persistence_services import CheckpointService, TraceService
from .state._step_contracts import AgentRunStore
from .state._steps import RuntimeAgentRunStore

if TYPE_CHECKING:
    from ._journal import ModelRequestJournal

_MEMORY_CAPABILITY_ID = "linktools.ai.memory"
_logger = environ.get_logger("ai.runtime.capabilities")


def runtime_feature_leaves(
    capability: AbstractCapability[None],
) -> tuple[AbstractCapability[None], ...]:
    """Inspect declared feature leaves through native containers and wrappers."""
    if isinstance(capability, WrapperCapability):
        return runtime_feature_leaves(capability.wrapped)
    if isinstance(capability, CombinedCapability):
        return tuple(leaf for child in capability.capabilities for leaf in runtime_feature_leaves(child))
    leaves: list[AbstractCapability[None]] = []
    capability.apply(leaves.append)
    return tuple(leaves)


async def compose_platform_capabilities(
    *,
    agent_id: str,
    agent_run_id: str,
    execution_id: str | None = None,
    agent_run_seq: int | None,
    history_id: str | None,
    memory_scope: str | None,
    run_store: AgentRunStore,
    memory_store: MemoryStore | None,
    tool_policy: tuple[str, ...],
    compaction_policy: CompactionPolicy,
    limits: PromptLimits,
    planning: bool,
    context_target_tokens: int | None,
    parent_agent_run_id: str | None,
    plan_store_resolver: Callable[[PydanticRunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    model_journal: "ModelRequestJournal | None" = None,
    model_request_recorder: "ExternalModelRequestRecorder | None" = None,
    recorder: AgentRunRecorder | None = None,
    budget: RunBudgetContext | None = None,
    runtime_capabilities: Sequence[AbstractCapability[None]] = (),
) -> tuple[AbstractCapability[None], ...]:
    capabilities: list[AbstractCapability[None]] = []
    run_recorder = recorder or AgentRunRecorder(
        run_store,
        execution_id=execution_id,
        agent_run_id=agent_run_id,
    )
    guarantees = (True, True, True)
    if isinstance(run_store, RuntimeAgentRunStore):
        guarantees = run_store.execution_persistence(agent_run_id)
    async def require_active_producer() -> None:
        if isinstance(run_store, RuntimeAgentRunStore):
            await run_store.require_active_producer(agent_run_id)

    persistence = AgentRunCapture(
        recorder=run_recorder,
        agent_id=agent_id,
        agent_run_id=agent_run_id,
        parent_agent_run_id=parent_agent_run_id,
        metadata={
            "capability_scope": "parent",
            **({} if history_id is None else {"history_id": history_id}),
            **(
                {}
                if agent_run_seq is None
                else {"agent_run_seq": str(agent_run_seq)}
            ),
        },
        deferred_pause_sink=deferred_pause_sink,
        producer_guard=require_active_producer,
    )
    capabilities.append(persistence)
    bound_types: set[type[object]] = set()

    def bind(capability: AbstractCapability[None]) -> AbstractCapability[None] | None:
        if isinstance(capability, WrapperCapability):
            wrapped = capability.wrapped.visit_and_replace(bind)
            if wrapped is None:
                return None
            if wrapped is capability.wrapped:
                return capability
            rebound = copy(capability)
            rebound.wrapped = wrapped
            return rebound
        if isinstance(capability, CapabilityPersistence):
            if not isinstance(run_store, RuntimeAgentRunStore) or run_store.capability_writes is None or execution_id is None or capability.id is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            bound = run_store.capability_writes.bind(
                capability.id, agent_run_id, execution_id, capability,
                revision=run_store.capability_revisions[capability.id],
            )
            if not isinstance(bound, AbstractCapability):
                raise TypeError("bind_persistence must preserve the SDK capability type")
            return bound
        for enabled, feature, service in zip(guarantees, (ConversationCapability, CheckpointCapability, TraceCapability), (ConversationService, CheckpointService, TraceService)):
            if isinstance(capability, feature):
                if not enabled:
                    return None
                bound_types.add(feature)
                if isinstance(capability, TraceCapability) and isinstance(run_store, RuntimeAgentRunStore):
                    run_store.trace_persistence.configure(
                        agent_run_id, buffer=capability.commit.buffer,
                        terminal=capability.commit.mode == "terminal",
                        batch_size=capability.commit.batch_size or DEFAULT_TRACE_BATCH_SIZE,
                        flush_interval=capability.commit.flush_interval or DEFAULT_TRACE_FLUSH_INTERVAL,
                    )
                return capability.bind(service(persistence))
        return capability

    for template in runtime_capabilities:
        bound = template.visit_and_replace(bind)
        if bound is not None:
            capabilities.append(bound)
    for enabled, feature, service in zip(guarantees, (ConversationCapability, CheckpointCapability, TraceCapability), (ConversationService, CheckpointService, TraceService)):
        if enabled and feature not in bound_types:
            capabilities.append(feature().bind(service(persistence)))
    selected_memory = select_harness_memory_tools(tool_policy)
    if memory_scope is not None and selected_memory:
        if memory_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        memory_capability = build_harness_memory(
            memory_store,
            allow_tools=tool_policy,
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
        CompactionCapability(
            context_target_tokens,
            limits=limits,
            policy=compaction_policy,
            journal=model_journal,
            request_recorder=model_request_recorder,
            projection_sink=persistence.remember_context_projection,
            budget=budget,
        )
    )
    _logger.debug(
        "platform capabilities composed: agent=%s step=%s memory_tools=%s "
        "planning=%s compaction_policy=per-run",
        agent_id,
        agent_run_id,
        selected_memory,
        planning,
    )
    return tuple(capabilities)


__all__ = [
    "compose_platform_capabilities",
    "runtime_feature_leaves",
]
