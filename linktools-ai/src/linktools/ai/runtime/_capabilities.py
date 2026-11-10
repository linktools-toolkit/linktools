#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from linktools.core import environ
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..core import PromptLimits, ToolEffects, DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..errors import AIError, ErrorCode
from ..persistence.capabilities import CapabilityPersistence
from ..capability.steps import AtomicStepPersistence, StepRunRegistration
from ._agent_run_recorder import AgentRunRecorder
from ._step_store import RuntimeStepStore
from ._tool import ToolOperationBridge
from ._budget import RunBudgetContext
from ._compaction import ExternalModelRequestRecorder, CompactionCapability, CompactionPolicy
from ..capability import ExecutionCapability, MemoryCapability, tool_effect_policy_from_metadata
from ._harness import HarnessPlanStoreAdapter
from ..capability.planning import HarnessPlanning
from pydantic_ai_harness.memory import MemoryStore
from ._tool_boundary import require_tool_effects
from ..capability import RunCallbackScope, TraceCapability
from ._plan import RuntimePlanStore
from ._run_hooks import AgentRunCapture
from ..spec import CapabilityRef
from ._conversation_service import ConversationService
from ._persistence_services import TraceService
from ._persistence_configuration import execution_configuration
from ..persistence.runs import AgentRunStore
from ..persistence.runs import RuntimeAgentRunStore

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.capabilities import CapabilityOrdering
    from ..observe.journal import ModelRequestJournal

_logger = environ.get_logger("ai.runtime.capabilities")


class _PersistenceCapability(WrapperCapability[None]):
    """Keep persistence attached to the effective native lifecycle instance."""

    def __init__(
        self, capability: CapabilityPersistence, replace_owner: Callable[[CapabilityPersistence], None],
    ) -> None:
        if not isinstance(capability, AbstractCapability):
            raise TypeError("bind_persistence must preserve the SDK capability type")
        super().__init__(capability)
        self.replace_owner = replace_owner

    def apply(self, visitor: Callable[[AbstractCapability[None]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractCapability[None]], AbstractCapability[None] | None],
    ) -> AbstractCapability[None] | None:
        replacement = self.wrapped.visit_and_replace(visitor)
        if replacement is None or replacement is self.wrapped:
            return None if replacement is None else self
        effective = cast(CapabilityPersistence, replacement)
        self.replace_owner(effective)
        return type(self)(effective, self.replace_owner)

    def get_ordering(self) -> CapabilityOrdering | None:
        return self.wrapped.get_ordering()

    def for_agent(self, agent: AbstractAgent[None, object]) -> AbstractCapability[None]:
        rebound = super().for_agent(agent)
        self.replace_owner(cast(CapabilityPersistence, rebound.wrapped))
        return rebound

    async def for_run(self, ctx: PydanticRunContext[None]) -> AbstractCapability[None]:
        rebound = await super().for_run(ctx)
        self.replace_owner(cast(CapabilityPersistence, rebound.wrapped))
        return rebound


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
    tool_effects: ToolEffects = "live",
    compaction_policy: CompactionPolicy,
    limits: PromptLimits,
    planning: bool,
    context_target_tokens: int | None,
    parent_agent_run_id: str | None,
    plan_store_resolver: Callable[[PydanticRunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    deferred_capability_refs: dict[str, CapabilityRef] | None = None,
    model_journal: "ModelRequestJournal | None" = None,
    model_request_recorder: "ExternalModelRequestRecorder | None" = None,
    recorder: AgentRunRecorder | None = None,
    budget: RunBudgetContext | None = None,
    runtime_capabilities: Sequence[AbstractCapability[None]] = (),
    capability_revisions: Mapping[str, int] | None = None,
    transcript_start: int = 0,
    initial_output_retries_used: int = 0,
    tool_operations: ToolOperationBridge | None = None,
) -> tuple[AbstractCapability[None], ...]:
    capabilities: list[AbstractCapability[None]] = []
    run_recorder = recorder or AgentRunRecorder(
        run_store,
        execution_id=execution_id,
        agent_run_id=agent_run_id,
    )
    leaves = tuple(leaf for capability in runtime_capabilities for leaf in runtime_feature_leaves(capability))
    guarantees = (
        execution_configuration(runtime_capabilities).retain_history,
        any(isinstance(leaf, StepPersistence) for leaf in leaves),
        any(isinstance(leaf, TraceCapability) for leaf in leaves),
    )
    if isinstance(run_store, RuntimeAgentRunStore):
        guarantees = run_store.execution_persistence(agent_run_id, declared=guarantees)
        run_store.bind_execution_persistence(
            agent_run_id, conversation=guarantees[0], restart_checkpoints=guarantees[1], durable_trace=guarantees[2],
        )
    async def require_active_producer() -> None:
        if isinstance(run_store, RuntimeAgentRunStore):
            await run_store.require_active_producer(agent_run_id)

    producer_generation = (
        run_store.execution_producer_generation(agent_run_id, execution_id=execution_id)
        if isinstance(run_store, RuntimeAgentRunStore) and execution_id is not None else None
    )

    async def flush_trace() -> None:
        if not isinstance(run_store, RuntimeAgentRunStore) or execution_id is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await run_store.require_active_producer(agent_run_id)
        await run_store.flush_execution_projection(
            agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
            require_durable=True,
        )

    persistence = AgentRunCapture(
        recorder=run_recorder,
        initial_output_retries_used=initial_output_retries_used,
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
        deferred_capability_refs={} if deferred_capability_refs is None else deferred_capability_refs,
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
                revision=(run_store.capability_revisions if capability_revisions is None else capability_revisions)[capability.id],
                adapt=_PersistenceCapability,
            )
            if not isinstance(bound, AbstractCapability):
                raise TypeError("bind_persistence must preserve the SDK capability type")
            return bound
        if isinstance(capability, ExecutionCapability):
            bound_types.add(ExecutionCapability)
            return capability.bind(ConversationService(persistence), retain_history=guarantees[0])
        if isinstance(capability, StepPersistence):
            if not guarantees[1]:
                return None
            bound_types.add(StepPersistence)
            persistence.metadata = {**capability.metadata, **persistence.metadata}
            # Preserve user-defined hook overrides; only adapt the original native class.
            if type(capability) is StepPersistence:
                capability = AtomicStepPersistence(
                    id=capability.id, description=capability.description, defer_loading=capability.defer_loading,
                    store=capability.store, agent_name=capability.agent_name,
                    run_id=capability.run_id, parent_run_id=capability.parent_run_id,
                    metadata=capability.metadata, capture_frontier=capability.capture_frontier,
                )
            native_metadata = {
                **persistence.metadata, "linktools.transcript_start": str(transcript_start),
                "linktools.transcript_run_id": (run_store.checkpoint_transcript_run_id(agent_run_id)
                    if isinstance(run_store, RuntimeAgentRunStore) else agent_run_id),
            }
            if type(capability) is AtomicStepPersistence and isinstance(run_store, RuntimeAgentRunStore):
                run_store.activate_step_boundary(agent_run_id)
                if producer_generation is not None:
                    registration = StepRunRegistration(
                        run_id=agent_run_id, parent_run_id=parent_agent_run_id,
                        agent_name=agent_id, metadata=native_metadata,
                        share_started=run_store.can_share_native_start(agent_run_id, capability.store),
                        lifecycle_participant_types=(AgentRunCapture, AtomicStepPersistence, TraceCapability, ExecutionCapability, RunCallbackScope),
                        lifecycle_observer_types=(AtomicStepPersistence, TraceCapability),
                    )
                    persistence.native_registration = registration
                    capability = replace(capability, run_registration=registration)
            return replace(
                capability, run_id=agent_run_id, parent_run_id=parent_agent_run_id,
                store=capability.store if tool_operations is None else RuntimeStepStore(
                    capability.store, tool_operations,
                    native_boundary=isinstance(capability, AtomicStepPersistence),
                    output_retries_used=persistence.checkpoint_output_retries_used,
                    working_set=run_store if isinstance(run_store, RuntimeAgentRunStore) else None,
                    commit_boundary=(run_store.step_boundary(agent_run_id)
                                     if isinstance(run_store, RuntimeAgentRunStore)
                                     and isinstance(capability, AtomicStepPersistence) else None),
                ),
                agent_name=agent_id,
                metadata=native_metadata,
            )
        for enabled, feature, service in ((guarantees[2], TraceCapability, TraceService),):
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
                return capability.bind(
                    service(persistence), flush=flush_trace,
                    model_request_seq=run_recorder.model_request_seq_for_tool_call,
                )
        return capability

    for template in runtime_capabilities:
        bound = template.visit_and_replace(bind)
        if bound is not None:
            capabilities.append(bound)
    if ExecutionCapability not in bound_types:
        capabilities.append(ExecutionCapability().bind(
            ConversationService(persistence), retain_history=guarantees[0],
        ))
    for enabled, feature, service in ((guarantees[2], TraceCapability, TraceService),):
        if enabled and feature not in bound_types:
            capabilities.append(feature().bind(
                service(persistence), flush=flush_trace,
                model_request_seq=run_recorder.model_request_seq_for_tool_call,
            ))
    selected_memory = MemoryCapability.select_tools(tool_policy)
    if memory_scope is not None and selected_memory:
        if memory_store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        memory_capability = MemoryCapability(scope=memory_scope)
        if tool_effects != "live":
            for tool in memory_capability.tool_definitions():
                if tool.name in selected_memory:
                    require_tool_effects(tool_effects, kind="tool", effect_policy=tool_effect_policy_from_metadata(
                        tool.metadata, require=True,
                    ))
        capabilities.append(memory_capability.bind(selected_memory, store=memory_store))
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

        planning_capability = HarnessPlanning.bind(resolve_plan_store)
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
