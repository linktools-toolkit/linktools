#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
    WrapModelRequestHandler,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext as PydanticRunContext
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence

from ..capability import SUBAGENT_TOOL_NAMES
from ._compaction import ExternalModelRequestObserver, RuntimeCompaction
from ._harness import (
    HarnessPlanStoreAdapter,
    HarnessStepStoreAdapter,
)
from ._harness_memory import build_harness_memory
from ._memory import MemoryStore
from ._plan import RuntimePlanStore
from .state._step_contracts import (
    StepStore,
)
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ._journal import ModelRequestFact, ModelRequestJournal

MEMORY_TOOL_NAMES = (
    "delete_memory",
    "read_memory",
    "search_memory",
    "write_memory",
)
MEMORY_READ_TOOL_NAMES = ("read_memory", "search_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
_MEMORY_CAPABILITY_ID = "linktools-memory"
_PLANNING_CAPABILITY_ID = "linktools-planning"


def _tool_name_allowed(name: str, allow_tools: tuple[str, ...]) -> bool:
    return "*" in allow_tools or name in allow_tools


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
            if _tool_name_allowed(name, ordinary_tool_policy)
        )
    if planning:
        names.update(PLANNING_TOOL_NAMES)
    if subagent_available:
        names.update(SUBAGENT_TOOL_NAMES)
    return tuple(sorted(names))


@dataclass(kw_only=True, eq=False)
class _RuntimeStepPersistence(StepPersistence[None]):
    """Keep Harness graph persistence and Runtime snapshot integration together."""

    deferred_pause_sink: Callable[[int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    model_journal: "ModelRequestJournal | None" = field(
        default=None,
        repr=False,
        compare=False,
    )
    model_observation_enabled: bool = field(
        default=False,
        repr=False,
        compare=False,
    )
    _last_observed_step_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.store, HarnessStepStoreAdapter):
            raise TypeError("store must be HarnessStepStoreAdapter")

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    @property
    def _runtime_store(self) -> HarnessStepStoreAdapter:
        return cast(HarnessStepStoreAdapter, self.store)

    def _runtime_run_id(self, ctx: PydanticRunContext[Any]) -> str:
        value = self.run_id or ctx.run_id
        if not isinstance(value, str) or not value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def _model_metadata(
        self,
        fact: "ModelRequestFact",
        *,
        response: ModelResponse | None = None,
    ) -> dict[str, str]:
        metadata = fact.metadata(
            include_observation=(
                self.model_observation_enabled and fact.duration_ns is not None
            ),
        )
        if not self.model_observation_enabled:
            metadata.pop("linktools.ai.duration_ns", None)
        if response is not None:
            usage = response.usage
            metadata.update(
                {
                    "linktools.ai.model_usage.input_tokens": str(
                        usage.input_tokens
                    ),
                    "linktools.ai.model_usage.output_tokens": str(
                        usage.output_tokens
                    ),
                    "linktools.ai.model_usage.cache_read_tokens": str(
                        usage.cache_read_tokens
                    ),
                    "linktools.ai.model_usage.cache_write_tokens": str(
                        usage.cache_write_tokens
                    ),
                }
            )
        return metadata

    def _with_model_metadata(
        self,
        fact: "ModelRequestFact",
        *,
        response: ModelResponse | None = None,
    ) -> dict[str, str]:
        original = dict(self.metadata)
        self.metadata.update(self._model_metadata(fact, response=response))
        return original

    def _restore_metadata(self, original: dict[str, str]) -> None:
        self.metadata.clear()
        self.metadata.update(original)

    def _current_model_fact(
        self,
        ctx: PydanticRunContext[Any],
    ) -> "ModelRequestFact | None":
        if self.model_journal is None:
            return None
        return self.model_journal.latest_for_step(ctx.run_step)

    async def before_model_request(
        self,
        ctx: PydanticRunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        fact = self._current_model_fact(ctx)
        if fact is None:
            return await super().before_model_request(ctx, request_context)
        original = self._with_model_metadata(fact)
        try:
            return await super().before_model_request(ctx, request_context)
        finally:
            self._restore_metadata(original)

    async def after_node_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        observed = await super().after_node_run(ctx, node=node, result=result)
        self._last_observed_step_index = ctx.run_step
        return observed

    async def after_run(
        self,
        ctx: PydanticRunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        output = result.output
        interrupted = isinstance(output, DeferredToolRequests)
        if interrupted:
            if self._last_observed_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self.deferred_pause_sink is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self.deferred_pause_sink(self._last_observed_step_index)
            self._runtime_store.mark_interrupted(self._runtime_run_id(ctx))
        observed = await super().after_run(ctx, result=result)
        if interrupted:
            await self._runtime_store.save_interrupted_snapshot(
                run_id=self._runtime_run_id(ctx),
                step_index=cast(int, self._last_observed_step_index),
                messages=result.all_messages(),
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
            )
        return observed

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        self._runtime_store.remember_context_projection(source, projected)

    async def after_model_request(
        self,
        ctx: PydanticRunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._runtime_store.capture_model_context(request_context.messages)
        fact = self._current_model_fact(ctx)
        if fact is not None:
            original = self._with_model_metadata(fact, response=response)
            try:
                return await super().after_model_request(
                    ctx,
                    request_context=request_context,
                    response=response,
                )
            finally:
                self._restore_metadata(original)
        return await super().after_model_request(
            ctx,
            request_context=request_context,
            response=response,
        )

    async def wrap_model_request(
        self,
        ctx: PydanticRunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        try:
            return await handler(request_context)
        except asyncio.CancelledError as error:
            fact = self._current_model_fact(ctx)
            if fact is None:
                await self._record_event(
                    ctx,
                    kind="model_request_failed",
                    error=repr(error),
                )
            else:
                original = self._with_model_metadata(fact)
                try:
                    await self._record_event(
                        ctx,
                        kind="model_request_failed",
                        error=repr(error),
                    )
                finally:
                    self._restore_metadata(original)
            raise

    async def on_model_request_error(
        self,
        ctx: PydanticRunContext[None],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        self._runtime_store.capture_model_context(request_context.messages)
        fact = self._current_model_fact(ctx)
        if fact is not None:
            original = self._with_model_metadata(fact)
            try:
                return await super().on_model_request_error(
                    ctx,
                    request_context=request_context,
                    error=error,
                )
            finally:
                self._restore_metadata(original)
        return await super().on_model_request_error(
            ctx,
            request_context=request_context,
            error=error,
        )


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
    runtime_tool_names: tuple[str, ...],
    context_target_tokens: int | None,
    parent_step_run_id: str | None,
    plan_store_resolver: Callable[[PydanticRunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    workspace_read_available: bool = False,
    model_journal: "ModelRequestJournal | None" = None,
    model_observation_enabled: bool = False,
    model_request_observer: "ExternalModelRequestObserver | None" = None,
) -> tuple[AbstractCapability[None], ...]:
    capabilities: list[AbstractCapability[None]] = []
    persistence = _RuntimeStepPersistence(
        store=HarnessStepStoreAdapter(step_store, execution_id=execution_id),
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
    selected = frozenset(runtime_tool_names)
    selected_memory = tuple(name for name in MEMORY_TOOL_NAMES if name in selected)
    if selected_memory:
        if memory_store is None or memory_scope is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            build_harness_memory(
                memory_store,
                selected_tool_names=selected_memory,
                capability_id=_MEMORY_CAPABILITY_ID,
            )
        )
    if any(name in selected for name in PLANNING_TOOL_NAMES):
        if plan_store_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        def resolve_plan_store(
            ctx: PydanticRunContext[None],
        ) -> HarnessPlanStoreAdapter:
            store = plan_store_resolver(ctx)
            if store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return HarnessPlanStoreAdapter(store)

        capabilities.append(
            Planning(
                id=_PLANNING_CAPABILITY_ID,
                tools=PLANNING_TOOL_NAMES,
                store_resolver=resolve_plan_store,
            )
        )
    capabilities.append(
        RuntimeCompaction(
            context_target_tokens,
            workspace_read_available=workspace_read_available,
            journal=model_journal,
            observer=model_request_observer,
            projection_sink=persistence.remember_context_projection,
        )
    )
    return tuple(capabilities)


__all__ = [
    "MEMORY_READ_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "PLANNING_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "compose_platform_capabilities",
    "select_runtime_tool_names",
]
