#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from linktools.core import environ
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
from pydantic_ai_harness.step_persistence import StepPersistence

from ..errors import AIError, ErrorCode
from ._compaction import (
    ExternalModelRequestObserver,
    RuntimeCompaction,
    RuntimeCompactionPolicy,
)
from ._harness import (
    HarnessPlanStoreAdapter,
    HarnessStepStoreAdapter,
)
from ._harness_memory import (
    build_harness_memory,
    select_harness_memory_tools,
)
from ._harness_planning import build_harness_planning
from ._journal import (
    MODEL_USAGE_CACHE_READ_METADATA_KEY,
    MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    MODEL_USAGE_INPUT_METADATA_KEY,
    MODEL_USAGE_OUTPUT_METADATA_KEY,
)
from ._memory import MemoryStore
from ._plan import RuntimePlanStore
from .state._step_contracts import StepStore

if TYPE_CHECKING:
    from ._journal import ModelRequestFact, ModelRequestJournal

_MEMORY_CAPABILITY_ID = "linktools.ai.memory"
_logger = environ.get_logger("ai.runtime.capabilities")


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
                    MODEL_USAGE_INPUT_METADATA_KEY: str(
                        usage.input_tokens
                    ),
                    MODEL_USAGE_OUTPUT_METADATA_KEY: str(
                        usage.output_tokens
                    ),
                    MODEL_USAGE_CACHE_READ_METADATA_KEY: str(
                        usage.cache_read_tokens
                    ),
                    MODEL_USAGE_CACHE_WRITE_METADATA_KEY: str(
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
    ordinary_tool_policy: tuple[str, ...],
    compaction_policy: RuntimeCompactionPolicy,
    planning: bool,
    context_target_tokens: int | None,
    parent_step_run_id: str | None,
    plan_store_resolver: Callable[[PydanticRunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    model_journal: "ModelRequestJournal | None" = None,
    model_observation_enabled: bool = False,
    model_request_observer: "ExternalModelRequestObserver | None" = None,
) -> tuple[AbstractCapability[None], ...]:
    capabilities: list[AbstractCapability[None]] = []
    persistence = _RuntimeStepPersistence(
        id="linktools.ai.step-persistence",
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
            policy=compaction_policy,
            journal=model_journal,
            observer=model_request_observer,
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
