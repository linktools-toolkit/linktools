#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime node-boundary adaptation of PydanticAI Harness StepPersistence.

Hook and record construction algorithms adapted from pydantic-ai-harness 0.66.0
step_persistence/_capability.py. Native public run binding, wrap_run, and
compaction identity remain intact. No external call runs inside a storage transaction.

The MIT License (MIT)

Copyright (c) 2026 Pydantic Services Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import anyio
from pydantic_ai import CallToolsNode, DeferredToolRequests, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability, durable_operation
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai_harness.step_persistence import (
    StepPersistence,
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    SnapshotSaved,
    SnapshotState,
    StepEvent,
    is_provider_valid,
)


@runtime_checkable
class AtomicStepStore(Protocol):
    """Commit a complete native safety point before reporting success.

    Implementations must commit all supplied records atomically or raise.
    Unknown outcomes require source readback using the native record keys.
    Individual native writes must not be silently buffered into this operation.
    Tool effects are deliberately outside this contract.
    """

    async def commit_step(
        self,
        *,
        run: RunRecord | None = None,
        events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        """Persist the optional run, ordered events and optional snapshot."""
        ...


@runtime_checkable
class RuntimeStepBoundaryStore(Protocol):
    """Native records bound to Runtime's leaf-owned effects and control fence."""

    @property
    def runtime_bound(self) -> bool: ...

    async def commit_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        """Confirm same-group publication or the explicit split-store handoff."""
        ...


class _BoundaryModel(WrapperModel):
    def __init__(self, wrapped: Model, publish: Callable[[], Awaitable[None]]) -> None:
        super().__init__(wrapped)
        self._publish = publish

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self._publish()
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        await self._publish()
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as response:
            yield response


def _snapshot_state(messages: list[ModelMessage]) -> SnapshotState:
    return "complete" if is_provider_valid(messages) else "interrupted"


def _has_model_response(messages: list[ModelMessage]) -> bool:
    return any(isinstance(message, ModelResponse) for message in messages)


@dataclass(kw_only=True)
class StepRunRegistration:
    """Prepare one native registration for all views of an owned attempt.

    Preparation is not commit confirmation. The native durable operation supplies
    the replay token; consumers may project the other fields before publication.
    """

    run_id: str
    parent_run_id: str | None
    agent_name: str | None
    metadata: dict[str, str]
    share_started: bool = False
    _record: RunRecord | None = field(default=None, init=False, repr=False)
    _started: StepEvent | None = field(default=None, init=False, repr=False)
    _prepare_token: Callable[[], Awaitable[tuple[str, datetime]]] | None = field(default=None, init=False, repr=False)

    def bind_registration(self, prepare_token: Callable[[], Awaitable[tuple[str, datetime]]]) -> None:
        """Use the native registration's existing durable identity operation."""
        self._prepare_token = prepare_token

    async def prepare_started(self, *, conversation_id: str | None, step_index: int) -> tuple[RunRecord, StepEvent]:
        """Derive the canonical start from one replay-stable registration clock."""
        if not self.share_started or self._prepare_token is None:
            raise RuntimeError("native start producer is not bound")
        if self._started is None:
            token, timestamp = await self._prepare_token()
            if (not isinstance(token, str) or not token or not isinstance(timestamp, datetime)
                    or timestamp.tzinfo is None):
                raise ValueError("native registration seed is invalid")
            record = self.prepare(conversation_id=conversation_id, registration_id=token, started_at=timestamp)
            self._started = StepEvent(
                run_id=record.run_id, kind="run_started", step_index=step_index,
                timestamp=timestamp, conversation_id=record.conversation_id,
                parent_run_id=record.parent_run_id, agent_name=record.agent_name,
                metadata=dict(record.metadata), idempotency_key=f"0:{step_index}:run_started:",
            )
        elif self._started.step_index != step_index or self._started.conversation_id != conversation_id:
            raise ValueError("native start boundary changed")
        prepared = self.prepared_started()
        assert prepared is not None
        return prepared

    def prepared_started(self) -> tuple[RunRecord, StepEvent] | None:
        """Return a prepared source, including before the native before_run hook."""
        return None if self._record is None or self._started is None else deepcopy((self._record, self._started))

    def bind(self, *, run_id: str, parent_run_id: str | None, agent_name: str | None) -> None:
        """Retain lineage resolved by the native capability before hooks run."""
        if run_id != self.run_id:
            raise ValueError("native registration run identity changed")
        if self._record is not None and (parent_run_id, agent_name) != (
                self._record.parent_run_id, self._record.agent_name):
            raise ValueError("native registration lineage changed after preparation")
        self.parent_run_id, self.agent_name = parent_run_id, agent_name

    def prepare(self, *, conversation_id: str | None, registration_id: str | None = None,
                started_at: datetime | None = None) -> RunRecord:
        record = self._record
        if record is None:
            record = RunRecord(
                run_id=self.run_id, conversation_id=conversation_id,
                parent_run_id=self.parent_run_id, agent_name=self.agent_name,
                metadata=dict(self.metadata), registration_id=registration_id,
                **({} if started_at is None else {"started_at": started_at}),
            )
        elif record.conversation_id != conversation_id:
            raise ValueError("native registration conversation changed")
        elif started_at is not None and record.started_at != started_at:
            raise ValueError("native registration clock changed after preparation")
        elif registration_id is not None:
            if record.registration_id not in (None, registration_id):
                raise ValueError("native registration replay identity changed")
            record = replace(record, registration_id=registration_id)
        self._record = record
        return deepcopy(record)


@dataclass
class AtomicStepPersistence(StepPersistence[AgentDepsT]):
    """Prepare native facts and publish explicit Runtime node boundaries.

    Hook callbacks prepare facts, not independent committed writes. The Runtime
    driver owns final leaf admission/result records and the control transaction.
    Standalone agents use the original StepPersistence capability.
    """

    run_registration: StepRunRegistration | None = field(default=None, repr=False, compare=False)

    _atomic_model_context: RunContext[AgentDepsT] | None = field(default=None, init=False, repr=False)
    _atomic_model_wrappers: tuple[type[AbstractCapability[Any]], ...] = field(default=(), repr=False)
    _atomic_pending_run: RunRecord | None = field(default=None, init=False, repr=False)
    _atomic_pending_events: list[StepEvent] = field(default_factory=list, init=False, repr=False)

    _atomic_start_published: bool = field(default=False, init=False, repr=False)
    _atomic_event_sequence: int = field(default=0, init=False, repr=False)
    _atomic_snapshot_sequence: int = field(default=0, init=False, repr=False)
    _atomic_last_snapshot_messages: list[ModelMessage] | None = field(
        default=None, init=False, repr=False
    )
    _atomic_last_snapshot_step: int | None = field(default=None, init=False, repr=False)
    _atomic_live_history: tuple[list[ModelMessage], int] | None = field(
        default=None, init=False, repr=False
    )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AtomicStepPersistence[AgentDepsT]:
        bound = await super().for_run(ctx)
        if not isinstance(bound, AtomicStepPersistence):
            raise TypeError("native run binding changed the step capability")
        if bound.run_registration is not None:
            bound.run_registration.bind(
                run_id=bound._atomic_effective_run_id(ctx),
                parent_run_id=bound.parent_run_id, agent_name=bound.agent_name,
            )
            bound.run_registration.bind_registration(bound._atomic_registration_seed)
        wrappers: list[type[AbstractCapability[Any]]] = []

        def collect(capability: AbstractCapability[Any]) -> None:
            cls = type(capability)
            if (not isinstance(capability, AtomicStepPersistence)
                    and cls.get_wrapper_model is not AbstractCapability.get_wrapper_model
                    and cls not in wrappers):
                wrappers.append(cls)

        root = ctx.root_capability
        if root is None and ctx.agent is not None:
            root = ctx.agent.root_capability
        if root is not None:
            root.apply(collect)
        bound._atomic_model_wrappers = tuple(wrappers)
        return bound

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost", wrapped_by=self._atomic_model_wrappers)

    async def prepare_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self._atomic_model_context = ctx
        return request_context

    def get_wrapper_model(self, model: Model) -> Model | None:
        ctx = self._atomic_model_context
        if ctx is None:
            return None
        return _BoundaryModel(model, lambda: self._atomic_publish_model(ctx))

    def _atomic_effective_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if self.run_id is not None:
            return self.run_id
        if ctx.run_id is not None:
            return ctx.run_id
        raise RuntimeError("StepPersistence run id was not materialized by `for_run`.")

    @durable_operation("atomic_registration_id")
    async def _atomic_registration_seed(self) -> tuple[str, datetime]:
        # Reuse the one registration operation. Durable backends retain only
        # identity and clock, never another copy of the run or event body.
        return str(uuid4()), datetime.now(UTC)

    def _atomic_prepare_event(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        kind: EventKind,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> StepEvent:
        if self.run_registration is not None and self.run_registration.share_started:
            self._atomic_event_sequence = max(1, self._atomic_event_sequence)
        event_index = self._atomic_event_sequence
        self._atomic_event_sequence += 1
        return StepEvent(
            run_id=self._atomic_effective_run_id(ctx),
            kind=kind,
            step_index=ctx.run_step,
            timestamp=datetime.now(UTC),
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            error=error,
            metadata=dict(self.metadata),
            idempotency_key=f"{event_index}:{ctx.run_step}:{kind}:{tool_call_id or ''}",
        )

    def _atomic_prepare_snapshot(
        self,
        ctx: RunContext[AgentDepsT],
        messages: list[ModelMessage],
        step_index: int,
        state: SnapshotState = "complete",
    ) -> ContinuableSnapshot:
        snapshot_index = self._atomic_snapshot_sequence
        self._atomic_snapshot_sequence += 1
        return ContinuableSnapshot(
            run_id=self._atomic_effective_run_id(ctx),
            step_index=step_index,
            messages=deepcopy(messages),
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            timestamp=datetime.now(UTC),
            state=state,
            idempotency_key=f"{snapshot_index}:{step_index}:{state}",
        )

    async def _atomic_commit(
        self,
        *,
        run: RunRecord | None = None,
        events: tuple[StepEvent, ...] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        if not isinstance(self.store, RuntimeStepBoundaryStore) or not self.store.runtime_bound:
            raise TypeError("AtomicStepPersistence requires a Runtime-bound step store")
        prepared_run = run if run is not None else self._atomic_pending_run
        pending_events = tuple(self._atomic_pending_events)
        shared_start = self.run_registration is not None and self.run_registration.share_started
        if shared_start and not self._atomic_start_published and prepared_run is None:
            prepared = self.run_registration.prepared_started()
            if prepared is not None:
                prepared_run, started = prepared
                pending_events = (started, *pending_events)
        prepared_events = (*pending_events, *events)
        await self._atomic_write(run=prepared_run, events=prepared_events, snapshot=snapshot)
        self._atomic_pending_run = None
        self._atomic_pending_events.clear()
        if shared_start:
            self._atomic_start_published = True

    @durable_operation("commit_boundary")
    async def _atomic_write(
        self, *, run: RunRecord | None, events: tuple[StepEvent, ...],
        snapshot: ContinuableSnapshot | None,
    ) -> None:
        await self.store.commit_step(run=run, events=events, snapshot=snapshot)

    async def _atomic_confirm_snapshot(
        self, ctx: RunContext[AgentDepsT], snapshot: ContinuableSnapshot
    ) -> None:
        # Rebuild on durable replay, but never advance before commit confirmation.
        self._atomic_last_snapshot_messages = deepcopy(snapshot.messages)
        self._atomic_last_snapshot_step = snapshot.step_index
        await ctx.emit(
            SnapshotSaved(
                persistence_run_id=snapshot.run_id,
                conversation_id=snapshot.conversation_id,
                step_index=snapshot.step_index,
                state=snapshot.state,
            )
        )

    async def after_run(
        self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        messages = result.all_messages()
        snapshot = None
        step = self._atomic_live_history[1] if self._atomic_live_history is not None else ctx.run_step
        interrupted = isinstance(result.output, DeferredToolRequests)
        if (self._atomic_last_snapshot_messages != messages or self._atomic_last_snapshot_step != step) and (
            interrupted or is_provider_valid(messages)
        ):
            # Deferred runs can end with confirmed sibling returns but an incomplete
            # tool round. Recovery reads this native owner, not the Trace projection.
            snapshot = self._atomic_prepare_snapshot(
                ctx, list(messages), step, "interrupted" if interrupted else "complete",
            )
        event = self._atomic_prepare_event(ctx, kind="run_completed")
        await self._atomic_commit(events=(event,), snapshot=snapshot)
        if snapshot is not None:
            await self._atomic_confirm_snapshot(ctx, snapshot)
        return result

    def _atomic_stash_live_history(self, ctx: RunContext[AgentDepsT]) -> None:
        self._atomic_live_history = (ctx.messages, ctx.run_step)

    async def on_run_error(
        self, ctx: RunContext[AgentDepsT], *, error: BaseException
    ) -> AgentRunResult[Any]:
        with anyio.CancelScope(shield=True):
            snapshot = None
            if self._atomic_live_history is not None:
                messages, step_index = self._atomic_live_history
                captured = list(messages)
                if _has_model_response(captured) or (
                    self.capture_frontier and captured
                ):
                    snapshot = self._atomic_prepare_snapshot(
                        ctx, captured, step_index, _snapshot_state(captured)
                    )
            event = self._atomic_prepare_event(
                ctx, kind="run_failed", error=repr(error)
            )
            await self._atomic_commit(events=(event,), snapshot=snapshot)
            if snapshot is not None:
                await self._atomic_confirm_snapshot(ctx, snapshot)
        raise error

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self._atomic_stash_live_history(ctx)
        self._atomic_pending_events.append(self._atomic_prepare_event(ctx, kind="model_request_started"))
        return request_context

    async def _atomic_publish_model(self, ctx: RunContext[AgentDepsT]) -> None:
        self._atomic_stash_live_history(ctx)
        messages = list(ctx.messages)
        snapshot = None
        if self.capture_frontier or (_has_model_response(messages) and is_provider_valid(messages)):
            snapshot = self._atomic_prepare_snapshot(ctx, messages, ctx.run_step, _snapshot_state(messages))
        await self._atomic_commit(snapshot=snapshot)
        if snapshot is not None:
            await self._atomic_confirm_snapshot(ctx, snapshot)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._atomic_pending_events.append(self._atomic_prepare_event(ctx, kind="model_request_completed"))
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        self._atomic_pending_events.append(self._atomic_prepare_event(
            ctx, kind="model_request_failed", error=repr(error)
        ))
        raise error

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        raise error

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        self._atomic_stash_live_history(ctx)
        messages = list(ctx.messages)
        if isinstance(node, ModelRequestNode):
            snapshot = (self._atomic_prepare_snapshot(
                ctx, messages, ctx.run_step, _snapshot_state(messages),
            ) if self.capture_frontier else None)
            await self._atomic_commit(snapshot=snapshot)
            if snapshot is not None:
                await self._atomic_confirm_snapshot(ctx, snapshot)
        if isinstance(node, CallToolsNode):
            if isinstance(result, ModelRequestNode):
                messages = [*messages, result.request]
            self._atomic_live_history = (messages, ctx.run_step)
        return result

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        if not isinstance(self.store, RuntimeStepBoundaryStore) or not self.store.runtime_bound:
            raise TypeError("AtomicStepPersistence requires a Runtime-bound step store; standalone agents use StepPersistence")
        if self.run_registration is not None and self.run_registration.share_started:
            self._atomic_pending_run, started = await self.run_registration.prepare_started(
                conversation_id=ctx.conversation_id, step_index=ctx.run_step,
            )
            self._atomic_pending_events.append(started)
            self._atomic_event_sequence = max(1, self._atomic_event_sequence)
            return
        registration_id, _ = await self._atomic_registration_seed()
        if self.run_registration is None:
            self._atomic_pending_run = RunRecord(
                run_id=self._atomic_effective_run_id(ctx),
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                metadata=dict(self.metadata),
                registration_id=registration_id,
            )
        else:
            self._atomic_pending_run = self.run_registration.prepare(
                conversation_id=ctx.conversation_id, registration_id=registration_id,
            )
        self._atomic_pending_events.append(self._atomic_prepare_event(ctx, kind="run_started"))


class StepBoundaryCapabilities(CombinedCapability[AgentDepsT]):
    """Resolve native run capabilities before placing the model publication leaf."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        resolved = await super().for_run(ctx)
        wrappers: list[type[AbstractCapability[Any]]] = []

        def collect(capability: AbstractCapability[Any]) -> None:
            cls = type(capability)
            if (not isinstance(capability, AtomicStepPersistence)
                    and cls.get_wrapper_model is not AbstractCapability.get_wrapper_model
                    and cls not in wrappers):
                wrappers.append(cls)

        resolved.apply(collect)

        def bind(capability: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
            if isinstance(capability, AtomicStepPersistence):
                return replace(capability, _atomic_model_wrappers=tuple(wrappers))
            return capability

        rebound = resolved.visit_and_replace(bind)
        if rebound is None:
            raise RuntimeError("native step boundary composition removed its capabilities")
        return rebound


__all__ = ["AtomicStepPersistence", "AtomicStepStore", "RuntimeStepBoundaryStore", "StepBoundaryCapabilities", "StepRunRegistration"]
