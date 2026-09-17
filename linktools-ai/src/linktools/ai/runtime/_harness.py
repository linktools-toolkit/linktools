#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness adapters for Runtime-owned durable state."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, cast

from linktools.core import environ
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
)
from pydantic_ai_harness.planning import (
    TaskStatus,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot as HarnessContinuableSnapshot,
)
from pydantic_ai_harness.step_persistence import (
    RunRecord as HarnessRunRecord,
)
from pydantic_ai_harness.step_persistence import (
    StepEvent as HarnessStepEvent,
)
from pydantic_ai_harness.step_persistence import (
    ToolEffectRecord,
)

from ..capability import ToolCallRetry
from ..core import UsageMetrics
from ..errors import AIError, ErrorCode
from ._journal import ModelRequestFact
from ._message import project_transient_binary_content
from ._model_interaction import (
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    build_context_projection,
    build_inline_context_projection,
    extend_prefix_digest,
    message_prefix_digest,
    model_identity,
    request_envelope,
)
from ._plan import PlanItem, RuntimePlanStore
from .state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
)

_logger = environ.get_logger("ai.runtime.harness")


class _InteractionStagingStore(Protocol):
    def intern_payload(self, run_id: str, payload: bytes) -> tuple[str, int]: ...

    def stage_model_interaction(self, interaction: object) -> None: ...


class HarnessPlanStoreAdapter:
    """Expose the Runtime plan store through Harness' public PlanStore contract."""

    def __init__(self, store: RuntimePlanStore) -> None:
        if not isinstance(store, RuntimePlanStore):
            raise TypeError("store must be RuntimePlanStore")
        self._store = store

    async def get_items(self) -> list[HarnessPlanItem]:
        return _harness_plan_items(await self._store.get_items())

    async def set_items(self, items: list[HarnessPlanItem]) -> None:
        await self._store.write_plan(_runtime_plan_items(items))

    async def get_item(self, item_id: str) -> HarnessPlanItem | None:
        return next(
            (item for item in await self.get_items() if item.id == item_id), None
        )

    async def add_item(self, item: HarnessPlanItem) -> HarnessPlanItem:
        candidate = item.model_copy(deep=True)

        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            if any(value.id == candidate.id for value in values):
                raise ValueError(
                    f"A step with id {candidate.id!r} is already in this plan."
                )
            values.append(candidate.model_copy(deep=True))
            return _runtime_plan_items(values)

        await self._store.edit_items(edit)
        return candidate.model_copy(deep=True)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> HarnessPlanItem | None:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            selected = next((item for item in values if item.id == item_id), None)
            if selected is None:
                return current
            if content is not None:
                selected.content = content
            if status is not None:
                selected.status = status
            if active_form is not None:
                selected.active_form = active_form
            if parent_id is not None:
                selected.parent_id = parent_id
            if depends_on is not None:
                selected.depends_on = list(depends_on)
            return _runtime_plan_items(values)

        previous, current, _revision = await self._store.edit_items(edit)
        if not any(item.id == item_id for item in _harness_plan_items(previous)):
            return None
        selected = next(
            (item for item in _harness_plan_items(current) if item.id == item_id),
            None,
        )
        if selected is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if active_form is not None:
            selected.active_form = active_form
        if depends_on is not None:
            selected.depends_on = list(depends_on)
        return selected.model_copy(deep=True)

    async def remove_item(self, item_id: str) -> bool:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            return _runtime_plan_items(
                [item for item in values if item.id != item_id]
            )

        previous, _current, _revision = await self._store.edit_items(edit)
        return any(item.id == item_id for item in _harness_plan_items(previous))


def _harness_plan_items(items: Sequence[PlanItem]) -> list[HarnessPlanItem]:
    return [
        HarnessPlanItem(
            id=_plan_item_id(index),
            content=item.content,
            status=TaskStatus(item.status),
        )
        for index, item in enumerate(items)
    ]


def _runtime_plan_items(items: list[HarnessPlanItem]) -> list[PlanItem]:
    if not isinstance(items, list):
        raise ToolCallRetry(
            "Plan items are invalid for this runtime. Correct the plan and retry."
        )
    values: list[PlanItem] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if (
            not isinstance(item.parent_id, (str, type(None)))
            or not isinstance(item.depends_on, list)
            or any(not isinstance(value, str) for value in item.depends_on)
            or item.parent_id is not None
            or item.depends_on
        ):
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.status, TaskStatus):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if item.status is TaskStatus.blocked:
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.content, str) or not item.content.strip():
            raise ToolCallRetry(
                "Each plan item must have non-empty content. Fill every item and retry."
            )
        if item.status not in {
            TaskStatus.pending,
            TaskStatus.in_progress,
            TaskStatus.completed,
            TaskStatus.cancelled,
        }:
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        values.append(PlanItem(item.content, cast(str, item.status.value)))
    return values


def _plan_item_id(index: int) -> str:
    return f"item-{index + 1}"


class HarnessStepStoreAdapter:
    """Persist Harness StepPersistence facts through the Runtime StepStore contract."""

    def __init__(
        self,
        store: StepStore,
        *,
        execution_id: str | None,
        step_run_id: str | None = None,
    ) -> None:
        self._store = store
        self._interaction_store = cast(_InteractionStagingStore, store)
        self._execution_id = execution_id
        self._step_run_id = step_run_id
        self._effects: dict[tuple[str, str], ToolEffectRecord] = {}
        self._effects_lock = asyncio.Lock()
        self._interrupted_runs: set[str] = set()
        self._projection_source: tuple[ModelMessage, ...] | None = None
        self._projection_messages: tuple[ModelMessage, ...] | None = None
        self._prefix_source: tuple[ModelMessage, ...] | None = None
        self._prefix_digest: str | None = None
        self._interaction_projections: dict[int, StagedContextProjection] = {}
        self._interaction_requests: dict[int, tuple[ModelMessage, ...]] = {}
        self._interaction_payloads: dict[int, str] = {}
        self._interaction_runs: dict[int, str] = {}
        self._interaction_models: dict[int, dict[str, str]] = {}

    async def register_run(self, record: HarnessRunRecord) -> None:
        value = RunRecord(
            run_id=record.run_id,
            conversation_id=record.conversation_id,
            parent_run_id=record.parent_run_id,
            agent_name=record.agent_name,
            metadata=dict(record.metadata),
            started_at=record.started_at,
        )
        if self._execution_id is None:
            await self._store.register_run(value)
        else:
            await self._store.register_run(value, execution_id=self._execution_id)

    async def get_run(self, *, run_id: str) -> HarnessRunRecord | None:
        record = await self._store.get_run(run_id=run_id)
        return None if record is None else _harness_run(record)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[HarnessRunRecord]:
        values = await self._store.list_runs(
            parent_run_id=parent_run_id,
            conversation_id=conversation_id,
        )
        return [_harness_run(record) for record in values]

    async def append_event(self, event: HarnessStepEvent) -> None:
        kind = (
            "run_interrupted"
            if event.kind == "run_completed" and event.run_id in self._interrupted_runs
            else event.kind
        )
        await self._append_step_event(
            StepEvent(
                run_id=event.run_id,
                kind=cast(object, kind),  # type: ignore[arg-type]
                step_index=event.step_index,
                timestamp=event.timestamp,
                conversation_id=event.conversation_id,
                parent_run_id=event.parent_run_id,
                agent_name=event.agent_name,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                error=event.error,
                metadata=dict(event.metadata),
            )
        )

    async def list_events(self, *, run_id: str) -> list[HarnessStepEvent]:
        values = await self._store.list_events(run_id=run_id)
        result: list[HarnessStepEvent] = []
        for event in values:
            kind = "run_completed" if event.kind == "run_interrupted" else event.kind
            result.append(
                HarnessStepEvent(
                    run_id=event.run_id,
                    kind=cast(object, kind),  # type: ignore[arg-type]
                    step_index=event.step_index,
                    timestamp=event.timestamp,
                    conversation_id=event.conversation_id,
                    parent_run_id=event.parent_run_id,
                    agent_name=event.agent_name,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    error=event.error,
                    metadata=dict(event.metadata),
                )
            )
        return result

    async def save_snapshot(self, snapshot: HarnessContinuableSnapshot) -> None:
        state = (
            "interrupted"
            if snapshot.run_id in self._interrupted_runs
            else snapshot.state
        )
        await self._save_step_snapshot(
            ContinuableSnapshot(
                run_id=snapshot.run_id,
                step_index=snapshot.step_index,
                messages=list(snapshot.messages),
                conversation_id=snapshot.conversation_id,
                parent_run_id=snapshot.parent_run_id,
                agent_name=snapshot.agent_name,
                timestamp=snapshot.timestamp,
                state=state,
                context_messages=self.snapshot_context_messages(snapshot.messages),
            )
        )

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> HarnessContinuableSnapshot | None:
        snapshot = await self._store.latest_snapshot(
            run_id=run_id,
            include_interrupted=include_interrupted,
        )
        return None if snapshot is None else _harness_snapshot(snapshot)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        async with self._effects_lock:
            self._effects[(record.run_id, record.tool_call_id)] = replace(record)

    async def get_tool_effect(
        self,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        async with self._effects_lock:
            value = self._effects.get((run_id, tool_call_id))
            return None if value is None else replace(value)

    async def list_unresolved_tool_effects(
        self,
        *,
        run_id: str,
    ) -> list[ToolEffectRecord]:
        async with self._effects_lock:
            return [
                replace(value)
                for (candidate_run_id, _), value in self._effects.items()
                if candidate_run_id == run_id and value.status == "started"
            ]

    def mark_interrupted(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._interrupted_runs.add(run_id)

    async def save_interrupted_snapshot(
        self,
        *,
        run_id: str,
        step_index: int,
        messages: Sequence[ModelMessage],
        conversation_id: str | None,
        parent_run_id: str | None,
        agent_name: str | None,
    ) -> None:
        values = list(messages)
        await self._save_step_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=step_index,
                messages=values,
                conversation_id=conversation_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                state="interrupted",
                idempotency_key=f"pause:{step_index}:{len(values)}",
                context_messages=self.snapshot_context_messages(values),
            )
        )

    async def _append_step_event(self, event: StepEvent) -> None:
        if self._execution_id is None:
            await self._store.append_event(event)
        else:
            await self._store.append_event(event, execution_id=self._execution_id)

    async def _save_step_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self._execution_id is None:
            await self._store.save_snapshot(snapshot)
        else:
            await self._store.save_snapshot(
                snapshot,
                execution_id=self._execution_id,
            )

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        source_values = tuple(source)
        if projected is not None:
            self._projection_source = source_values
            self._projection_messages = tuple(projected)
            return
        current = self._projection_messages
        if current is not None and _message_prefix(source_values, current):
            self._projection_source = source_values
            self._projection_messages = source_values
            return
        self._projection_source = None
        self._projection_messages = None

    def capture_model_context(self, messages: Sequence[ModelMessage]) -> None:
        projected = self._projection_messages
        if projected is None:
            return
        message_values = tuple(messages)
        if _message_prefix(message_values, projected):
            source = self._projection_source
            self._projection_messages = message_values
            self._projection_source = (
                message_values
                if source is None
                else (*source, *message_values[len(projected) :])
            )
            return
        source = self._projection_source
        if source is not None and _message_prefix(message_values, source):
            self._projection_messages = (
                *projected,
                *message_values[len(source) :],
            )
            return
        self._projection_source = None
        self._projection_messages = None

    def begin_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        run_id = self._step_run_id or self._run_id_from_messages(messages)
        if run_id is None:
            run_id = self._execution_id
        if not isinstance(run_id, str) or not run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        projected = tuple(messages)
        if source_messages is not None:
            source = tuple(source_messages)
            projection = build_context_projection(
                source,
                projected,
                lambda payload: self._interaction_store.intern_payload(run_id, payload),
                source_prefix_digest=self._source_prefix_digest(source),
            )
        elif fact.purpose == "compaction":
            projection = build_inline_context_projection(
                projected,
                lambda payload: self._interaction_store.intern_payload(run_id, payload),
            )
        else:
            source = self._projection_source or projected
            projection = build_context_projection(
                source,
                projected,
                lambda payload: self._interaction_store.intern_payload(run_id, payload),
                source_prefix_digest=self._source_prefix_digest(source),
            )
        _logger.debug(
            "model interaction context staged: run=%s sequence=%s "
            "purpose=%s source_count=%s source_digest=%s",
            run_id,
            fact.request_sequence,
            fact.purpose,
            projection.source_message_count,
            projection.source_prefix_digest,
        )
        _envelope, envelope_bytes = request_envelope(
            model_settings=model_settings,
            parameters=parameters,
            streaming=streaming,
        )
        digest, _size = self._interaction_store.intern_payload(run_id, envelope_bytes)
        self._interaction_projections[fact.request_sequence] = projection
        self._interaction_requests[fact.request_sequence] = projected
        self._interaction_payloads[fact.request_sequence] = digest
        self._interaction_runs[fact.request_sequence] = run_id
        self._interaction_models[fact.request_sequence] = model_identity(
            model,
            route_id=model_id,
        )

    def finish_model_interaction(
        self,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        duration_ns: int,
        usage: object | None,
    ) -> None:
        del model
        request_sequence = fact.request_sequence
        projection = self._interaction_projections.pop(request_sequence)
        request_messages = self._interaction_requests.pop(request_sequence)
        run_id = self._interaction_runs.pop(request_sequence)
        envelope_digest = self._interaction_payloads.pop(request_sequence)
        model_value = self._interaction_models.pop(request_sequence)
        if status != "SUCCEEDED":
            projection = build_inline_context_projection(
                request_messages,
                lambda payload: self._interaction_store.intern_payload(run_id, payload),
            )
        response_projection = None
        if response is not None:
            if fact.purpose == "compaction":
                response_projection = build_inline_context_projection(
                    (response,),
                    lambda payload: self._interaction_store.intern_payload(run_id, payload),
                )
            else:
                response_projection = StagedContextProjection(
                    projection.source_message_count,
                    projection.source_prefix_digest,
                    (
                        StagedContextSpan(
                            projection.source_message_count,
                            projection.source_message_count + 1,
                        ),
                    ),
                )
        self._interaction_store.stage_model_interaction(
            StagedModelInteraction(
                run_id,
                fact.step_index,
                request_sequence,
                fact.purpose,
                fact.output_retry_index,
                model_value,
                projection,
                envelope_digest,
                response_projection,
                status,
                error_code,
                duration_ns,
                _usage_metrics(usage),
            )
        )

    def _run_id_from_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> str | None:
        for message in messages:
            run_id = getattr(message, "run_id", None)
            if isinstance(run_id, str) and run_id:
                return run_id
        return None

    def _source_prefix_digest(
        self,
        source: Sequence[ModelMessage],
    ) -> str:
        values = tuple(source)
        prefix = self._prefix_source
        digest = self._prefix_digest
        if prefix is not None and digest is not None and len(values) >= len(prefix):
            if values[: len(prefix)] == prefix:
                for message in values[len(prefix) :]:
                    digest = extend_prefix_digest(digest, message)
                self._prefix_source = values
                self._prefix_digest = digest
                return digest
        digest = message_prefix_digest(values)
        self._prefix_source = values
        self._prefix_digest = digest
        return digest

    def snapshot_context_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> list[ModelMessage] | None:
        message_values = tuple(messages)
        source = self._projection_source
        projected = self._projection_messages
        context_values = message_values
        has_projection = False
        if projected is not None:
            if _message_prefix(message_values, projected):
                has_projection = True
            elif source is not None and _message_prefix(message_values, source):
                context_values = (
                    *projected,
                    *message_values[len(source) :],
                )
                has_projection = True
        binary_projected = project_transient_binary_content(context_values)
        if binary_projected != context_values:
            context_values = binary_projected
            has_projection = True
        return list(context_values) if has_projection else None


def _usage_metrics(value: object | None) -> UsageMetrics | None:
    if value is None:
        return None
    return UsageMetrics(
        model_requests=1,
        input_tokens=int(getattr(value, "input_tokens", 0)),
        output_tokens=int(getattr(value, "output_tokens", 0)),
        cache_read_tokens=int(getattr(value, "cache_read_tokens", 0)),
        cache_write_tokens=int(getattr(value, "cache_write_tokens", 0)),
    )


def _harness_run(record: RunRecord) -> HarnessRunRecord:
    return HarnessRunRecord(
        run_id=record.run_id,
        conversation_id=record.conversation_id,
        parent_run_id=record.parent_run_id,
        agent_name=record.agent_name,
        metadata=dict(record.metadata),
        started_at=record.started_at,
    )


def _harness_snapshot(snapshot: ContinuableSnapshot) -> HarnessContinuableSnapshot:
    return HarnessContinuableSnapshot(
        run_id=snapshot.run_id,
        step_index=snapshot.step_index,
        messages=list(snapshot.messages),
        conversation_id=snapshot.conversation_id,
        parent_run_id=snapshot.parent_run_id,
        agent_name=snapshot.agent_name,
        timestamp=snapshot.timestamp,
        state=snapshot.state,
    )


def _message_prefix(
    values: Sequence[ModelMessage],
    prefix: Sequence[ModelMessage],
) -> bool:
    return len(values) >= len(prefix) and tuple(values[: len(prefix)]) == tuple(prefix)


__all__ = [
    "HarnessPlanStoreAdapter",
    "HarnessStepStoreAdapter",
]
