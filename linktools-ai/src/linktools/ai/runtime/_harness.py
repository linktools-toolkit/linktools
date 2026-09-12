#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness adapters for Runtime-owned durable state."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
    TaskStatus,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot as HarnessContinuableSnapshot,
    RunRecord as HarnessRunRecord,
    StepEvent as HarnessStepEvent,
    ToolEffectRecord,
)

from ..errors import AIError, ErrorCode
from ._plan import PlanItem, RuntimePlanStore
from .state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
)


class HarnessPlanStoreAdapter:
    """Expose the Runtime plan store through Harness' public PlanStore contract."""

    def __init__(self, store: RuntimePlanStore) -> None:
        if not isinstance(store, RuntimePlanStore):
            raise TypeError("store must be RuntimePlanStore")
        self._store = store

    async def get_items(self) -> list[HarnessPlanItem]:
        values = await self._store.get_items()
        return [
            HarnessPlanItem(
                id=_plan_item_id(index),
                content=item.content,
                status=TaskStatus(item.status),
            )
            for index, item in enumerate(values)
        ]

    async def set_items(self, items: list[HarnessPlanItem]) -> None:
        await self._store.write_plan(_runtime_plan_items(items))

    async def get_item(self, item_id: str) -> HarnessPlanItem | None:
        return next(
            (item for item in await self.get_items() if item.id == item_id), None
        )

    async def add_item(self, item: HarnessPlanItem) -> HarnessPlanItem:
        values = await self.get_items()
        if any(current.id == item.id for current in values):
            raise ValueError(f"A step with id {item.id!r} is already in this plan.")
        values.append(item.model_copy(deep=True))
        await self.set_items(values)
        return item.model_copy(deep=True)

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
        values = await self.get_items()
        selected = next((item for item in values if item.id == item_id), None)
        if selected is None:
            return None
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
        await self.set_items(values)
        return selected.model_copy(deep=True)

    async def remove_item(self, item_id: str) -> bool:
        values = await self.get_items()
        next_values = [item for item in values if item.id != item_id]
        if len(next_values) == len(values):
            return False
        await self.set_items(next_values)
        return True


def _runtime_plan_items(items: list[HarnessPlanItem]) -> list[PlanItem]:
    values: list[PlanItem] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise TypeError("plan items must be Harness PlanItem values")
        if (
            item.status is TaskStatus.blocked
            or item.parent_id is not None
            or item.depends_on
        ):
            raise ValueError("subtask planning is not enabled")
        values.append(PlanItem(item.content, cast(str, item.status.value)))
    return values


def _plan_item_id(index: int) -> str:
    return f"item-{index + 1}"


class HarnessStepStoreAdapter:
    """Persist Harness StepPersistence facts through the Runtime StepStore contract."""

    def __init__(self, store: StepStore, *, execution_id: str | None) -> None:
        self._store = store
        self._execution_id = execution_id
        self._effects: dict[tuple[str, str], ToolEffectRecord] = {}
        self._effects_lock = asyncio.Lock()
        self._interrupted_runs: set[str] = set()
        self._projection_source: tuple[ModelMessage, ...] | None = None
        self._projection_messages: tuple[ModelMessage, ...] | None = None

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
            self._projection_messages = message_values
            self._projection_source = message_values
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

    def snapshot_context_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> list[ModelMessage] | None:
        source = self._projection_source
        projected = self._projection_messages
        if projected is None:
            return None
        message_values = tuple(messages)
        if _message_prefix(message_values, projected):
            return list(message_values)
        if source is None or not _message_prefix(message_values, source):
            return None
        return [*projected, *message_values[len(source) :]]


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
