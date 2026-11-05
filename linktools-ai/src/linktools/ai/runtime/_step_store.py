#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-bound native storage view with Runtime's single tool-effect owner."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from ._tool import RuntimeToolOperationBridge, ToolOperationBridge
from ..model.messages import native_checkpoint_messages
from ..capability.steps import AtomicStepStore

if TYPE_CHECKING:
    from ..persistence.runs import RuntimeAgentRunStore


class StepBoundaryCommitter(Protocol):
    async def __call__(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        """Return only after the complete Runtime boundary commit is confirmed."""
        ...


class RuntimeStepStore:
    """Borrow step storage while routing this attempt's effects to dispatch."""

    def __init__(self, store: StepStore, tool_operations: ToolOperationBridge, *,
                 working_set: "RuntimeAgentRunStore | None" = None,
                 commit_boundary: StepBoundaryCommitter | None = None,
                 native_boundary: bool = False) -> None:
        self._store = store
        self._tool_operations = tool_operations
        self._working_set = working_set
        self._commit_boundary = commit_boundary
        self._native_boundary = native_boundary
        if native_boundary and isinstance(tool_operations, RuntimeToolOperationBridge):
            tool_operations.bind_native_step_store(store)

    @property
    def runtime_bound(self) -> bool:
        return self._tool_operations is not None

    async def commit_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        writer = self._commit_boundary
        if writer is None:
            writer = (self._store.commit_step if isinstance(self._store, AtomicStepStore)
                      else self._commit_split_step)
        stored_snapshot = None if snapshot is None else replace(
            snapshot, messages=native_checkpoint_messages(snapshot.messages),
        )
        try:
            await writer(run=run, events=events, snapshot=stored_snapshot)
        except BaseException:
            if self._working_set is not None:
                for run_id in {record.run_id for record in (run, snapshot, *events) if record is not None}:
                    self._working_set.invalidate_step_run(run_id)
            raise
        if run is not None and self._native_boundary and isinstance(self._tool_operations, RuntimeToolOperationBridge):
            self._tool_operations.confirm_native_run(run)
        if self._working_set is not None:
            if run is not None:
                self._working_set.confirm_step_run(run)
            if snapshot is not None:
                self._working_set.confirm_step_snapshot(snapshot)

    async def _commit_split_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        """Publish original individual writes for a caller-owned native store.

        This adapter is deliberately non-atomic. Each successful native write
        remains durable if a later write fails; keyed source repair is required.
        """
        if run is not None:
            existing = await self._store.get_run(run_id=run.run_id)
            if existing is None:
                await self._store.register_run(run)
            elif existing.registration_id != run.registration_id:
                raise ValueError("native run_id is single-shot")
        for event in events:
            await self._store.append_event(event)
        if snapshot is not None:
            await self._store.save_snapshot(snapshot)

    async def register_run(self, record: RunRecord) -> None:
        try:
            await self._store.register_run(record)
        except BaseException:
            if self._working_set is not None:
                self._working_set.invalidate_step_run(record.run_id)
            raise
        if self._working_set is not None:
            self._working_set.confirm_step_run(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await self._store.get_run(run_id=run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await self._store.list_runs(
            parent_run_id=parent_run_id, conversation_id=conversation_id,
        )

    async def append_event(self, event: StepEvent) -> None:
        await self._store.append_event(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await self._store.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        try:
            await self._store.save_snapshot(replace(snapshot, messages=native_checkpoint_messages(snapshot.messages)))
        except BaseException:
            if self._working_set is not None:
                self._working_set.invalidate_step_run(snapshot.run_id)
            raise
        if self._working_set is not None:
            self._working_set.confirm_step_snapshot(snapshot)

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        snapshot = await self._store.latest_snapshot(
            run_id=run_id, include_interrupted=include_interrupted,
        )
        return None if snapshot is None else replace(
            snapshot, messages=native_checkpoint_messages(snapshot.messages, restore=True),
        )

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        if not self._native_boundary:
            # Original/custom native hooks observe lifecycle; only the actual
            # authorized leaf may create or settle the business operation.
            current = await self._tool_operations.get_tool_effect(
                run_id=record.run_id, tool_call_id=record.tool_call_id,
            )
            if current is None:
                if record.idempotency_key is not None or record.effect_summary is not None:
                    raise RuntimeError("tool annotations require an admitted effect")
                return
            if current.tool_name != record.tool_name:
                raise ValueError("native tool observation changed the admitted tool")
            # An error hook may report failure after an uncertain effect. It
            # cannot turn that observation into a new outcome or annotation.
            if record.status != current.status:
                return
            if (record.idempotency_key in {None, current.idempotency_key}
                    and record.effect_summary in {None, current.effect_summary}):
                return
            record = replace(current,
                idempotency_key=record.idempotency_key or current.idempotency_key,
                effect_summary=record.effect_summary if record.effect_summary is not None else current.effect_summary,
            )
        await self._tool_operations.record_tool_effect(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._tool_operations.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await self._tool_operations.list_unresolved_tool_effects(run_id=run_id)


__all__ = ["RuntimeStepStore", "StepBoundaryCommitter"]
