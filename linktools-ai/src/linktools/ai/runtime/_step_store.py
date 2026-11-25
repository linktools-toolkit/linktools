#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-bound native storage view with Runtime's single tool-effect owner."""

from collections.abc import Callable, Sequence
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
from ..core import deterministic_id
from ..errors import AIError, ErrorCode
from ..model.messages import native_checkpoint_messages
from ..capability.steps import AtomicStepStore
from ..persistence import MessageArchive, MessageSnapshot, SqlStepStore, StepWriteReceipt
from ..persistence.runs import NativeCheckpoint

if TYPE_CHECKING:
    from ..persistence.runs import RuntimeAgentRunStore
    from ..persistence.state import StateStore
    from ..persistence.turns import SessionTurnKey, SqlSessionTurns
    from ._canonical_split_step import _CanonicalSplitStep


class StepBoundaryCommitter(Protocol):
    async def __call__(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> StepWriteReceipt | None:
        """Return only after the complete Runtime boundary commit is confirmed."""
        ...


class RuntimeStepStore:
    """Borrow step storage while routing this attempt's effects to dispatch."""

    def __init__(self, store: StepStore, tool_operations: ToolOperationBridge, *,
                 working_set: "RuntimeAgentRunStore | None" = None,
                 commit_boundary: StepBoundaryCommitter | None = None,
                 native_boundary: bool = False,
                 output_retries_used: Callable[[], int] | None = None) -> None:
        self._store = store
        self._tool_operations = tool_operations
        self._working_set = working_set
        self._commit_boundary = commit_boundary
        self._canonical_turn_boundary = False
        self._canonical_split_source: _CanonicalSplitStep | None = None
        self._interrupted_checkpoint: str | None = None
        self._resume_generation: int | None = None
        self._resume_step_base = 0
        self._resume_run: RunRecord | None = None
        self._native_boundary = native_boundary
        self._output_retries_used = output_retries_used
        self._message_archive = MessageArchive()
        self._checkpoint_messages: tuple[MessageSnapshot, MessageSnapshot] | None = None
        if native_boundary and isinstance(tool_operations, RuntimeToolOperationBridge):
            tool_operations.bind_native_step_store(store)

    @classmethod
    def for_canonical_turn(
        cls, native: SqlStepStore, tool_operations: RuntimeToolOperationBridge, *,
        turn_store: "SqlSessionTurns", state_store: "StateStore",
        key: "SessionTurnKey", generation: int, run_id: str,
        output_retries_used: Callable[[], int] | None = None,
        split_source: "_CanonicalSplitStep | None" = None,
    ) -> "RuntimeStepStore":
        """Publish checkpoints under the selected canonical turn authority."""
        group = state_store.storage_group
        context = getattr(group, "context", None)
        if context is None or native.storage_context.engine is not context.engine:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY, "Step and its write authority require one SQL transaction")
        if run_id != key.turn_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if split_source is not None and (
            split_source._turns is not turn_store or split_source._store is not state_store
            or split_source._context is not native.storage_context
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def guard(transaction):
            if split_source is None:
                await turn_store.guard_step_in_transaction(transaction, key, generation=generation)
            else:
                await split_source.guard(transaction, key, generation=generation)

        async def commit(*, run=None, events=(), snapshot=None):
            records = (() if run is None else (run,)) + tuple(events) + (() if snapshot is None else (snapshot,))
            if not records:
                return None
            if any(record.run_id != run_id for record in records):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if store._resume_run is not None:
                # AtomicStepPersistence emits a run_started draft for every SDK
                # invocation. The existing native registration owns this turn.
                run = None
                events = tuple(store._resume_event(event) for event in events if event.kind != "run_started")
                if snapshot is not None:
                    snapshot = store._resume_snapshot(snapshot)
                if not events and snapshot is None:
                    return None
            prepared = await native.prepare_step(run=run, events=events, snapshot=snapshot)

            async def write(transaction):
                return await native.write_prepared_step(transaction.transaction(state_store), prepared, turn_guard=guard)

            if split_source is None:
                receipt = await group.mutate((state_store,), write)
                # A provisional or unknown commit must not seed native caches.
                native.confirm_prepared_step(receipt)
            else:
                # Split source owns initial adoption and cache confirmation.
                receipt = await split_source.commit_step(prepared, key=key, generation=generation)
            if snapshot is not None and snapshot.state == "interrupted":
                store._interrupted_checkpoint = snapshot.idempotency_key
            return receipt

        tool_operations.prepare_canonical_boundaries(
            native, state_store=state_store, key=key, generation=generation, guard=guard,
            prepare_event=lambda event: store._resume_event(event),
        )
        store = cls(native, tool_operations, commit_boundary=commit, native_boundary=True,
                    output_retries_used=output_retries_used)
        store._canonical_turn_boundary = True
        store._canonical_split_source = split_source
        return store

    def require_interrupted_checkpoint(self) -> str:
        """Return the exact confirmed checkpoint occurrence, never the latest query."""
        if not self._canonical_turn_boundary or self._interrupted_checkpoint is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return self._interrupted_checkpoint

    async def bind_canonical_resume(
        self, snapshot: ContinuableSnapshot, *, key: "SessionTurnKey", generation: int,
    ) -> None:
        if (not self._canonical_turn_boundary or not isinstance(self._tool_operations, RuntimeToolOperationBridge)
                or self._resume_run is not None or generation < 2
                or snapshot.run_id != key.turn_id or snapshot.conversation_id != key.session_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        existing = await self._store.get_run(run_id=key.turn_id)
        if existing is None or existing.conversation_id != key.session_id:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        self._resume_run = existing
        self._resume_generation = generation
        self._resume_step_base = snapshot.step_index
        self._tool_operations.confirm_native_run(existing)

    def _resume_event(self, event: StepEvent) -> StepEvent:
        if self._resume_generation is None:
            return event
        return replace(event, step_index=event.step_index + self._resume_step_base,
                       idempotency_key=deterministic_id("canonical-step-event", self._resume_generation,
                                                        event.idempotency_key)
                       if event.idempotency_key is not None else None)

    def _resume_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        return replace(snapshot, step_index=snapshot.step_index + self._resume_step_base,
                       idempotency_key=deterministic_id("canonical-step-snapshot", self._resume_generation,
                                                        snapshot.idempotency_key)
                       if snapshot.idempotency_key is not None else None)

    @property
    def runtime_bound(self) -> bool:
        return self._tool_operations is not None

    async def commit_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> None:
        writer = self._commit_boundary
        if writer is None:
            writer = (self._store.commit_step_receipt if isinstance(self._store, SqlStepStore)
                      else self._store.commit_step if isinstance(self._store, AtomicStepStore)
                      else self._commit_split_step)
        stored_snapshot = None if snapshot is None else self._encode_snapshot(snapshot)
        try:
            receipt = await writer(run=run, events=events, snapshot=stored_snapshot)
        except BaseException:
            if self._working_set is not None:
                for run_id in {record.run_id for record in (run, snapshot, *events) if record is not None}:
                    self._working_set.invalidate_step_run(run_id)
            raise
        source = None if run is None else self._resume_run
        if source is None and run is not None:
            source = (self._store.run_from_receipt(receipt)
                  if isinstance(self._store, SqlStepStore) and isinstance(receipt, StepWriteReceipt)
                  else receipt if isinstance(receipt, RunRecord) else run)
        if source is not None and source.run_id != run.run_id:
            raise ValueError("native receipt changed the requested run identity")
        if source is not None and self._native_boundary and isinstance(self._tool_operations, RuntimeToolOperationBridge):
            self._tool_operations.confirm_native_run(source)
        if self._working_set is not None:
            if source is not None:
                self._working_set.confirm_step_run(source, receipt=(
                    receipt if isinstance(receipt, StepWriteReceipt) else None))
            if snapshot is not None:
                self._working_set.confirm_step_snapshot(stored_snapshot)

    async def _commit_split_step(
        self, *, run: RunRecord | None = None, events: Sequence[StepEvent] = (),
        snapshot: ContinuableSnapshot | None = None,
    ) -> RunRecord | None:
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
            else:
                run = existing
        for event in events:
            await self._store.append_event(event)
        if snapshot is not None:
            await self._store.save_snapshot(snapshot)
        return run

    async def register_run(self, record: RunRecord) -> None:
        if self._canonical_turn_boundary:
            await self.commit_step(run=record)
            return
        receipt = None
        try:
            if isinstance(self._store, SqlStepStore):
                receipt = await self._store.register_run_receipt(record)
            else:
                await self._store.register_run(record)
        except BaseException:
            if self._working_set is not None:
                self._working_set.invalidate_step_run(record.run_id)
            raise
        if self._working_set is not None:
            self._working_set.confirm_step_run(record, receipt=receipt)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await self._store.get_run(run_id=run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await self._store.list_runs(
            parent_run_id=parent_run_id, conversation_id=conversation_id,
        )

    async def append_event(self, event: StepEvent) -> None:
        if self._canonical_turn_boundary:
            await self.commit_step(events=(event,))
            return
        await self._store.append_event(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await self._store.list_events(run_id=run_id)

    def capture_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        """Capture mutable SDK history once; arbitrary historical edits remain visible.

        SDK hooks may mutate any nested message. This boundary is necessarily
        linear in the mutable history, unlike subsequent frozen storage operations.
        """
        snapshot = replace(snapshot, messages=self._message_archive.snapshot(snapshot.messages))
        if self._output_retries_used is not None:
            return NativeCheckpoint.from_snapshot(snapshot, output_retries_used=self._output_retries_used())
        return snapshot

    def _encode_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        native = isinstance(self._store, SqlStepStore)
        if native:
            source = self._message_archive.snapshot(snapshot.messages)
            previous = self._checkpoint_messages
            shared = 0 if previous is None else source.shared_prefix_length(previous[0])
            messages = self._message_archive.concatenate(
                () if previous is None else previous[1][:shared],
                native_checkpoint_messages(source[shared:]),
            )
            self._checkpoint_messages = source, messages
        else:
            # Caller-owned SDK stores serialize the published list contract directly.
            messages = native_checkpoint_messages(snapshot.messages)
        snapshot = replace(snapshot, messages=messages)
        if self._output_retries_used is None and not isinstance(snapshot, NativeCheckpoint):
            return snapshot
        retries = (snapshot.output_retries_used if isinstance(snapshot, NativeCheckpoint)
                   else self._output_retries_used())
        encode = NativeCheckpoint.from_snapshot if native else NativeCheckpoint.encode
        return encode(snapshot, output_retries_used=retries)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self._canonical_turn_boundary:
            await self.commit_step(snapshot=snapshot)
            return
        stored_snapshot = self._encode_snapshot(snapshot)
        try:
            await self._store.save_snapshot(stored_snapshot)
        except BaseException:
            if self._working_set is not None:
                self._working_set.invalidate_step_run(snapshot.run_id)
            raise
        if self._working_set is not None:
            self._working_set.confirm_step_snapshot(stored_snapshot)

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        snapshot = await self._store.latest_snapshot(
            run_id=run_id, include_interrupted=include_interrupted,
        )
        if snapshot is None:
            return None
        restored = NativeCheckpoint.decode(snapshot)
        return replace(restored, messages=native_checkpoint_messages(restored.messages, restore=True))

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
