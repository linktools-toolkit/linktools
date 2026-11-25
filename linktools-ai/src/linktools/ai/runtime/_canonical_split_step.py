#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private source-first ownership choreography for a canonical split Step run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import deterministic_id
from ..errors import AIError, ErrorCode
from ..persistence.runs import PreparedArchive, StepRunAlreadyRegistered
from ..persistence.state import SqlStateTransaction
from ..persistence.turns import SplitStepPin, SplitStepSourceObservation, SqlSplitStepSource

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot
    from ..core import JsonValue
    from ..persistence._session_turn_sql import (
        SessionInstructionFrontier, SessionTurnKey, SqlSessionTurns, TerminalState, _SplitTargetObservation,
    )
    from ..persistence._sql import SqlStateStore
    from ..persistence._step_store import PreparedStep, SqlStepStore, StepWriteReceipt


@dataclass(frozen=True, slots=True)
class _SplitResume:
    checkpoint: ContinuableSnapshot
    messages: tuple[ModelMessage, ...]
    output_retries_used: int


class _CanonicalSplitStep:
    """Compose committed source and target receipts; never infer an unknown outcome."""

    def __init__(
        self, turns: SqlSessionTurns, source_store: SqlStateStore, step_store: SqlStepStore, *,
        read_effects: Callable[[str], Awaitable[Sequence[object]]] | None = None,
    ) -> None:
        self._turns = turns
        self._store = source_store
        self._step_store = step_store
        self._read_effects = read_effects
        self._context = step_store.storage_context
        group = source_store.storage_group
        if source_store.context.engine is not self._context.engine:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        self._source = SqlSplitStepSource(
            group.metadata.tables["ai_step_runs"], group.metadata.tables["ai_step_snapshots"],
            store_key=step_store.store_key,
        )

    async def _write(self, callback):
        async def mutate(transaction):
            if not isinstance(transaction, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            return await transaction.run_sql(self._context, callback)

        return await self._store.mutate(mutate)

    @staticmethod
    def _identity(key: SessionTurnKey, generation: int) -> dict:
        return {"turn_id": key.turn_id, "session_id": key.session_id, "generation": generation}

    async def commit_step(
        self, prepared: PreparedStep, *, key: SessionTurnKey, generation: int,
    ) -> StepWriteReceipt:
        """Adopt an already accepted first generation with its source registration."""
        run, events, snapshot = prepared.run, prepared.events, prepared.snapshot
        records = (*events, *((snapshot,) if snapshot is not None else ()))
        if any(record.run_id != key.turn_id for record in records):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if run is not None and (
            generation != 1 or (run.run_id, run.conversation_id) != (key.turn_id, key.session_id)
            or not events or events[0].kind != "run_started"
            or any(event.kind != "model_request_started" for event in events[1:])
            or any(event.conversation_id != key.session_id for event in events)
            or (snapshot is not None and (
                snapshot.conversation_id != key.session_id or snapshot.state != "complete"
            ))
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        identity = self._identity(key, generation)

        async def mutate(transaction):
            if not isinstance(transaction, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if run is None:
                await self.guard(transaction, key, generation=generation)
            try:
                receipt = await self._step_store.write_prepared_step(transaction, prepared)
            except StepRunAlreadyRegistered as error:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            if run is not None and not await transaction.run_sql(
                self._context, lambda db: self._source.adopt(db, **identity),
            ):
                # Roll back the registration if the exact source cannot be adopted.
                # A retry after an unknown commit is not a fresh admission.
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return receipt

        receipt = await self._store.mutate(mutate)
        self._step_store.confirm_prepared_step(receipt)
        return receipt

    async def guard(
        self, transaction: SqlStateTransaction, key: SessionTurnKey, *, generation: int,
    ) -> None:
        """Use in the same transaction that prepares the source effect receipt."""
        await transaction.run_sql(
            self._context, lambda db: self._source.guard(db, **self._identity(key, generation)),
        )

    async def _read_pin(self, identity: dict) -> SplitStepSourceObservation | None:
        async with self._context.sessions() as session:
            return await self._source.observe_pin(session, **identity)

    async def seal(
        self, key: SessionTurnKey, *, generation: int, pin: SplitStepPin,
        finish_source: Callable[[SqlStateTransaction], Awaitable[None]],
    ) -> SplitStepSourceObservation:
        """Atomically write source-owned body/evidence and close its write fence."""
        identity = self._identity(key, generation)

        async def mutate(transaction):
            if not isinstance(transaction, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            await self.guard(transaction, key, generation=generation)
            if not await transaction.run_sql(
                self._context, lambda db: self._source.seal(db, **identity, pin=pin),
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await finish_source(transaction)

        try:
            await self._store.mutate(mutate)
        except AIError as error:
            if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                raise
            observed = await self._read_pin(identity)
            if observed is None or observed.pin != pin:
                raise
            return observed
        return SplitStepSourceObservation(key.turn_id, key.session_id, generation, "sealed", pin)

    async def finish(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, state: TerminalState,
        result: JsonValue = None, transcript: Sequence[ModelMessage] | None = None,
        binding_refs: tuple[tuple[int, int | None], ...] = (),
        output_ref: tuple[int, int] | None = None,
    ) -> bool:
        prepared = PreparedArchive()
        root = (None if transcript is None
                else await self._step_store.prepare_archived_messages(transcript, prepared))
        pin = SplitStepPin(
            deterministic_id("canonical-split-terminal", key.session_id, key.turn_id, generation, state),
            terminal_state=state, transcript=root, result=result, binding_refs=binding_refs, output_ref=output_ref,
        )

        async def write_source(transaction: SqlStateTransaction) -> None:
            if root is not None:
                # seal() has already pinned this exact root by source CAS
                # inside this same transaction; rollback drops both.
                await self._step_store.publish_archived_messages(transaction, prepared)

        try:
            source = await self.seal(key, generation=generation, pin=pin, finish_source=write_source)
        except AIError as error:
            if state != "cancelled" or error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            source = await self._read_pin(self._identity(key, generation))
            if source is None:
                target = await self._turns.split_target_lease(
                    key, generation=generation, cancelling=True, expired=False,
                )
                if target is None:
                    return False
                source = await self._claim_unregistered_stop(key, target=target, result=result)
            elif source.pin.terminal_state in ("succeeded", "failed"):
                return await self.discard_terminal(key, turn_no=turn_no, source=source, result=result)
            elif source.pin.terminal_state != "cancelled" or source.pin.result != result:
                raise
        return await self.publish_terminal(key, turn_no=turn_no, state=state, source=source)

    async def publish_terminal(
        self, key: SessionTurnKey, *, turn_no: int, state: TerminalState,
        source: SplitStepSourceObservation, _expired: bool = False,
    ) -> bool:
        """Publish no more than one immutable source receipt into its target turn."""
        if (source.turn_id, source.session_id, source.pin.terminal_state) != (key.turn_id, key.session_id, state):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        receipt = source.pin.receipt_id
        scope = {"generation": source.generation, "state": state, "receipt_id": receipt}
        if source.phase != "sealed":
            committed = False
        else:
            try:
                committed = await self._turns.publish_split_terminal(
                    key, generation=source.generation, turn_no=turn_no, state=state, source=source,
                    result=source.pin.result, binding_refs=source.pin.binding_refs, output_ref=source.pin.output_ref,
                    _expired=_expired,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                observed = await self._turns.observe_split_target(key)
                if observed is None or not observed.confirms(**scope):
                    if await self._reconcile_superseding_stop(source, observed):
                        return False
                    raise
                committed = True
        if not committed:
            observed = await self._turns.observe_split_target(key)
            if observed is None or not observed.confirms(**scope):
                await self._reconcile_superseding_stop(source, observed)
                return False
        if source.phase == "sealed":
            await self._write(lambda db: self._source.reconcile(
                db, **self._identity(key, source.generation), exact_target_receipt=receipt,
            ))
        return True

    async def _reconcile_superseding_stop(self, source: SplitStepSourceObservation, target) -> bool:
        if (source.pin.terminal_state not in ("succeeded", "failed") or target is None
                or not target.confirms(generation=source.generation, state="cancelled",
                                       receipt_id=source.pin.receipt_id)):
            return False
        if source.phase != "discarded":
            await self._write(lambda db: self._source.reconcile_discarded(db, source))
        # The requested success/failure did not publish even though the exact
        # target outcome is known. Do not emit the losing outcome to clients.
        return True

    async def discard_terminal(
        self, key: SessionTurnKey, *, turn_no: int, source: SplitStepSourceObservation,
        result: JsonValue = None, _expired: bool = False,
    ) -> bool:
        """Reconcile Stop against the same sealed terminal identity, never its body."""
        if ((source.turn_id, source.session_id) != (key.turn_id, key.session_id)
                or source.pin.terminal_state not in ("succeeded", "failed")):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        receipt = source.pin.receipt_id
        proof = {"generation": source.generation, "state": "cancelled", "receipt_id": receipt}

        async def confirmed() -> bool:
            observed = await self._turns.observe_split_target(key)
            return observed is not None and observed.confirms(**proof)

        if source.phase == "sealed":
            try:
                committed = await self._turns.publish_split_discard(
                    key, turn_no=turn_no, source=source, result=result, _expired=_expired,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or not await confirmed():
                    raise
                committed = True
            if not committed and not await confirmed():
                return False
        elif not await confirmed():
            return False
        if source.phase != "discarded":
            await self._write(lambda db: self._source.reconcile_discarded(db, source))
        return True

    async def recover_terminal(
        self, key: SessionTurnKey, *, turn_no: int, source: SplitStepSourceObservation,
        cancel_result: JsonValue = None,
    ) -> bool:
        """Finish an expired exact generation without authorizing a new producer."""
        from typing import cast

        terminal = source.pin.terminal_state
        if terminal not in ("succeeded", "failed", "cancelled"):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if await self.publish_terminal(
            key, turn_no=turn_no, state=cast("TerminalState", terminal), source=source, _expired=True,
        ):
            return True
        return terminal in ("succeeded", "failed") and await self.discard_terminal(
            key, turn_no=turn_no, source=source, _expired=True,
            result={"error_code": ErrorCode.EXECUTION_CANCELLED.value} if cancel_result is None else cancel_result,
        )

    async def require_no_effects(self, source: SplitStepSourceObservation) -> None:
        """Only the bounded, configured no-tool topology supports cold takeover."""
        if self._read_effects is None or await self._read_effects(source.turn_id):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    async def recover_open_turn(
        self, key: SessionTurnKey, *, generation: int, turn_no: int, lease_seconds: int, cancelling: bool,
        require_effects_resolved: Callable[[SplitStepSourceObservation], Awaitable[None]] | None = None,
        cancel_result: JsonValue = None,
    ) -> tuple[str, _SplitResume | None, SessionInstructionFrontier | None]:
        """Freeze the exact old source before issuing a replacement generation."""
        identity = self._identity(key, generation)
        read_effects = self.require_no_effects if require_effects_resolved is None else require_effects_resolved
        target = await self._turns.split_target_lease(
            key, generation=generation, cancelling=cancelling, expired=True,
        )
        if target is None:
            return "not_acquired", None, None
        source = await self._read_pin(identity)
        if source is None:
            if generation > 1 and target.receipt_id is not None:
                predecessor = await self._read_pin(self._identity(key, generation - 1))
                if (predecessor is not None and predecessor.phase == "sealed"
                        and predecessor.pin.receipt_id == target.receipt_id
                        and predecessor.pin.terminal_state is None
                        and predecessor.pin.checkpoint_id is not None):
                    await read_effects(predecessor)
                    # The persisted target proves this generation was already
                    # granted. Repair only its subordinate source; do not
                    # produce anything under the already expired target lease.
                    await self._write(lambda db: self._source.adopt(db, **identity, predecessor=predecessor.pin))

            async def freeze(session):
                await self._source.guard(session, **identity)
                checkpoint = (None if cancelling else
                              await self._source.latest_complete_checkpoint_key(
                                  session, turn_id=key.turn_id, session_id=key.session_id,
                              ))
                pin = SplitStepPin(
                    deterministic_id("canonical-split-recovery", key.turn_id, generation),
                    checkpoint_id=checkpoint, terminal_state="cancelled" if cancelling else None,
                    result=cancel_result if cancelling else None,
                )
                if not await self._source.seal(session, **identity, pin=pin):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return SplitStepSourceObservation(key.turn_id, key.session_id, generation, "sealed", pin)

            try:
                source = await self._write(freeze)
            except AIError as error:
                if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                    raise
                source = await self._read_pin(identity)
                if source is None:
                    if error.code is not ErrorCode.STORAGE_CONFLICT or not cancelling:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
                    source = await self._claim_unregistered_stop(key, target=target, result=cancel_result)

        if source.pin.terminal_state is not None:
            if not await self.recover_terminal(
                key, turn_no=turn_no, source=source, cancel_result=cancel_result,
            ):
                return "not_acquired", None, None
            return "settled", None, None
        if cancelling:
            if target.state != "cancelling":
                return "not_acquired", None, None
            try:
                cancelled = await self._write(lambda db: self._source.settle_sealed_cancel(
                    db, source, result=cancel_result,
                ))
            except AIError as error:
                if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                    raise
                cancelled = await self._read_pin(identity)
            if (cancelled is None or cancelled.pin.receipt_id != source.pin.receipt_id
                    or cancelled.pin.terminal_state != "cancelled"
                    or cancelled.pin.result != cancel_result):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if await self.recover_terminal(key, turn_no=turn_no, source=cancelled, cancel_result=cancel_result):
                return "settled", None, None
            return "not_acquired", None, None
        if source.phase != "sealed" or source.pin.checkpoint_id is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await read_effects(source)
        async with self._context.sessions() as session:
            checkpoint = await self._step_store.capture_turn_checkpoint(
                session, run_id=key.turn_id, conversation_id=key.session_id,
                replay_key=source.pin.checkpoint_id,
            )
        if checkpoint is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        snapshot = await checkpoint()
        if snapshot.state != "complete" or snapshot.idempotency_key != source.pin.checkpoint_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        validated = await self._validate_resume(key, snapshot)
        granted, instructions = await self._regrant(key, source=source, lease_seconds=lease_seconds)
        return ("resumed", validated, instructions) if granted else ("not_acquired", None, None)

    async def _validate_resume(self, key: SessionTurnKey, snapshot: ContinuableSnapshot) -> _SplitResume:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from ..model.messages import native_checkpoint_messages
        from ..persistence.runs import NativeCheckpoint

        checkpoint = NativeCheckpoint.decode(snapshot)
        if checkpoint.run_id != key.turn_id or checkpoint.conversation_id != key.session_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = tuple(native_checkpoint_messages(checkpoint.messages, restore=True))
        last = messages[-1] if messages else None
        if (not isinstance(last, ModelResponse) or last.text is None
                or any(isinstance(part, ToolCallPart) for part in last.parts)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        events = await self._step_store.list_events(run_id=key.turn_id)
        terminal = next((event for event in reversed(events)
                         if event.kind in ("run_completed", "run_failed", "run_cancelled")), None)
        if terminal is None or terminal.kind != "run_completed":
            # A checkpoint preceding a provider result is never replayed.
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        retries = checkpoint.output_retries_used if isinstance(checkpoint, NativeCheckpoint) else 0
        return _SplitResume(checkpoint, messages, retries)

    async def _claim_unregistered_stop(
        self, key: SessionTurnKey, *, target: _SplitTargetObservation, result: JsonValue,
    ) -> SplitStepSourceObservation:
        if (target.state != "cancelling" or target.generation != 1
                or target.receipt_id is not None or target.admitted_at is None):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        pin = SplitStepPin(
            deterministic_id("canonical-split-unregistered-stop", key.turn_id, target.generation),
            terminal_state="cancelled", result=result,
        )
        try:
            source = await self._write(lambda db: self._source.cancel_unregistered(
                db, self._context, turn_id=key.turn_id, session_id=key.session_id,
                admitted_at=target.admitted_at.isoformat(), pin=pin,
            ))
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            source = None
        if source is None:
            observed = await self._read_pin(self._identity(key, target.generation))
            if observed is None or observed.pin != pin:
                # An original registration can win the unique insert and may
                # have reached the provider. Missing proof cannot settle it.
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            source = observed
        return source

    async def _regrant(
        self, key: SessionTurnKey, *, source: SplitStepSourceObservation, lease_seconds: int,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        generation = source.generation + 1

        async def confirm():
            return await self._turns.confirmed_split_regrant(
                key, generation=generation, receipt_id=source.pin.receipt_id,
            )

        try:
            granted, instructions = await self._turns.take_over_split_generation(
                key, generation=source.generation, lease_seconds=lease_seconds, source=source,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            granted, instructions = await confirm()
            if not granted:
                raise
        if not granted:
            granted, instructions = await confirm()
            if not granted:
                return False, None
        adopted = await self._write(lambda db: self._source.adopt(
            db, **self._identity(key, generation), predecessor=source.pin,
        ))
        return (True, instructions) if adopted else (False, None)
