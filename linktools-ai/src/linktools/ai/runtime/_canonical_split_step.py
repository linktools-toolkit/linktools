#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private source-first ownership choreography for a canonical split Step run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeVar

from ..core import canonical_json_bytes, canonical_sha256, deterministic_id
from ..errors import AIError, ErrorCode
from ..persistence.runs import PreparedArchive, StepRunAlreadyRegistered
from ..persistence.state import SqlStateTransaction
from ..persistence.turns import (
    CanonicalDeferredFrontier, SplitStepPin, SplitStepSourceObservation, SplitStepWaiting, SqlSplitStepSource,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.tools import DeferredToolResults
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot
    from ..agent import AgentBinding
    from ._agent_binding_resolver import _AgentBindingResolver
    from ._tool import RuntimeToolOperationBridge
    from ..core import JsonValue
    from ..persistence.contracts import (
        ApprovalRecord, CheckpointRepositories, ExternalCallRecord, ToolOperationRecord, ToolStateRepository,
    )
    from ..persistence._session_turn_sql import (
        SessionInstructionFrontier, SessionTurnKey, SqlSessionTurns, TerminalState, _SplitTargetObservation,
    )
    from ..persistence._sql import SqlStateStore
    from ..persistence.state import StateStore
    from ..persistence._step_store import PreparedStep, SqlStepStore, StepWriteReceipt


_Resolution = TypeVar("_Resolution")


@dataclass(frozen=True, slots=True)
class _SplitResume:
    checkpoint: ContinuableSnapshot
    messages: tuple[ModelMessage, ...]
    output_retries_used: int


@dataclass(frozen=True, slots=True)
class _SplitWaitingRecovery:
    frontier: CanonicalDeferredFrontier
    source: SplitStepSourceObservation
    operation: ToolOperationRecord
    resolution: ApprovalRecord | ExternalCallRecord


@dataclass(frozen=True, slots=True)
class _PreparedSplitWaitingRecovery:
    generation: int
    evidence: _SplitWaitingRecovery
    binding: AgentBinding
    resume: _SplitResume
    results: DeferredToolResults
    approval: tuple[CheckpointRepositories, ApprovalRecord, ToolOperationRecord] | None = None


class _CanonicalSplitStep:
    """Compose committed source and target receipts; never infer an unknown outcome."""

    def __init__(
        self, turns: SqlSessionTurns, source_store: SqlStateStore, step_store: SqlStepStore, *,
        read_effects: Callable[[str], Awaitable[Sequence[object]]] | None = None,
        stop_effects: tuple[ToolStateRepository, str] | None = None,
    ) -> None:
        self._turns = turns
        self._store = source_store
        self._step_store = step_store
        self._read_effects = read_effects
        self._stop_effects = stop_effects
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

    async def publish_waiting_source(
        self, key: SessionTurnKey, *, generation: int, pin: SplitStepPin,
        checkpoint: CheckpointRepositories, tenant_id: str, publish_receipts: Callable[[], Awaitable[None]],
        staged_operation: ToolOperationRecord | None = None,
    ) -> SplitStepSourceObservation:
        """Seal one split waiting leaf at its existing source; do not publish the target."""
        from ..core import ApprovalStatus, ExternalCallStatus, ToolOperationStatus
        from ..persistence.contracts import deferred_resource_id

        waiting = pin.waiting
        if waiting is None or waiting.original_generation != generation:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        approvals = waiting.kind == "approval"
        receipts = checkpoint.approvals if approvals else checkpoint.external_calls
        tools = checkpoint.tools
        group = self._store.storage_group
        members = tuple(dict.fromkeys((self._store, tools.state_store, receipts.state_store)))
        if any(member.storage_group is not group for member in members):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        identity = self._identity(key, generation)

        async def mutate(transaction):
            source = transaction.transaction(self._store)
            if not isinstance(source, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if not await source.run_sql(
                self._context, lambda db: self._source.seal_waiting(db, **identity, pin=pin),
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if staged_operation is not None:
                if not approvals or staged_operation.tool_operation_id != waiting.operation_id:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                (operation,) = await tools.reserve_many_in_transaction(
                    transaction.transaction(tools.state_store), (staged_operation,),
                )
            else:
                operation = await tools.get_operation(waiting.operation_id, tenant_id=tenant_id)
            if (
                operation is None or operation.execution_id != key.turn_id or operation.agent_run_id != key.turn_id
                or operation.producer_generation != generation or operation.status is not ToolOperationStatus.PENDING
                or operation.owner is not None or operation.arguments_payload is None
            ):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            receipt_id = deferred_resource_id(
                "approval-v1" if approvals else "external-call-v1",
                tenant_id, key.turn_id, key.turn_id, operation.tool_call_id,
            )
            if receipt_id != waiting.receipt_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            await publish_receipts()
            receipt = await receipts.get_in_transaction(
                transaction.transaction(receipts.state_store), receipt_id, tenant_id=tenant_id,
            )
            pending = ApprovalStatus.PENDING if approvals else ExternalCallStatus.PENDING
            if receipt is None or receipt.execution_id != key.turn_id or receipt.status is not pending:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        try:
            await group.mutate(members, mutate)
        except AIError as error:
            if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                raise
            observed = await self._read_pin(identity)
            if observed is None or observed.phase != "waiting" or observed.pin != pin:
                raise
            return observed
        return SplitStepSourceObservation(key.turn_id, key.session_id, generation, "waiting", pin)

    async def publish_waiting(
        self, frontier: CanonicalDeferredFrontier, *, waiting: SplitStepWaiting,
        checkpoint: CheckpointRepositories, tenant_id: str, max_bytes: int,
        publish_receipts: Callable[[], Awaitable[None]],
        staged_operation: ToolOperationRecord | None = None,
    ) -> bool:
        """Publish a live wait only after the exact source commits."""
        if (frontier.approval_operation_ids != ((waiting.operation_id,) if waiting.kind == "approval" else ())
                or frontier.external_operation_ids != ((waiting.operation_id,) if waiting.kind == "external" else ())):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if len(canonical_json_bytes(frontier.to_json())) > max_bytes:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
        pin = SplitStepPin(frontier.split_receipt_id, checkpoint_id=frontier.checkpoint_id, waiting=waiting)
        source = await self.publish_waiting_source(
            frontier.key, generation=frontier.generation, pin=pin, checkpoint=checkpoint, tenant_id=tenant_id,
            publish_receipts=publish_receipts, staged_operation=staged_operation,
        )
        return await self._publish_waiting_target(frontier, source=source, max_bytes=max_bytes, expired=False)

    async def recover_waiting(
        self, key: SessionTurnKey, *, generation: int, max_bytes: int, source: SplitStepSourceObservation,
    ) -> bool:
        """Reconcile an expired target from its observed source, without producing."""
        waiting = source.pin.waiting
        if (source.phase != "waiting" or waiting is None
                or (source.turn_id, source.session_id, source.generation, waiting.original_generation)
                != (key.turn_id, key.session_id, generation, generation)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        frontier = CanonicalDeferredFrontier.from_split_waiting(
            key, checkpoint_id=source.pin.checkpoint_id, waiting=waiting,
        )
        if source.pin.receipt_id != frontier.split_receipt_id:
            # No source evidence owns richer approval/replay metadata in this
            # tranche. Do not reconstruct it from a caller or current config.
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return await self._publish_waiting_target(frontier, source=source, max_bytes=max_bytes, expired=True)

    async def _publish_waiting_target(
        self, frontier: CanonicalDeferredFrontier, *, source: SplitStepSourceObservation,
        max_bytes: int, expired: bool,
    ) -> bool:
        try:
            if await self._turns.publish_split_waiting(frontier, source=source, max_bytes=max_bytes, _expired=expired):
                return True
        except AIError as error:
            if (error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN
                    or not await self._turns.confirms_split_waiting(frontier, receipt_id=source.pin.receipt_id)):
                raise
            return True
        return await self._turns.confirms_split_waiting(frontier, receipt_id=source.pin.receipt_id)

    async def settle_waiting_source(
        self, frontier: CanonicalDeferredFrontier, *, source: SplitStepSourceObservation,
        receipt_store: StateStore, tool_store: StateStore, settle: Callable[[], Awaitable[_Resolution]],
        is_ready: Callable[[], Awaitable[bool]], read_exact: Callable[[], Awaitable[_Resolution | None]],
    ) -> tuple[SplitStepSourceObservation, _Resolution]:
        """Settle the original leaf and mark readiness without resuming the producer.

        The authorized caller supplies the existing receipt command, proof of
        its settled effect requirements, and exact read-only idempotency check.
        All writes run in one source transaction; there is no target mutation.
        """
        key = frontier.key
        if (source.phase != "waiting" or source.pin.waiting is None
                or (source.turn_id, source.session_id, source.generation) != (
                    key.turn_id, key.session_id, frontier.generation,
                )
                or source.pin.receipt_id != frontier.split_receipt_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        group = self._store.storage_group
        members = tuple(dict.fromkeys((self._store, receipt_store, tool_store)))
        if any(store.storage_group is not group for store in members):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        ready = replace(source, phase="ready")

        async def mutate(transaction):
            store = transaction.transaction(self._store)
            if not isinstance(store, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if not await store.run_sql(self._context, lambda db: self._source.mark_waiting_ready(db, source)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = await settle()
            if not await is_ready():
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return result

        try:
            result = await group.mutate(members, mutate)
        except AIError as error:
            if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                raise
            if await self._read_pin(self._identity(key, source.generation)) != ready:
                raise
            observed = await read_exact()
            if observed is None:
                raise
            return ready, observed
        return ready, result

    async def settle_waiting(
        self, frontier: CanonicalDeferredFrontier, *, receipt_store: StateStore, tool_store: StateStore,
        settle: Callable[[], Awaitable[_Resolution]], is_ready: Callable[[], Awaitable[bool]],
        read_exact: Callable[[], Awaitable[_Resolution | None]],
    ) -> _Resolution:
        """Confirm source settlement before publishing the same original target READY."""
        source = await self._read_pin(self._identity(frontier.key, frontier.generation))
        if (source is None or source.pin.waiting is None or source.phase not in ("waiting", "ready")
                or source.pin.receipt_id != frontier.split_receipt_id):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if source.phase == "ready":
            # A previous attempt can have committed at the source but not the
            # target. Only its exact authorized command can finish the handoff.
            result = await read_exact()
            if result is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            ready = source
        else:
            ready, result = await self.settle_waiting_source(
                frontier, source=source, receipt_store=receipt_store, tool_store=tool_store,
                settle=settle, is_ready=is_ready, read_exact=read_exact,
            )
        if not await self._turns.publish_split_ready(frontier, source=ready):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return result

    async def regrant_waiting(
        self, frontier: CanonicalDeferredFrontier, *, lease_seconds: int,
        verify_resolved: Callable[[], Awaitable[bool]],
    ) -> tuple[SplitStepSourceObservation | None, SessionInstructionFrontier | None]:
        """Grant the original resolved split continuation, without running it."""
        key, generation = frontier.key, frontier.generation
        source = await self._read_pin(self._identity(key, generation))
        if (source is None or source.phase != "ready" or source.pin.waiting is None
                or source.pin.receipt_id != frontier.split_receipt_id
                or not await verify_resolved()):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        async def confirm():
            return await self._turns.confirmed_split_regrant(
                key, generation=generation + 1, receipt_id=source.pin.receipt_id,
            )

        try:
            granted, instructions = await self._turns.take_over_split_ready(
                frontier, lease_seconds=lease_seconds, source=source,
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
                return None, None
        # The target is now proven. Only one adopter may proceed; a lost
        # source ACK never grants producer ownership to this caller.
        adopted = await self._write(lambda db: self._source.adopt_ready(db, source))
        return (replace(source, generation=generation + 1, phase="resuming"), instructions) if adopted else (None, None)

    async def _claim_resuming_approved_leaf(
        self, transaction, source: SplitStepSourceObservation, *, checkpoint: CheckpointRepositories,
        tenant_id: str, approval: ApprovalRecord, pending: ToolOperationRecord, request,
    ):
        """Claim the original approved leaf inside the caller's enlisted source write."""
        from ..core import ApprovalDecision, ApprovalStatus, ToolOperationStatus
        from ..persistence.contracts import deferred_resource_id

        waiting = source.pin.waiting
        if (source.phase != "resuming" or waiting is None or waiting.kind != "approval"
                or (pending.tool_operation_id, pending.execution_id, pending.agent_run_id)
                != (waiting.operation_id, source.turn_id, source.turn_id)
                or pending.producer_generation != waiting.original_generation
                or waiting.original_generation >= source.generation
                or pending.status is not ToolOperationStatus.PENDING or pending.owner is not None
                or (request.tool_operation_id, request.execution_id, request.agent_run_id,
                    request.producer_generation) != (waiting.operation_id, source.turn_id, source.turn_id, source.generation)
                or request.recovery_agent_run_id is not None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        identity = deferred_resource_id(
            "approval-v1", tenant_id, source.turn_id, source.turn_id, pending.tool_call_id,
        )
        if identity != waiting.receipt_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        approvals, tools = checkpoint.approvals, checkpoint.tools
        if (approvals.state_store.storage_group is not self._store.storage_group
                or tools.state_store.storage_group is not self._store.storage_group):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        source_transaction = transaction.transaction(self._store)
        if not isinstance(source_transaction, SqlStateTransaction):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        await source_transaction.run_sql(
            self._context, lambda db: self._source.guard_resuming_approval(db, source),
        )
        saved = await approvals.get_in_transaction(
            transaction.transaction(approvals.state_store), identity, tenant_id=tenant_id,
        )
        if (saved is None or saved != approval or saved.execution_id != source.turn_id
                or saved.status is not ApprovalStatus.APPROVED or saved.decision is not ApprovalDecision.APPROVE
                or saved.decided_at is None or saved.idempotency_key is None):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return await tools.admit_in_transaction(
            transaction.transaction(tools.state_store), request, expected_pending=pending,
        )

    async def commit_resumed_checkpoint(
        self, prepared: PreparedStep, *, source: SplitStepSourceObservation,
    ) -> StepWriteReceipt:
        """Consume the original wait pin only with the first resumed checkpoint."""
        snapshot = prepared.snapshot
        if (source.phase != "resuming" or source.pin.waiting is None or prepared.run is not None
                or snapshot is None or snapshot.state != "complete"
                or (snapshot.run_id, snapshot.conversation_id) != (source.turn_id, source.session_id)
                or snapshot.idempotency_key is None or snapshot.idempotency_key == source.pin.checkpoint_id
                or any(event.run_id != source.turn_id or event.conversation_id != source.session_id
                       for event in prepared.events)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction):
            if not isinstance(transaction, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if not await transaction.run_sql(
                self._context, lambda db: self._source.consume_resumed_checkpoint(db, source),
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await self._step_store.write_prepared_step(transaction, prepared)

        receipt = await self._store.mutate(mutate)
        self._step_store.confirm_prepared_step(receipt)
        return receipt

    async def read_waiting_recovery(
        self, key: SessionTurnKey, *, generation: int, checkpoint: CheckpointRepositories, tenant_id: str,
        ready: bool = False,
    ) -> _SplitWaitingRecovery:
        """Capture only the original settled wait; the caller still needs authority to resume."""
        from ..core import ApprovalDecision, ApprovalStatus, ExternalCallStatus, ToolOperationStatus
        from ..persistence.contracts import ApprovalRecord, ExternalCallRecord, deferred_resource_id

        if generation < (1 if ready else 2):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        source = await self._read_pin(self._identity(key, generation))
        if ready:
            if source is None or source.phase != "ready":
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        elif source is None:
            # The target grant can commit before the source acknowledges it.
            source = await self._read_pin(self._identity(key, generation - 1))
            if source is None or source.phase not in ("ready", "resume_hold"):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        elif source.phase not in ("resuming", "resume_hold"):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        waiting = source.pin.waiting
        if waiting is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        tools = checkpoint.tools
        receipts = checkpoint.approvals if waiting.kind == "approval" else checkpoint.external_calls
        group = self._store.storage_group
        members = tuple(dict.fromkeys((tools.state_store, receipts.state_store)))
        if any(member.storage_group is not group for member in members):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def read(transaction):
            operation = await tools.get_operation(waiting.operation_id, tenant_id=tenant_id)
            if (operation is None or operation.execution_id != key.turn_id or operation.agent_run_id != key.turn_id
                    or operation.producer_generation is None
                    or not waiting.original_generation <= operation.producer_generation <= source.generation
                    or (source.phase == "ready" and waiting.original_generation != source.generation)
                    or operation.arguments_payload is None):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            expected = deferred_resource_id(
                "approval-v1" if waiting.kind == "approval" else "external-call-v1",
                tenant_id, key.turn_id, key.turn_id, operation.tool_call_id,
            )
            if expected != waiting.receipt_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            resolution = await receipts.get_in_transaction(
                transaction.transaction(receipts.state_store), expected, tenant_id=tenant_id,
            )
            if resolution is None or resolution.execution_id != key.turn_id or resolution.idempotency_key is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if isinstance(resolution, ApprovalRecord) and waiting.kind == "approval":
                if resolution.status is ApprovalStatus.APPROVED and resolution.decision is ApprovalDecision.APPROVE:
                    if operation.producer_generation == waiting.original_generation:
                        status = ToolOperationStatus.PENDING
                    elif operation.owner is not None and operation.status in (
                        ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN,
                        ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED,
                    ):
                        return operation, resolution
                    else:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                elif resolution.status is ApprovalStatus.DENIED and resolution.decision is ApprovalDecision.DENY:
                    status = ToolOperationStatus.FAILED
                else:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if resolution.decided_at is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            elif isinstance(resolution, ExternalCallRecord) and waiting.kind == "external":
                if (resolution.status is not ExternalCallStatus.SUPPLIED or resolution.supplied_at is None
                        or resolution.result_payload is None or resolution.resolution_kind is None):
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                status = (ToolOperationStatus.COMPLETED if resolution.resolution_kind == "succeeded"
                          else ToolOperationStatus.FAILED)
            else:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if (operation.producer_generation != waiting.original_generation
                    or operation.status is not status or operation.owner is not None):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return operation, resolution

        operation, resolution = await group.read_many(members, read)
        frontier = (await self._turns.read_split_ready_frontier(key, source=source) if ready
                    else await self._turns.read_split_resuming_frontier(
                        key, generation=generation, original_generation=waiting.original_generation,
                        receipt_id=source.pin.receipt_id,
                    ))
        if (frontier is None or frontier.checkpoint_id != source.pin.checkpoint_id
                or source.pin.receipt_id != frontier.split_receipt_id
                or any(frontier.operation_metadata.values())
                or frontier.approval_operation_ids != ((waiting.operation_id,) if waiting.kind == "approval" else ())
                or frontier.external_operation_ids != ((waiting.operation_id,) if waiting.kind == "external" else ())):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if operation.status in (ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN):
            # The old worker may have invoked the effect. A regrant is not
            # proof of its outcome, even when the tool advertises replay safety.
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        return _SplitWaitingRecovery(frontier, source, operation, resolution)

    async def prepare_waiting_recovery(
        self, key: SessionTurnKey, *, generation: int, checkpoint: CheckpointRepositories,
        tenant_id: str, binding_payload: JsonValue, resolver: _AgentBindingResolver,
        read_payload: Callable[..., Awaitable[object]],
        tools: RuntimeToolOperationBridge | None = None, ready: bool = False,
    ) -> _PreparedSplitWaitingRecovery:
        """Read only; caller must supply the already-authorized target's frozen binding."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
        from pydantic_ai.tools import DeferredToolResults, ToolApproved, ToolDenied

        from ..agent import AgentBindingContract
        from ..capability import PydanticToolControlCapability
        from ..core import ApprovalStatus, ToolOperationStatus
        from ..model.messages import native_checkpoint_messages
        from ..persistence.contracts import ApprovalRecord
        from ..persistence.runs import NativeCheckpoint

        evidence = await self.read_waiting_recovery(
            key, generation=generation, checkpoint=checkpoint, tenant_id=tenant_id, ready=ready,
        )
        operation, resolution = evidence.operation, evidence.resolution
        call_id = operation.tool_call_id
        if binding_payload is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        binding = await resolver._restore_split_wait_binding(AgentBindingContract.from_payload(binding_payload), operation)
        approved = isinstance(resolution, ApprovalRecord) and resolution.status is ApprovalStatus.APPROVED
        pending_approval = approved and operation.status is ToolOperationStatus.PENDING
        if approved and not pending_approval and (tools is None or tools._repository is not checkpoint.tools
                or operation.status not in (ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        occurrence = evidence.frontier.checkpoint_id
        async with self._context.sessions() as session:
            hydrate = await self._step_store.capture_turn_checkpoint(
                session, run_id=key.turn_id, conversation_id=key.session_id, replay_key=occurrence,
            )
        if hydrate is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        saved = NativeCheckpoint.decode(await hydrate())
        if ((saved.run_id, saved.conversation_id, saved.idempotency_key)
                != (key.turn_id, key.session_id, occurrence) or saved.state != "interrupted"):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = tuple(native_checkpoint_messages(saved.messages, restore=True))
        last_response = next((i for i in range(len(messages) - 1, -1, -1)
                              if isinstance(messages[i], ModelResponse)), -1)
        if last_response < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        pending = {}
        for part in messages[last_response].parts:
            if isinstance(part, ToolCallPart):
                if part.tool_call_id in pending:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                pending[part.tool_call_id] = part.tool_name
        for message in messages[last_response + 1:]:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                        pending.pop(part.tool_call_id, None)
        if pending != {call_id: operation.tool_name}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        results = DeferredToolResults()
        approval = None
        if pending_approval:
            arguments = await read_payload(operation.arguments_payload)
            if not isinstance(arguments, dict) or canonical_sha256(arguments) != operation.arguments_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            results.approvals[call_id] = ToolApproved(override_args=arguments)
            approval = checkpoint, resolution, operation
        elif approved:
            results.calls[call_id] = await tools._canonical_effect_result(operation)
        elif isinstance(resolution, ApprovalRecord):
            results.approvals[call_id] = ToolDenied(
                message=resolution.decision_message or "The tool call was denied.",
            )
        else:
            result = await read_payload(resolution.result_payload)
            if resolution.resolution_kind == "succeeded":
                results.calls[call_id] = result
            elif resolution.resolution_kind == "retry" and isinstance(result, str):
                results.calls[call_id] = PydanticToolControlCapability.build_model_retry(result)
            elif resolution.resolution_kind == "failed" and isinstance(result, str):
                results.calls[call_id] = PydanticToolControlCapability.build_tool_failed(result)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        results.metadata[call_id] = dict(resolution.resolution_metadata)
        retries = saved.output_retries_used if isinstance(saved, NativeCheckpoint) else 0
        return _PreparedSplitWaitingRecovery(
            generation, evidence, binding, _SplitResume(saved, messages, retries), results, approval,
        )

    async def reacquire_waiting_recovery(
        self, prepared: _PreparedSplitWaitingRecovery, *, lease_seconds: int,
    ) -> tuple[_PreparedSplitWaitingRecovery | None, SessionInstructionFrontier | None]:
        """Freeze only a prepared owner, then prove and adopt the next grant."""
        evidence, generation = prepared.evidence, prepared.generation
        key = evidence.frontier.key
        if evidence.source.phase == "ready" and evidence.source.generation == generation:
            async def verify_resolved() -> bool:
                return await self._read_pin(self._identity(key, generation)) == evidence.source

            source, instructions = await self.regrant_waiting(
                evidence.frontier, lease_seconds=lease_seconds, verify_resolved=verify_resolved,
            )
            if source is None:
                return None, None
            return replace(prepared, evidence=replace(evidence, source=source)), instructions
        hold = replace(evidence.source, generation=generation, phase="resume_hold")
        try:
            frozen = await self._write(lambda db: self._source.freeze_resuming(
                db, evidence.source, expired_generation=generation,
            ))
        except AIError as error:
            if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                raise
            if await self._read_pin(self._identity(key, generation)) != hold:
                raise
            frozen = True
        if not frozen and await self._read_pin(self._identity(key, generation)) != hold:
            return None, None

        async def confirm():
            return await self._turns.confirmed_split_regrant(
                key, generation=generation + 1, receipt_id=hold.pin.receipt_id,
            )

        try:
            granted, instructions = await self._turns.take_over_split_resuming(
                evidence.frontier, source=hold, lease_seconds=lease_seconds,
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
                return None, None
        if not await self._write(lambda db: self._source.adopt_frozen_resume(db, hold)):
            return None, None
        adopted = replace(evidence, source=replace(hold, generation=generation + 1, phase="resuming"))
        return replace(prepared, evidence=adopted), instructions

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

    async def _stop_waiting_source(
        self, key: SessionTurnKey, *, target: _SplitTargetObservation, source: SplitStepSourceObservation,
        result: JsonValue,
    ) -> SplitStepSourceObservation:
        waiting = source.pin.waiting
        if waiting is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        unacknowledged = (
            target.receipt_id is None and source.phase == "waiting"
            and waiting.original_generation == source.generation == target.generation
        )
        if (self._stop_effects is None or target.state != "cancelling"
                or (not unacknowledged and target.receipt_id != source.pin.receipt_id)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        tools, tenant_id = self._stop_effects
        group = self._store.storage_group
        if tools.state_store.storage_group is not group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        async def write(transaction):
            source_tx = transaction.transaction(self._store)
            if not isinstance(source_tx, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)

            async def lock(db):
                return await self._source._move_waiting_pin(
                    db, source, generation=source.generation, phase=source.phase,
                )

            if not await source_tx.run_sql(self._context, lock):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if await source_tx.run_sql(
                self._context, lambda db: self._source.observe_pin(db, **self._identity(key, source.generation)),
            ) != source:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            from ..core import ToolOperationStatus

            operation = await tools.get_operation_in_transaction(
                transaction.transaction(tools.state_store), waiting.operation_id, tenant_id=tenant_id,
            )
            if (operation is None or operation.execution_id != key.turn_id or operation.agent_run_id != key.turn_id
                    or operation.producer_generation is None
                    or not waiting.original_generation <= operation.producer_generation <= source.generation):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if unacknowledged and (
                operation.producer_generation != waiting.original_generation
                or operation.status is not ToolOperationStatus.PENDING or operation.owner is not None
            ):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            outcome = ({"error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value}
                       if operation.status in (ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN)
                       else result)
            return await source_tx.run_sql(self._context, lambda db: self._source.settle_waiting_cancel(
                db, source, generation=target.generation, result=outcome,
            ))

        try:
            sealed = await group.mutate(tuple(dict.fromkeys((self._store, tools.state_store))), write)
        except AIError as error:
            if error.code not in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN):
                raise
            sealed = await self._read_pin(self._identity(key, target.generation))
            if sealed is None or sealed.pin.terminal_state != "cancelled":
                raise
        if (sealed is None or sealed.pin.receipt_id != source.pin.receipt_id
                or sealed.pin.checkpoint_id != source.pin.checkpoint_id):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return sealed

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
                if (cancelling and predecessor is not None and predecessor.pin.waiting is not None
                        and predecessor.phase in ("ready", "resume_hold") and predecessor.pin.receipt_id == target.receipt_id):
                    source = predecessor
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

            if source is None:
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

        if source.pin.waiting is not None:
            if not cancelling:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            source = await self._stop_waiting_source(key, target=target, source=source, result=cancel_result)
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
        from sqlalchemy.exc import IntegrityError

        from ..storage import IntegrityViolationKind

        if (target.state != "cancelling" or target.generation != 1
                or target.receipt_id is not None or target.admitted_at is None):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        pin = SplitStepPin(
            deterministic_id("canonical-split-unregistered-stop", key.turn_id, target.generation),
            terminal_state="cancelled", result=result,
        )
        duplicate = None
        try:
            source = await self._write(lambda db: self._source.cancel_unregistered(
                db, turn_id=key.turn_id, session_id=key.session_id,
                admitted_at=target.admitted_at.isoformat(), pin=pin,
            ))
        except (AIError, IntegrityError) as error:
            # The group boundary may translate a failed INSERT. Both paths
            # return only after rollback; don't interpret a bare conflict as
            # proof of this unique registration race.
            sql_error = (
                error.__cause__ if isinstance(error, AIError) and error.code is ErrorCode.STORAGE_CONFLICT else error
            )
            if (isinstance(sql_error, IntegrityError)
                    and self._context.dialect.classify_integrity_error(sql_error) is IntegrityViolationKind.UNIQUE_CONFLICT):
                duplicate = error
            elif not isinstance(error, AIError) or error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            source = None
        if source is None:
            observed = await self._read_pin(self._identity(key, target.generation))
            if observed is None and duplicate is not None:
                async with self._context.sessions() as session:
                    exact = await self._source.has_registration(
                        session, turn_id=key.turn_id, session_id=key.session_id,
                    )
                if not exact:
                    raise duplicate
            if observed is None or observed.pin != pin:
                # An original registration can win the unique insert and may
                # have reached the provider. Missing proof cannot settle it.
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            source = observed
        return source

    async def _regrant(
        self, key: SessionTurnKey, *, source: SplitStepSourceObservation, lease_seconds: int,
    ) -> tuple[bool, SessionInstructionFrontier | None]:
        if source.phase != "sealed" or source.pin.waiting is not None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
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
