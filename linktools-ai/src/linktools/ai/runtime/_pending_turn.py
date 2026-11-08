#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ownership of executions that have not reached a durable boundary."""

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from ..core import ExecutionStatus, IdempotencyStatus, SessionStatus, ToolOperationStatus
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload, TurnCoordination
from ..persistence.coordination import PendingTurnCoordination
from ..persistence.runs import ArchiveRoot
from ..persistence.contracts import ToolOperationAdmission, ToolOperationRecord
from ..execution import (
    ExecutionRecord,
    ExecutionStartReservation,
    ExecutionTerminalCommitResult,
    RuntimePayloadRef,
)
from ..execution import ExecutionRecoveryDetail, ExecutionRecoveryState
from ..session import SessionRecord


@dataclass(slots=True)
class _PendingExecutionTurn:
    tenant_id: str
    reservation: ExecutionStartReservation
    session: SessionRecord | None
    execution: ExecutionRecord
    committing: bool = False
    committed: ExecutionTerminalCommitResult | None = None
    failure: AIError | None = None
    uncertain: bool = False
    reconcile: Callable[[], Awaitable[None]] | None = None
    promotion_claim_id: str | None = None
    history_root: ArchiveRoot | None = None
    history_digest: str | None = None
    coordination: TurnCoordination | None = None
    boundary: asyncio.Lock = field(default_factory=asyncio.Lock)
    tool_operations: dict[str, ToolOperationRecord] = field(default_factory=dict)
    promote_tools: Callable[[], Awaitable[None]] | None = None
    promoted: bool = False

    def pending_tool_operations(self) -> tuple[ToolOperationRecord, ...] | None:
        return None if self.promoted else tuple(self.tool_operations.values())

    async def materialize_tool_operations(self) -> None:
        if not self.promoted:
            if self.promote_tools is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            await self.promote_tools()

    async def stage_tool_operation(self, request: ToolOperationAdmission) -> ToolOperationRecord | None:
        if not request.replay_safe or self.uncertain:
            await self.materialize_tool_operations()
            return None
        async with self.boundary:
            if self.promoted:
                return None
            if self.uncertain:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            self.require_active(request.producer_generation or 0, allow_cancelling=False)
            if (request.execution_id != self.execution.execution_id
                    or request.agent_run_id != self.execution.recovery.agent_run_id
                    or request.binding_digest != self.execution.binding_digest
                    or request.owner != self.execution.producer_claim_id
                    or request.recovery_agent_run_id is not None):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            current = self.tool_operations.get(request.tool_operation_id)
            if current is not None:
                if not current.matches_admission(request):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                if current.status is not ToolOperationStatus.PENDING:
                    return current
                current = replace(current, status=ToolOperationStatus.CLAIMED, owner=request.owner,
                                  fence=current.fence + 1, updated_at=datetime.now(timezone.utc))
            else:
                now = datetime.now(timezone.utc)
                current = ToolOperationRecord(
                    request.tool_operation_id, request.execution_id, request.agent_run_id, request.tool_call_id,
                    request.idempotency_key_digest, request.tool_name, request.arguments_digest,
                    request.binding_digest, True, ToolOperationStatus.CLAIMED, request.owner, 1, None, now, now,
                    producer_generation=request.producer_generation, arguments_payload=request.arguments_payload,
                    capability_ref=request.capability_ref,
                )
            self.tool_operations[current.tool_operation_id] = current
            return current

    async def settle_tool_operation(
        self, operation_id: str, *, owner: str, fence: int, status: ToolOperationStatus,
        result_payload: StoredPayload | None = None, error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> bool:
        """Retain a volatile outcome; this is not a source commit acknowledgement."""
        async with self.boundary:
            if self.promoted:
                return False
            if self.committing or self.committed is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = self.tool_operations.get(operation_id)
            if current is None or current.owner != owner or current.fence != fence:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}:
                if (current.status is not status or current.result_payload != result_payload
                        or current.error_code != error_code or current.error_payload != error_payload):
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                return True
            if current.status not in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if status not in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED,
                              ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.PENDING}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self.tool_operations[operation_id] = replace(
                current, status=status, owner=None if status is ToolOperationStatus.PENDING else owner,
                result_payload=result_payload, error_code=error_code, error_payload=error_payload,
                updated_at=datetime.now(timezone.utc),
            )
            return True

    def activate(
        self, agent_run_id: str, context: RuntimePayloadRef, *, now: datetime,
    ) -> ExecutionRecord:
        if self.committing or self.execution.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        if (self.execution.recovery is None
                or self.execution.recovery_state is not ExecutionRecoveryState.ADMITTED):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if self.coordination is None or self.coordination.execution_id != self.execution.execution_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        # These are semantic revisions, not storage CAS revisions. Promotion
        # or terminal commit acquires durable authority for the captured turn.
        self.execution = replace(
            self.execution, revision=self.execution.revision + 2,
            agent_run_seq=1, input_context=context, updated_at=now,
            producer_generation=self.execution.revision + 1,
            producer_claim_id=self.coordination.owner_id,
            recovery=replace(self.execution.recovery, agent_run_id=agent_run_id),
        )
        return self.execution

    def require_active(self, generation: int, *, allow_cancelling: bool) -> None:
        if self.failure is not None:
            raise self.failure
        if (generation < 1 or generation != self.execution.producer_generation
                or self.committing or self.committed is not None):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if (self.coordination is None
                or self.coordination.execution_id != self.execution.execution_id
                or self.execution.producer_claim_id != self.coordination.owner_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if self.execution.status is ExecutionStatus.CANCELLING:
            if not allow_cancelling:
                raise AIError(ErrorCode.EXECUTION_CANCELLED)
        elif self.execution.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def begin_commit(self) -> None:
        if self.execution.status not in {ExecutionStatus.STARTED, ExecutionStatus.CANCELLING} or self.committing:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.committing = True

    def confirm(self, committed: ExecutionTerminalCommitResult) -> None:
        if not self.committing or self.committed is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if committed.execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self.committed = committed
        self.execution = committed.execution


class _PendingExecutionTurns:
    """One Runtime's accepted turns; entries are never evicted as hot cache."""

    def __init__(self) -> None:
        self.coordination: PendingTurnCoordination | None = None
        self._executions: dict[tuple[str, str], _PendingExecutionTurn] = {}
        self._requests: dict[tuple[str, str, str], _PendingExecutionTurn] = {}
        self._sessions: dict[tuple[str, str], _PendingExecutionTurn] = {}
        self._session_membership_epoch = 0

    def bind_coordination(self, coordination: PendingTurnCoordination | None) -> None:
        self.coordination = coordination

    @property
    def has_pending(self) -> bool:
        return bool(self._executions)

    async def read_remote_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        if self.coordination is None:
            return None
        return await self.coordination.read_execution(execution_id, tenant_id=tenant_id)

    async def cancel_remote(self, execution_id: str, *, tenant_id: str) -> bool:
        return self.coordination is not None and await self.coordination.cancel(execution_id, tenant_id=tenant_id)

    async def read_stream(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, int, str] | None:
        turn = self.get(execution_id, tenant_id=tenant_id)
        if turn is not None and turn.coordination is not None and turn.execution.agent_run_seq == 1:
            turn.require_active(turn.execution.producer_generation, allow_cancelling=True)
            assert turn.execution.producer_claim_id is not None
            return turn.execution, turn.execution.producer_generation, turn.execution.producer_claim_id
        if self.coordination is None:
            return None
        return await self.coordination.read_stream(execution_id, tenant_id=tenant_id)

    def admit(
        self, tenant_id: str, reservation: ExecutionStartReservation,
        session: SessionRecord | None, *, now: datetime,
    ) -> tuple[_PendingExecutionTurn, bool]:
        execution = reservation.execution
        identity = reservation.idempotency
        request_key = (tenant_id, identity.scope, identity.idempotency_key_digest)
        existing = self._requests.get(request_key)
        if existing is not None:
            if (existing.reservation.idempotency.request_digest != identity.request_digest
                    or existing.reservation.idempotency.request_identity_version != identity.request_identity_version
                    or existing.reservation.execution.principal_id != execution.principal_id
                    or existing.reservation.execution.principal_kind != execution.principal_kind):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return existing, False
        if (execution.session_id != (None if session is None else session.session_id)
                or identity.resource_id != execution.execution_id
                or identity.status is not IdempotencyStatus.RESERVED
                or execution.status is not ExecutionStatus.PENDING_START
                or execution.revision != 0 or execution.event_seq != 0
                or execution.agent_run_seq != 0):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        session_key = None if session is None else (tenant_id, session.session_id)
        if session is not None:
            if session.status is not SessionStatus.OPEN:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if session.active_execution_id is not None or session_key in self._sessions:
                raise AIError(ErrorCode.SESSION_BUSY)
        execution_key = (tenant_id, execution.execution_id)
        if execution_key in self._executions:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        reservation, session = deepcopy((reservation, session))
        started = replace(
            reservation.execution, status=ExecutionStatus.STARTED, revision=1,
            event_seq=1, started_at=now, updated_at=now,
            recovery=ExecutionRecoveryDetail(),
        )
        turn = _PendingExecutionTurn(
            tenant_id, reservation, session, started,
        )
        self._executions[execution_key] = turn
        self._requests[request_key] = turn
        if session_key is not None:
            self._sessions[session_key] = turn
            self._session_membership_epoch += 1
        return turn, True

    def observe_session(
        self, session_id: str, *, tenant_id: str,
    ) -> tuple[ExecutionRecord | None, int]:
        """Capture local admission without settling unknown commits.

        The constant-space epoch detects admission/removal between two absent
        observations, including changes to unrelated sessions in this Runtime.
        It is an observation token, not another admission authority.
        """
        turn = self._sessions.get((tenant_id, session_id))
        if turn is not None:
            if turn.failure is not None:
                raise turn.failure
            if turn.uncertain:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return (None if turn is None else turn.execution), self._session_membership_epoch

    def get(self, execution_id: str, *, tenant_id: str) -> _PendingExecutionTurn | None:
        return self._executions.get((tenant_id, execution_id))

    async def read_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        turn = self.get(execution_id, tenant_id=tenant_id)
        if turn is not None:
            await self.settle(turn)
            turn = self.get(execution_id, tenant_id=tenant_id)
            if turn is not None and turn.failure is not None:
                raise turn.failure
        return None if turn is None else turn.execution

    async def settle(self, turn: _PendingExecutionTurn) -> bool:
        if turn.uncertain and turn.reconcile is not None:
            await turn.reconcile()
            return True
        return False

    async def read_session_execution(self, session_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        await self.settle_session(session_id, tenant_id=tenant_id)
        turn = self._sessions.get((tenant_id, session_id))
        if turn is not None:
            return turn.execution
        if self.coordination is None:
            return None
        return await self.coordination.read_session_execution(session_id, tenant_id=tenant_id)

    async def settle_session(self, session_id: str, *, tenant_id: str) -> bool:
        turn = self._sessions.get((tenant_id, session_id))
        return False if turn is None else await self.settle(turn)

    async def settle_all(self) -> None:
        for turn in tuple(self._executions.values()):
            await self.settle(turn)

    def fail(self, turn: _PendingExecutionTurn, error: AIError) -> None:
        turn.failure = error
        turn.uncertain = error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        if not turn.uncertain:
            self._release_admission(turn)

    def peek_request(
        self, *, tenant_id: str, scope: str, idempotency_key_digest: str,
    ) -> _PendingExecutionTurn | None:
        """Observe local admission without settling a possibly unknown commit."""
        return self._requests.get((tenant_id, scope, idempotency_key_digest))

    async def lookup(
        self, *, tenant_id: str, scope: str, idempotency_key_digest: str,
        request_digest: str,
    ) -> _PendingExecutionTurn | None:
        turn = self._requests.get((tenant_id, scope, idempotency_key_digest))
        if turn is not None and turn.reservation.idempotency.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if turn is not None:
            await self.settle(turn)
        return self._requests.get((tenant_id, scope, idempotency_key_digest))

    def release(self, turn: _PendingExecutionTurn) -> None:
        execution = turn.reservation.execution
        execution_key = (turn.tenant_id, execution.execution_id)
        if self._executions.get(execution_key) is not turn:
            return
        self._executions.pop(execution_key)
        self._release_admission(turn)

    def _release_admission(self, turn: _PendingExecutionTurn) -> None:
        identity = turn.reservation.idempotency
        request_key = (turn.tenant_id, identity.scope, identity.idempotency_key_digest)
        if self._requests.get(request_key) is turn:
            self._requests.pop(request_key)
        if turn.session is not None:
            session_key = (turn.tenant_id, turn.session.session_id)
            if self._sessions.get(session_key) is turn:
                self._sessions.pop(session_key)
                self._session_membership_epoch += 1


__all__: list[str] = []
