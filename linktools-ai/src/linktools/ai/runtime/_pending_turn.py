#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ownership of executions that have not reached a durable boundary."""

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime

from ..core import ExecutionStatus, IdempotencyStatus, SessionStatus
from ..errors import AIError, ErrorCode
from ..storage import TurnCoordination
from ..persistence.coordination import PendingTurnCoordination
from ..persistence.runs import ArchiveRoot
from ..execution import (
    ExecutionRecord,
    ExecutionStartReservation,
    ExecutionTerminalCommitResult,
    RuntimePayloadRef,
)
from ..persistence.contracts import RecoveryCheckpoint, RecoveryCheckpointState
from ..session import SessionRecord


@dataclass(slots=True)
class _PendingExecutionTurn:
    tenant_id: str
    reservation: ExecutionStartReservation
    session: SessionRecord | None
    execution: ExecutionRecord
    recovery: RecoveryCheckpoint
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

    def activate(
        self, agent_run_id: str, context: RuntimePayloadRef, *, now: datetime,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        if self.committing or self.execution.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        if self.recovery.state is not RecoveryCheckpointState.ADMITTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        # These are semantic revisions, not storage CAS revisions. Only the
        # terminal transaction acquires durable authority for the captured turn.
        self.execution = replace(
            self.execution, revision=self.execution.revision + 2,
            agent_run_seq=1, input_context=context, updated_at=now,
        )
        self.recovery = replace(
            self.recovery, revision=self.recovery.revision + 1,
            state=RecoveryCheckpointState.ACTIVE, agent_run_id=agent_run_id,
            updated_at=now,
        )
        return self.execution, self.recovery

    def require_active(self, generation: int, *, allow_cancelling: bool) -> None:
        if self.failure is not None:
            raise self.failure
        if generation != 2 or self.committing or self.committed is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if self.execution.status is ExecutionStatus.CANCELLING:
            if not allow_cancelling:
                raise AIError(ErrorCode.EXECUTION_CANCELLED)
        elif self.execution.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def begin_commit(self) -> None:
        if self.execution.status is ExecutionStatus.CANCELLING:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        if self.execution.status is not ExecutionStatus.STARTED or self.committing:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.committing = True

    def confirm(self, committed: ExecutionTerminalCommitResult) -> None:
        if not self.committing or self.committed is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if committed.execution.status is not ExecutionStatus.SUCCEEDED:
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

    def bind_coordination(self, coordination: PendingTurnCoordination | None) -> None:
        self.coordination = coordination

    async def read_remote_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        if self.coordination is None:
            return None
        return await self.coordination.read_execution(execution_id, tenant_id=tenant_id)

    async def cancel_remote(self, execution_id: str, *, tenant_id: str) -> bool:
        return self.coordination is not None and await self.coordination.cancel(execution_id, tenant_id=tenant_id)

    async def read_stream(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, int, str] | None:
        turn = self.get(execution_id, tenant_id=tenant_id)
        if turn is not None and turn.coordination is not None and turn.execution.agent_run_seq == 1:
            turn.require_active(2, allow_cancelling=True)
            return turn.execution, 2, turn.coordination.owner_id
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
        )
        turn = _PendingExecutionTurn(
            tenant_id, reservation, session, started,
            RecoveryCheckpoint(
                execution_id=execution.execution_id, agent_run_id=None,
                state=RecoveryCheckpointState.ADMITTED, revision=0,
                created_at=now, updated_at=now,
            ),
        )
        self._executions[execution_key] = turn
        self._requests[request_key] = turn
        if session_key is not None:
            self._sessions[session_key] = turn
        return turn, True

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

    async def settle(self, turn: _PendingExecutionTurn) -> None:
        if turn.uncertain and turn.reconcile is not None:
            await turn.reconcile()

    async def read_session_execution(self, session_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        await self.settle_session(session_id, tenant_id=tenant_id)
        turn = self._sessions.get((tenant_id, session_id))
        if turn is not None:
            return turn.execution
        if self.coordination is None:
            return None
        return await self.coordination.read_session_execution(session_id, tenant_id=tenant_id)

    async def settle_session(self, session_id: str, *, tenant_id: str) -> None:
        turn = self._sessions.get((tenant_id, session_id))
        if turn is not None:
            await self.settle(turn)

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


__all__: list[str] = []
