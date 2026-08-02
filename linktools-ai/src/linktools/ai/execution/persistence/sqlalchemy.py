#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy execution persistence with transactional lifecycle commands."""


import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, UniqueConstraint, asc, desc, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from linktools.core import environ
from ...storage.coordination.lease import Lease, claim, renew
from ...storage.database import CoordinationScope
from ...errors import StorageConflictError, StorageError
from ...json import JsonValue, normalize_json
from ..lifecycle import assert_approval_decided, assert_claimable, assert_owner, assert_resumable, assert_transition
from ..domain import ApprovalDecision, MessageCaptureState, RunApproval, RunDefinition, RunError, RunKind, RunRecord, RunStatus, RunnableType, RunUsage
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc
from ..domain import Page
from ...evaluation import RunEvaluation
from ..session import SessionRecord, SessionTurn
from ..snapshots import RunSnapshot
from ..trace_models import NewRunTraceStep, RunEvent, RunTraceStep

if TYPE_CHECKING:
    from ..commands import AbortExecution, AcknowledgeCancellation, ClaimExecution, CompleteExecution, DecideApproval, FailExecution, HeartbeatExecution, PauseExecution, RequestCancellation, ResumeExecution, StartExecution
    from ..snapshots import AgentSnapshotData
    from sqlalchemy.ext.asyncio import AsyncEngine


class SessionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}sessions"
    __table_args__ = (
        Index("ix_tenant_user", "tenant_id", "user_id", "id"),
    )
    session_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    tenant_id: "Mapped[str | None]" = mapped_column(String(255))
    user_id: "Mapped[str | None]" = mapped_column(String(255))
    next_turn_sequence: "Mapped[int]" = mapped_column(Integer, default=1)
    latest_completed_run_id: "Mapped[str | None]" = mapped_column(String(255))


class TurnRow(Base):
    __tablename__ = f"{TABLE_PREFIX}session_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence", name="uq_session_turn_sequence"),
        Index("ix_session_sequence", "session_id", "sequence"),
    )
    session_id: "Mapped[str]" = mapped_column(String(255))
    sequence: "Mapped[int]" = mapped_column(Integer)
    execution_id: "Mapped[str]" = mapped_column(String(255), unique=True, index=True)
    input: "Mapped[JsonValue]" = mapped_column(JSON)
    # TURN_DELTA: this turn's net-new messages (new_messages()), accumulated
    # across PAUSED/resume. Window-policy immune.
    delta_messages: "Mapped[list[JsonValue]]" = mapped_column(JSON, default=list)
    status: "Mapped[str]" = mapped_column(String(32), index=True)
    # Honesty marker: complete / partial / unavailable (MessageCaptureState).
    capture_state: "Mapped[str]" = mapped_column(String(32), default="complete")
    completed_at: "Mapped[datetime | None]" = mapped_column(DateTime(timezone=True))


class ExecutionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}executions"
    execution_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    session_id: "Mapped[str]" = mapped_column(String(255), index=True)
    kind: "Mapped[str]" = mapped_column(String(40))
    runnable_id: "Mapped[str]" = mapped_column(String(255))
    runnable_type: "Mapped[str]" = mapped_column(String(40))
    session_turn_sequence: "Mapped[int | None]" = mapped_column(Integer)
    parent_execution_id: "Mapped[str | None]" = mapped_column(String(255))
    root_execution_id: "Mapped[str]" = mapped_column(String(255), index=True)
    status: "Mapped[str]" = mapped_column(String(40), index=True)
    definition: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)
    definition_hash: "Mapped[str]" = mapped_column(String(64))
    # Variable payload (input / approval / error) packed into one JSON column to
    # keep the table's large-field count within the DBA limit; _record/_finish
    # (un)pack it. `definition` stays separate (it drives definition_hash + the
    # RunDefinition rebuild).
    data: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)
    owner: "Mapped[str | None]" = mapped_column(String(255))
    fence: "Mapped[int]" = mapped_column(Integer, default=0)
    lease_expires_at: "Mapped[datetime | None]" = mapped_column(DateTime(timezone=True), index=True)
    cancel_requested_at: "Mapped[datetime | None]" = mapped_column(DateTime(timezone=True))
    snapshot_revision: "Mapped[int]" = mapped_column(Integer, default=0)
    trace_sequence: "Mapped[int]" = mapped_column(Integer, default=0)
    event_sequence: "Mapped[int]" = mapped_column(Integer, default=0)
    tenant_id: "Mapped[str | None]" = mapped_column(String(255))
    user_id: "Mapped[str | None]" = mapped_column(String(255))


class SnapshotRow(Base):
    __tablename__ = f"{TABLE_PREFIX}execution_snapshots"
    execution_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    revision: "Mapped[int]" = mapped_column(Integer)
    # RESUME_CHECKPOINT (column name retained): non-empty ONLY for PAUSED --
    # holds all_messages() at the pause point. _finish clears it on every
    # terminal state so no cumulative history is retained (O(N^2) eliminated).
    resume_messages: "Mapped[list[JsonValue]]" = mapped_column(JSON, default=list)
    # final_output + usage packed into one JSON column (DBA large-field limit).
    outcome: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)
    status: "Mapped[str]" = mapped_column(String(32))
    trace_end_sequence: "Mapped[int]" = mapped_column(Integer)


class TraceRow(Base):
    __tablename__ = f"{TABLE_PREFIX}execution_trace_steps"
    __table_args__ = (
        UniqueConstraint("execution_id", "sequence", name="uq_execution_trace_sequence"),
    )
    execution_id: "Mapped[str]" = mapped_column(String(255))
    sequence: "Mapped[int]" = mapped_column(Integer)
    kind: "Mapped[str]" = mapped_column(String(40))
    payload: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)


class EventRow(Base):
    __tablename__ = f"{TABLE_PREFIX}execution_events"
    __table_args__ = (
        UniqueConstraint("execution_id", "sequence", name="uq_execution_event_sequence"),
    )
    execution_id: "Mapped[str]" = mapped_column(String(255))
    sequence: "Mapped[int]" = mapped_column(Integer)
    type: "Mapped[str]" = mapped_column(String(120))
    payload: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)


class EvaluationRow(Base):
    __tablename__ = f"{TABLE_PREFIX}execution_evaluations"
    __table_args__ = (
        Index("ix_execution_created", "execution_id", "created_at"),
    )
    evaluation_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    execution_id: "Mapped[str]" = mapped_column(String(255), index=True)
    evaluator: "Mapped[str]" = mapped_column(String(255), index=True)
    score: "Mapped[float | None]" = mapped_column(Float)
    result: "Mapped[dict[str, JsonValue]]" = mapped_column(JSON)


logger = environ.get_logger("ai.execution.persistence.sqlalchemy")


def _approval(data: "dict[str, JsonValue]") -> RunApproval:
    decision = data.get("decision")
    return RunApproval(
        data["approval_id"],
        data["tool_call_id"],
        data["tool_name"],
        data["binding_fingerprint"],
        ApprovalDecision(decision) if decision else None,
        data.get("decided_by"),
        as_utc(datetime.fromisoformat(data["decided_at"]))
        if data.get("decided_at")
        else None,
    )


async def _none() -> None:
    """Async ``None`` placeholder for optional branches of a ``gather``."""
    return None


def _record(row: ExecutionRow) -> RunRecord:
    data = row.data or {}
    return RunRecord(
        id=row.execution_id,
        session_id=row.session_id,
        kind=RunKind(row.kind),
        runnable_id=row.runnable_id,
        runnable_type=RunnableType(row.runnable_type),
        definition=RunDefinition(**row.definition),
        status=RunStatus(row.status),
        session_turn_sequence=row.session_turn_sequence,
        parent_execution_id=row.parent_execution_id,
        root_execution_id=row.root_execution_id,
        approval=_approval(data["approval"]) if data.get("approval") else None,
        lease=Lease(row.owner, row.fence, as_utc(row.lease_expires_at)),
        cancel_requested_at=as_utc(row.cancel_requested_at),
        snapshot_revision=row.snapshot_revision,
        trace_sequence=row.trace_sequence,
        event_sequence=row.event_sequence,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        created_at=as_utc(row.created_at),
        updated_at=as_utc(row.updated_at),
        error=RunError(**data["error"]) if data.get("error") else None,
        input=data.get("input"),
    )


class SqlAlchemyExecutionBackend:
    coordination_scope = CoordinationScope.SHARED_DATABASE

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    async def _session_row(session, session_id: str):
        return await session.scalar(
            select(SessionRow).where(SessionRow.session_id == session_id)
        )

    @staticmethod
    async def _run_row(session, run_id: str):
        return await session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == run_id)
        )

    @staticmethod
    async def _snapshot_row(session, run_id: str):
        return await session.scalar(
            select(SnapshotRow).where(SnapshotRow.execution_id == run_id)
        )

    @staticmethod
    async def _turn_row(session, session_id: str, sequence: int):
        return await session.scalar(
            select(TurnRow).where(
                TurnRow.session_id == session_id,
                TurnRow.sequence == sequence,
            )
        )

    async def create_session(self, *, session_id: str, user_id: "str | None", tenant_id: "str | None") -> SessionRecord:
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    existing = await self._session_row(
                        session,
                        session_id,
                    )
                    if existing is not None:
                        return self._owned_session(
                            existing,
                            user_id=user_id,
                            tenant_id=tenant_id,
                        )
                    now = datetime.now(timezone.utc)
                    row = SessionRow(session_id=session_id, user_id=user_id, tenant_id=tenant_id, next_turn_sequence=1, latest_completed_run_id=None, created_at=now, updated_at=now)
                    session.add(row)
                return self._owned_session(
                    row,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
        except IntegrityError:
            async with self.session_factory() as session:
                row = await self._session_row(session, session_id)
                if row is None:
                    raise StorageConflictError(
                        "session creation conflicted without a visible row"
                    )
                return self._owned_session(
                    row,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )

    @staticmethod
    def _owned_session(
        row: SessionRow,
        *,
        user_id: "str | None",
        tenant_id: "str | None",
    ) -> SessionRecord:
        if row.user_id != user_id or row.tenant_id != tenant_id:
            raise StorageConflictError("session ownership conflict")
        return SessionRecord(
            row.session_id,
            row.user_id,
            row.tenant_id,
            row.next_turn_sequence,
            row.latest_completed_run_id,
            as_utc(row.created_at),
            as_utc(row.updated_at),
        )

    async def get_session(self, session_id: str) -> "SessionRecord | None":
        async with self.session_factory() as session:
            row = await self._session_row(session, session_id)
            return None if row is None else SessionRecord(row.session_id, row.user_id, row.tenant_id, row.next_turn_sequence, row.latest_completed_run_id, row.created_at, row.updated_at)

    async def list_session_turns(self, session_id: str, *, before_sequence: "int | None" = None, limit: int = 50) -> "Page[SessionTurn]":
        async with self.session_factory() as session:
            query = select(TurnRow).where(TurnRow.session_id == session_id)
            if before_sequence is not None:
                query = query.where(TurnRow.sequence < before_sequence)
            rows = (await session.scalars(query.order_by(desc(TurnRow.sequence)).limit(limit + 1))).all()
        values = tuple(reversed(tuple(SessionTurn(row.session_id, row.sequence, row.execution_id, row.input, tuple(row.delta_messages or ()), RunStatus(row.status), MessageCaptureState(row.capture_state), row.created_at, row.completed_at) for row in rows[:limit])))
        return Page(values, len(rows) > limit, rows[limit - 1].sequence if len(rows) > limit else None)

    async def load_session_context(self, session_id: str) -> "tuple[JsonValue, ...]":
        # Session Context: COMPLETED turns' TURN_DELTA concatenated in sequence
        # order. 1 query. PAUSED/CANCELLED/FAILED deltas are excluded so a
        # paused-then-cancelled turn never pollutes the next turn's context.
        async with self.session_factory() as session:
            rows = (await session.scalars(
                select(TurnRow.delta_messages)
                .where(TurnRow.session_id == session_id, TurnRow.status == RunStatus.COMPLETED.value)
                .order_by(TurnRow.sequence)
            )).all()
        return tuple(msg for delta in rows for msg in (delta or ()))

    async def load_resume_messages(self, execution_id: str) -> "tuple[JsonValue, ...]":
        # Resume Context: the PAUSED run's RESUME_CHECKPOINT only.
        snapshot = await self.get_snapshot(execution_id)
        return () if snapshot is None else snapshot.resume_messages

    async def get_session_messages(self, session_id: str) -> "tuple[SessionTurn, ...]":
        # Audit History: every turn (any status) with its TURN_DELTA + status +
        # capture_state, in sequence order. The query layer shapes/groups it.
        async with self.session_factory() as session:
            rows = (await session.scalars(
                select(TurnRow)
                .where(TurnRow.session_id == session_id)
                .order_by(TurnRow.sequence)
            )).all()
        return tuple(
            SessionTurn(row.session_id, row.sequence, row.execution_id, row.input, tuple(row.delta_messages or ()), RunStatus(row.status), MessageCaptureState(row.capture_state), row.created_at, row.completed_at)
            for row in rows
        )

    async def get_turn(self, session_id: str, sequence: int) -> "SessionTurn | None":
        # O(1) single-turn read by (session_id, sequence); avoids pulling the
        # whole session when only one turn's delta is needed.
        async with self.session_factory() as session:
            row = await self._turn_row(session, session_id, sequence)
        if row is None:
            return None
        return SessionTurn(row.session_id, row.sequence, row.execution_id, row.input, tuple(row.delta_messages or ()), RunStatus(row.status), MessageCaptureState(row.capture_state), row.created_at, row.completed_at)

    async def start_run(self, command: "StartExecution") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                owner = await self._session_row(session, command.session_id)
                if owner is None:
                    raise StorageError("unknown session")
                if await self._run_row(session, command.run_id) is not None:
                    raise StorageConflictError("run already exists")
                now = datetime.now(timezone.utc)
                sequence = owner.next_turn_sequence if command.kind is RunKind.USER_TURN else None
                row = ExecutionRow(execution_id=command.run_id, session_id=owner.session_id, kind=command.kind.value, runnable_id=command.definition.runnable_id, runnable_type=command.definition.runnable_type.value, session_turn_sequence=sequence, parent_execution_id=command.parent_execution_id, root_execution_id=command.root_execution_id or command.run_id, status=RunStatus.PENDING.value, definition=asdict(command.definition), definition_hash=command.definition.spec_hash, data={"input": command.input}, owner=None, fence=0, lease_expires_at=None, cancel_requested_at=None, snapshot_revision=0, trace_sequence=0, event_sequence=1, tenant_id=owner.tenant_id, user_id=owner.user_id, created_at=now, updated_at=now)
                session.add(row)
                session.add(EventRow(execution_id=command.run_id, sequence=1, type="run.started", payload={}, created_at=now))
                if sequence is not None:
                    sequence_result = await session.execute(
                        update(SessionRow)
                        .where(
                            SessionRow.session_id == owner.session_id,
                            SessionRow.next_turn_sequence == sequence,
                        )
                        .values(next_turn_sequence=sequence + 1, updated_at=now)
                    )
                    if sequence_result.rowcount != 1:
                        raise StorageConflictError("session turn sequence conflict")
                    session.add(TurnRow(session_id=owner.session_id, sequence=sequence, execution_id=command.run_id, input=command.input, delta_messages=[], status=RunStatus.PENDING.value, capture_state=MessageCaptureState.COMPLETE.value, created_at=now, updated_at=now, completed_at=None))
                await session.flush()
                return _record(row)

    async def get_run(self, run_id: str) -> "RunRecord | None":
        async with self.session_factory() as session:
            row = await self._run_row(session, run_id)
            return None if row is None else _record(row)

    async def claim_run(self, command: "ClaimExecution") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_claimable(record, command.now)
                lease = claim(record.lease, owner=command.owner, now=command.now, duration=command.duration)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.status == row.status,
                        ExecutionRow.fence == row.fence,
                    )
                    .values(
                        status=RunStatus.RUNNING.value,
                        owner=lease.owner,
                        fence=lease.fence,
                        lease_expires_at=lease.expires_at,
                        updated_at=command.now,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run claim conflict")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = RunStatus.RUNNING.value
                row.owner = lease.owner
                row.fence = lease.fence
                row.lease_expires_at = lease.expires_at
                row.updated_at = command.now
                row.event_sequence = event_sequence
                session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.claimed", payload={}, created_at=command.now))
                await session.flush()
                return _record(row)

    async def heartbeat_run(self, command: "HeartbeatExecution") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                if row is None:
                    raise StorageError("unknown run")
                lease = renew(Lease(row.owner, row.fence, row.lease_expires_at), owner=command.owner, fence=command.fence, now=command.now, duration=command.duration)
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.owner == row.owner,
                        ExecutionRow.fence == row.fence,
                        ExecutionRow.lease_expires_at == row.lease_expires_at,
                    )
                    .values(lease_expires_at=lease.expires_at, updated_at=command.now)
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lease changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.lease_expires_at = lease.expires_at
                row.updated_at = command.now
                return _record(row)

    async def request_cancel(self, command: "RequestCancellation") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                if record.status in {RunStatus.PENDING, RunStatus.PAUSED}:
                    # Direct terminal cancel: PENDING was never claimed, PAUSED
                    # lease already released. No owner/fence check needed.
                    event_sequence = row.event_sequence + 1
                    now = command.requested_at
                    result = await session.execute(
                        update(ExecutionRow)
                        .where(
                            ExecutionRow.execution_id == command.run_id,
                            ExecutionRow.status == row.status,
                            ExecutionRow.event_sequence == row.event_sequence,
                        )
                        .values(
                            status=RunStatus.CANCELLED.value,
                            owner=None,
                            lease_expires_at=None,
                            cancel_requested_at=now,
                            updated_at=now,
                            event_sequence=event_sequence,
                        )
                    )
                    if result.rowcount != 1:
                        raise StorageConflictError("run lifecycle changed concurrently")
                    # CAS matched at the held state: reflect the UPDATE's values
                    # onto the loaded row and build the record without a re-SELECT.
                    row.status = RunStatus.CANCELLED.value
                    row.owner = None
                    row.lease_expires_at = None
                    row.cancel_requested_at = now
                    row.updated_at = now
                    row.event_sequence = event_sequence
                    session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.cancelled", payload={}, created_at=now))
                    if row.session_turn_sequence is not None:
                        turn = await self._turn_row(session, row.session_id, row.session_turn_sequence)
                        if turn is not None:
                            # PENDING->CANCELLED: no TURN_DELTA produced (AgentRun
                            # never started) -> COMPLETE. PAUSED->CANCELLED: the
                            # paused delta is already committed and retained ->
                            # COMPLETE. Either way the checkpoint must be cleared
                            # (no longer resumable) inside this same transaction.
                            turn.status = RunStatus.CANCELLED.value
                            turn.capture_state = MessageCaptureState.COMPLETE.value
                            turn.completed_at = now
                            snapshot_row = await self._snapshot_row(session, row.execution_id)
                            if snapshot_row is not None:
                                snapshot_row.resume_messages = []
                                snapshot_row.status = RunStatus.CANCELLED.value
                    return _record(row)
                assert_owner(record, command.owner, command.fence, command.requested_at)
                assert_transition(record.status, RunStatus.CANCELLING)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.status == row.status,
                        ExecutionRow.owner == command.owner,
                        ExecutionRow.fence == command.fence,
                        ExecutionRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        status=RunStatus.CANCELLING.value,
                        cancel_requested_at=command.requested_at,
                        updated_at=command.requested_at,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = RunStatus.CANCELLING.value
                row.cancel_requested_at = command.requested_at
                row.updated_at = command.requested_at
                row.event_sequence = event_sequence
                session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.cancelling", payload={}, created_at=command.requested_at))
                return _record(row)

    async def decide_approval(self, command: "DecideApproval") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                record = None if row is None else _record(row)
                if record is None or record.approval is None or record.approval.approval_id != command.approval_id:
                    raise StorageError("run has no pending approval")
                existing = record.approval.decision
                if existing is not None:
                    # Already decided: idempotent only for the same decision+decider.
                    # ALLOW leaves the run PAUSED; DENY already moved it to CANCELLED --
                    # both are reachable here for a replay of the recorded decision.
                    if existing == command.decision and record.approval.decided_by == command.decided_by:
                        return record
                    raise StorageConflictError("approval decision conflict")
                if record.status is not RunStatus.PAUSED:
                    raise StorageError("run has no pending approval")
                decided_at = datetime.now(timezone.utc)
                event_sequence = row.event_sequence + 1
                new_approval = normalize_json(asdict(
                    RunApproval(
                        command.approval_id,
                        record.approval.tool_call_id,
                        record.approval.tool_name,
                        record.approval.binding_fingerprint,
                        command.decision,
                        command.decided_by,
                        decided_at,
                    )
                ))
                if command.decision == ApprovalDecision.ALLOW:
                    result = await session.execute(
                        update(ExecutionRow)
                        .where(
                            ExecutionRow.execution_id == command.run_id,
                            ExecutionRow.status == RunStatus.PAUSED.value,
                            ExecutionRow.event_sequence == row.event_sequence,
                        )
                        .values(
                            data={**(row.data or {}), "approval": new_approval},
                            event_sequence=event_sequence,
                            updated_at=decided_at,
                        )
                    )
                    if result.rowcount != 1:
                        raise StorageConflictError("approval changed concurrently")
                    # CAS matched at the held state: reflect the UPDATE's values
                    # onto the loaded row and build the record without a re-SELECT.
                    row.data = {**(row.data or {}), "approval": new_approval}
                    row.event_sequence = event_sequence
                    row.updated_at = decided_at
                    session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.approval_decided", payload=new_approval, created_at=decided_at))
                    return _record(row)
                # DENY: terminal -- execution -> CANCELLED, lease released, turn -> CANCELLED.
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.status == RunStatus.PAUSED.value,
                        ExecutionRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        status=RunStatus.CANCELLED.value,
                        data={**(row.data or {}), "approval": new_approval},
                        owner=None,
                        lease_expires_at=None,
                        event_sequence=event_sequence,
                        updated_at=decided_at,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("approval changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = RunStatus.CANCELLED.value
                row.data = {**(row.data or {}), "approval": new_approval}
                row.owner = None
                row.lease_expires_at = None
                row.event_sequence = event_sequence
                row.updated_at = decided_at
                session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.approval_decided", payload=new_approval, created_at=decided_at))
                if record.session_turn_sequence is not None:
                    turn = await self._turn_row(session, record.session_id, record.session_turn_sequence)
                    if turn is not None:
                        # PAUSED->DENY->CANCELLED: same rule as PAUSED->CANCELLED --
                        # the paused delta is already committed (COMPLETE); clear
                        # the checkpoint (no longer resumable) in this transaction.
                        turn.status = RunStatus.CANCELLED.value
                        turn.capture_state = MessageCaptureState.COMPLETE.value
                        turn.completed_at = decided_at
                        snapshot_row = await self._snapshot_row(session, row.execution_id)
                        if snapshot_row is not None:
                            snapshot_row.resume_messages = []
                            snapshot_row.status = RunStatus.CANCELLED.value
                return _record(row)

    async def resume_run(self, command: "ResumeExecution") -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_resumable(record)
                assert_approval_decided(record)
                assert_transition(record.status, RunStatus.PENDING)
                resumed_at = datetime.now(timezone.utc)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.status == RunStatus.PAUSED.value,
                        ExecutionRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        status=RunStatus.PENDING.value,
                        updated_at=resumed_at,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = RunStatus.PENDING.value
                row.updated_at = resumed_at
                row.event_sequence = event_sequence
                session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.resumed", payload={}, created_at=resumed_at))
                return _record(row)

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: "tuple[NewRunTraceStep, ...]") -> int:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, run_id)
                if row is None or row.trace_sequence != expected_sequence:
                    raise StorageConflictError("trace sequence conflict")
                next_sequence = expected_sequence + len(steps)
                sequence_result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == run_id,
                        ExecutionRow.trace_sequence == expected_sequence,
                    )
                    .values(trace_sequence=next_sequence)
                )
                if sequence_result.rowcount != 1:
                    raise StorageConflictError("trace sequence conflict")
                for offset, step in enumerate(steps, 1):
                    session.add(TraceRow(execution_id=run_id, sequence=expected_sequence + offset, kind=step.kind, payload=step.payload, created_at=step.created_at))
                return next_sequence

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: "int | None" = None) -> "tuple[RunTraceStep, ...]":
        async with self.session_factory() as session:
            query = select(TraceRow).where(TraceRow.execution_id == run_id, TraceRow.sequence > after_sequence).order_by(asc(TraceRow.sequence))
            if through_sequence is not None:
                query = query.where(TraceRow.sequence <= through_sequence)
            rows = (await session.scalars(query)).all()
        return tuple(RunTraceStep(row.execution_id, row.sequence, row.kind, row.payload, row.created_at) for row in rows)

    async def get_snapshot(self, run_id: str) -> "RunSnapshot | None":
        async with self.session_factory() as session:
            row = await self._snapshot_row(session, run_id)
            if row is None:
                return None
            outcome = row.outcome or {}
            return RunSnapshot("run-snapshot.v1", row.execution_id, row.revision, tuple(row.resume_messages), outcome.get("final_output"), RunStatus(row.status), RunUsage(**(outcome.get("usage") or {})), row.trace_end_sequence, row.created_at)

    async def _finish(self, run_id: str, owner: str, fence: int, snapshot: "AgentSnapshotData", status: RunStatus, pending_approval: "RunApproval | None" = None, error: "RunError | None" = None) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, run_id)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                now = datetime.now(timezone.utc)
                assert_owner(record, owner, fence, now)
                assert_transition(record.status, status)
                event_sequence = row.event_sequence + 1
                # The store allocates the snapshot revision (expected + 1), not
                # the engine; the CAS on the row's snapshot_revision fences out
                # a stale execution's concurrent commit.
                new_revision = row.snapshot_revision + 1
                prior_data = row.data or {}
                merged_data = {
                    **prior_data,
                    "approval": asdict(pending_approval) if pending_approval is not None else prior_data.get("approval"),
                    "error": asdict(error) if error else None,
                }
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == run_id,
                        ExecutionRow.status == row.status,
                        ExecutionRow.owner == owner,
                        ExecutionRow.fence == fence,
                        ExecutionRow.snapshot_revision == row.snapshot_revision,
                    )
                    .values(
                        status=status.value,
                        data=merged_data,
                        owner=None,
                        lease_expires_at=None,
                        snapshot_revision=new_revision,
                        trace_sequence=snapshot.trace_end_sequence,
                        updated_at=now,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row -- no re-SELECT. ``row`` now carries the finished
                # state (status, data, owner, snapshot_revision, trace_sequence,
                # event_sequence, updated_at) and the unchanged business columns
                # (session_id, session_turn_sequence, ...) the sub-updates below
                # and the final _record() need.
                row.status = status.value
                row.data = merged_data
                row.owner = None
                row.lease_expires_at = None
                row.snapshot_revision = new_revision
                row.trace_sequence = snapshot.trace_end_sequence
                row.updated_at = now
                row.event_sequence = event_sequence
                # The snapshot, turn, and session-owner lookups are mutually
                # independent (each keys off columns already on ``row``); run them
                # concurrently so the round trips overlap under a multi-connection
                # pool (under SQLite single-connection they serialize transparently).
                needs_turn = row.session_turn_sequence is not None
                needs_owner = needs_turn and status is RunStatus.COMPLETED
                current, turn, owner_row = await asyncio.gather(
                    self._snapshot_row(session, run_id),
                    self._turn_row(session, row.session_id, row.session_turn_sequence) if needs_turn else _none(),
                    self._session_row(session, row.session_id) if needs_owner else _none(),
                )
                # RESUME_CHECKPOINT: non-empty ONLY for PAUSED. Every terminal
                # state clears it so steady-state snapshots hold no cumulative
                # history (O(N^2) eliminated).
                checkpoint = list(snapshot.checkpoint_messages) if status is RunStatus.PAUSED else []
                if current is None:
                    session.add(SnapshotRow(execution_id=run_id, revision=new_revision, resume_messages=checkpoint, outcome={"final_output": snapshot.final_output, "usage": asdict(snapshot.usage)}, status=status.value, trace_end_sequence=snapshot.trace_end_sequence, created_at=now, updated_at=now))
                else:
                    current.revision, current.resume_messages, current.outcome, current.status, current.trace_end_sequence, current.updated_at = new_revision, checkpoint, {"final_output": snapshot.final_output, "usage": asdict(snapshot.usage)}, status.value, snapshot.trace_end_sequence, now
                session.add(EventRow(execution_id=run_id, sequence=event_sequence, type=f"run.{status.value}", payload={}, created_at=now))
                if turn is not None:
                    # TURN_DELTA: append this run's new_messages() onto whatever
                    # the turn already holds. A fresh COMPLETED turn starts empty
                    # (append == set); a resumed turn already holds the paused
                    # m_partial and appends m_resume, so the audit delta is the
                    # full turn contribution, appearing exactly once.
                    turn.delta_messages = list(turn.delta_messages or []) + list(snapshot.delta_messages)
                    turn.status = status.value
                    turn.capture_state = snapshot.capture_state.value
                    turn.completed_at = now if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED} else None
                if owner_row is not None:
                    owner_row.latest_completed_run_id = run_id
                await session.flush()
                if environ.debug:
                    logger.debug(
                        "run %s finished: status=%s revision=%s checkpoint_len=%s",
                        run_id, status.value, new_revision, len(checkpoint),
                    )
                return _record(row)

    async def pause_run(self, command: "PauseExecution") -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.PAUSED, command.pending_approval)

    async def complete_run(self, command: "CompleteExecution") -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.COMPLETED)

    async def fail_run(self, command: "FailExecution") -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.FAILED, error=command.error)

    async def acknowledge_cancel(self, command: "AcknowledgeCancellation") -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.CANCELLED)

    async def abort_run(self, command: "AbortExecution") -> RunRecord:
        # Unlike fail_run, there is no snapshot to persist here -- an
        # AbortExecution fires on a programming/config/protocol error, before
        # the engine ever produced a coherent outcome to snapshot.
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_owner(record, command.owner, command.fence, datetime.now(timezone.utc))
                assert_transition(record.status, RunStatus.FAILED)
                event_sequence = row.event_sequence + 1
                now = datetime.now(timezone.utc)
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == command.run_id,
                        ExecutionRow.status == row.status,
                        ExecutionRow.owner == command.owner,
                        ExecutionRow.fence == command.fence,
                        ExecutionRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        status=RunStatus.FAILED.value,
                        data={**(row.data or {}), "error": asdict(command.error)},
                        owner=None,
                        lease_expires_at=None,
                        trace_sequence=command.trace_end_sequence,
                        updated_at=now,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = RunStatus.FAILED.value
                row.data = {**(row.data or {}), "error": asdict(command.error)}
                row.owner = None
                row.lease_expires_at = None
                row.trace_sequence = command.trace_end_sequence
                row.updated_at = now
                row.event_sequence = event_sequence
                session.add(EventRow(execution_id=row.execution_id, sequence=event_sequence, type="run.aborted", payload={}, created_at=now))
                if row.session_turn_sequence is not None:
                    turn = await self._turn_row(session, row.session_id, row.session_turn_sequence)
                    if turn is not None:
                        # abort_run fires before the engine produced a snapshot,
                        # so this turn has no trustworthy delta -> UNAVAILABLE
                        # (distinct from fail_run, whose PARTIAL delta the
                        # engine salvaged via _finish).
                        turn.status = RunStatus.FAILED.value
                        turn.capture_state = MessageCaptureState.UNAVAILABLE.value
                        turn.completed_at = now
                await session.flush()
                return _record(row)

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> "Page[RunEvent]":
        async with self.session_factory() as session:
            rows = (await session.scalars(select(EventRow).where(EventRow.execution_id == run_id, EventRow.sequence > after_sequence).order_by(asc(EventRow.sequence)).limit(limit + 1))).all()
        values = tuple(RunEvent(row.execution_id, row.sequence, row.type, row.payload, row.created_at) for row in rows)
        return Page(values[:limit], len(values) > limit, values[limit - 1].sequence if len(values) > limit else None)

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                session.add(EvaluationRow(evaluation_id=evaluation.evaluation_id, execution_id=evaluation.run_id, evaluator=evaluation.evaluator, score=evaluation.score, result=evaluation.result, created_at=evaluation.created_at, updated_at=evaluation.created_at))

    async def list_evaluations(self, run_id: str) -> "tuple[RunEvaluation, ...]":
        async with self.session_factory() as session:
            rows = (await session.scalars(select(EvaluationRow).where(EvaluationRow.execution_id == run_id).order_by(asc(EvaluationRow.created_at)))).all()
        return tuple(RunEvaluation(row.evaluation_id, row.execution_id, row.evaluator, row.score, row.result, row.created_at) for row in rows)


__all__ = ["SqlAlchemyExecutionBackend"]
