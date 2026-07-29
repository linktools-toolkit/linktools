"""SQLAlchemy execution persistence with transactional lifecycle commands."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, asc, desc, select, update
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.coordination.lease import Lease, claim, renew
from ...errors import StorageConflictError, StorageError
from ..commands import AcknowledgeRunCancel, ClaimRun, CompleteRun, DecideRunApproval, FailRun, HeartbeatRun, PauseRun, RequestRunCancel, ResumeRun, StartRun
from ..lifecycle import assert_approval_decided, assert_claimable, assert_owner, assert_resumable, assert_transition
from ..run import RunApproval, RunDefinition, RunKind, RunRecord, RunStatus, RunnableType, RunUsage
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ..models import NewRunTraceStep, Page, RunEvaluation, RunEvent, RunSnapshot, RunTraceStep, SessionRecord, SessionTurn


class SessionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}sessions"
    session_id: Mapped[str] = mapped_column(String(255), unique=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[str | None] = mapped_column(String(255))
    next_turn_sequence: Mapped[int] = mapped_column(Integer, default=1)
    latest_completed_run_id: Mapped[str | None] = mapped_column(String(255))


class TurnRow(Base):
    __tablename__ = f"{TABLE_PREFIX}session_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence", name="uq_session_turn_sequence"),
    )
    session_id: Mapped[str] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    user_prompt: Mapped[Any] = mapped_column(JSON)
    assistant_summary: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunRow(Base):
    __tablename__ = f"{TABLE_PREFIX}runs"
    run_id: Mapped[str] = mapped_column(String(255), unique=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    runnable_id: Mapped[str] = mapped_column(String(255))
    runnable_type: Mapped[str] = mapped_column(String(40))
    session_turn_sequence: Mapped[int | None] = mapped_column(Integer)
    parent_run_id: Mapped[str | None] = mapped_column(String(255))
    root_run_id: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON)
    pending_approval: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    fence: Mapped[int] = mapped_column(Integer, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snapshot_revision: Mapped[int] = mapped_column(Integer, default=0)
    trace_sequence: Mapped[int] = mapped_column(Integer, default=0)
    event_sequence: Mapped[int] = mapped_column(Integer, default=0)
    tenant_id: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[str | None] = mapped_column(String(255))


class SnapshotRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_snapshots"
    run_id: Mapped[str] = mapped_column(String(255), unique=True)
    revision: Mapped[int] = mapped_column(Integer)
    resume_messages: Mapped[list[Any]] = mapped_column(JSON)
    final_output: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    usage: Mapped[dict[str, int]] = mapped_column(JSON)
    trace_end_sequence: Mapped[int] = mapped_column(Integer)


class TraceRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_trace_steps"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_trace_sequence"),
    )
    run_id: Mapped[str] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class EventRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),
    )
    run_id: Mapped[str] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvaluationRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_evaluations"
    evaluation_id: Mapped[str] = mapped_column(String(255), unique=True)
    run_id: Mapped[str] = mapped_column(String(255), index=True)
    evaluator: Mapped[str] = mapped_column(String(255), index=True)
    score: Mapped[float | None] = mapped_column(Float)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)


def _dt(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _record(row: RunRow) -> RunRecord:
    return RunRecord(row.run_id, row.session_id, RunKind(row.kind), row.runnable_id, RunnableType(row.runnable_type), RunDefinition(**row.definition), RunStatus(row.status), row.session_turn_sequence, row.parent_run_id, row.root_run_id, RunApproval(**row.pending_approval) if row.pending_approval else None, Lease(row.owner, row.fence, _dt(row.lease_expires_at)), _dt(row.cancel_requested_at), row.snapshot_revision, row.trace_sequence, row.event_sequence, row.tenant_id, row.user_id, _dt(row.created_at), _dt(row.updated_at))


class SqlAlchemyExecutionBackend:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    async def _session_row(session, session_id: str, *, for_update: bool = False):
        query = select(SessionRow).where(SessionRow.session_id == session_id)
        if for_update:
            query = query.with_for_update()
        return await session.scalar(query)

    @staticmethod
    async def _run_row(session, run_id: str, *, for_update: bool = False):
        query = select(RunRow).where(RunRow.run_id == run_id)
        if for_update:
            query = query.with_for_update()
        return await session.scalar(query)

    @staticmethod
    async def _snapshot_row(session, run_id: str, *, for_update: bool = False):
        query = select(SnapshotRow).where(SnapshotRow.run_id == run_id)
        if for_update:
            query = query.with_for_update()
        return await session.scalar(query)

    @staticmethod
    async def _turn_row(
        session,
        session_id: str,
        sequence: int,
        *,
        for_update: bool = False,
    ):
        query = select(TurnRow).where(
            TurnRow.session_id == session_id,
            TurnRow.sequence == sequence,
        )
        if for_update:
            query = query.with_for_update()
        return await session.scalar(query)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        async with self.session_factory() as session:
            async with session.begin():
                now = datetime.now(timezone.utc)
                row = SessionRow(session_id=session_id, user_id=user_id, tenant_id=tenant_id, next_turn_sequence=1, latest_completed_run_id=None, created_at=now, updated_at=now)
                session.add(row)
            return SessionRecord(row.session_id, row.user_id, row.tenant_id, row.next_turn_sequence, row.latest_completed_run_id, row.created_at, row.updated_at)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self.session_factory() as session:
            row = await self._session_row(session, session_id)
            return None if row is None else SessionRecord(row.session_id, row.user_id, row.tenant_id, row.next_turn_sequence, row.latest_completed_run_id, row.created_at, row.updated_at)

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page[SessionTurn]:
        async with self.session_factory() as session:
            query = select(TurnRow).where(TurnRow.session_id == session_id)
            if before_sequence is not None:
                query = query.where(TurnRow.sequence < before_sequence)
            rows = (await session.scalars(query.order_by(desc(TurnRow.sequence)).limit(limit + 1))).all()
        values = tuple(reversed(tuple(SessionTurn(row.session_id, row.sequence, row.run_id, row.user_prompt, row.assistant_summary, RunStatus(row.status), row.created_at, row.completed_at) for row in rows[:limit])))
        return Page(values, len(rows) > limit, rows[limit - 1].sequence if len(rows) > limit else None)

    async def load_session_context(self, session_id: str) -> tuple[Any, ...]:
        session = await self.get_session(session_id)
        if session is None or session.latest_completed_run_id is None:
            return ()
        snapshot = await self.get_snapshot(session.latest_completed_run_id)
        return () if snapshot is None else snapshot.resume_messages

    async def start_run(self, command: StartRun) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                owner = await self._session_row(session, command.session_id, for_update=True)
                if owner is None:
                    raise StorageError("unknown session")
                if await self._run_row(session, command.run_id) is not None:
                    raise StorageConflictError("run already exists")
                now = datetime.now(timezone.utc)
                sequence = owner.next_turn_sequence if command.kind is RunKind.USER_TURN else None
                row = RunRow(run_id=command.run_id, session_id=owner.session_id, kind=command.kind.value, runnable_id=command.definition.runnable_id, runnable_type=command.definition.runnable_type.value, session_turn_sequence=sequence, parent_run_id=command.parent_run_id, root_run_id=command.root_run_id or command.run_id, status=RunStatus.PENDING.value, definition=asdict(command.definition), pending_approval=None, owner=None, fence=0, lease_expires_at=None, cancel_requested_at=None, snapshot_revision=0, trace_sequence=0, event_sequence=1, tenant_id=owner.tenant_id, user_id=owner.user_id, created_at=now, updated_at=now)
                session.add(row)
                session.add(EventRow(run_id=command.run_id, sequence=1, type="run.started", payload={}, created_at=now))
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
                    session.add(TurnRow(session_id=owner.session_id, sequence=sequence, run_id=command.run_id, user_prompt=command.user_prompt, assistant_summary=None, status=RunStatus.PENDING.value, created_at=now, updated_at=now, completed_at=None))
                await session.flush()
                return _record(row)

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self.session_factory() as session:
            row = await self._run_row(session, run_id)
            return None if row is None else _record(row)

    async def claim_run(self, command: ClaimRun) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id, for_update=True)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_claimable(record, command.now)
                lease = claim(record.lease, owner=command.owner, now=command.now, duration=command.duration)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == command.run_id,
                        RunRow.status == row.status,
                        RunRow.fence == row.fence,
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
                claimed = await self._run_row(session, command.run_id)
                session.add(EventRow(run_id=claimed.run_id, sequence=event_sequence, type="run.claimed", payload={}, created_at=command.now))
                await session.flush()
                return _record(claimed)

    async def heartbeat_run(self, command: HeartbeatRun) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id, for_update=True)
                if row is None:
                    raise StorageError("unknown run")
                lease = renew(Lease(row.owner, row.fence, row.lease_expires_at), owner=command.owner, fence=command.fence, now=command.now, duration=command.duration)
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == command.run_id,
                        RunRow.owner == row.owner,
                        RunRow.fence == row.fence,
                        RunRow.lease_expires_at == row.lease_expires_at,
                    )
                    .values(lease_expires_at=lease.expires_at, updated_at=command.now)
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lease changed concurrently")
                updated = await self._run_row(session, command.run_id)
                return _record(updated)

    async def request_cancel(self, command: RequestRunCancel) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id, for_update=True)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_owner(record, command.owner, command.fence, command.requested_at)
                assert_transition(record.status, RunStatus.CANCELLING)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == command.run_id,
                        RunRow.status == row.status,
                        RunRow.owner == command.owner,
                        RunRow.fence == command.fence,
                        RunRow.event_sequence == row.event_sequence,
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
                cancelling = await self._run_row(session, command.run_id)
                session.add(EventRow(run_id=row.run_id, sequence=event_sequence, type="run.cancelling", payload={}, created_at=command.requested_at))
                return _record(cancelling)

    async def decide_approval(self, command: DecideRunApproval) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id, for_update=True)
                record = None if row is None else _record(row)
                if record is None or record.status is not RunStatus.PAUSED or record.pending_approval is None or record.pending_approval.approval_id != command.approval_id:
                    raise StorageError("run has no pending approval")
                if record.pending_approval.decision is not None:
                    if (
                        record.pending_approval.decision == command.decision
                        and record.pending_approval.decided_by == command.decided_by
                    ):
                        return record
                    raise StorageConflictError("approval decision conflict")
                decided_at = datetime.now(timezone.utc)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == command.run_id,
                        RunRow.status == RunStatus.PAUSED.value,
                        RunRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        pending_approval=asdict(RunApproval(command.approval_id, record.pending_approval.tool_call_id, record.pending_approval.tool_name, record.pending_approval.arguments, command.decision, command.decided_by)),
                        event_sequence=event_sequence,
                        updated_at=decided_at,
                    )
                )
                decided = await self._run_row(session, command.run_id)
                if result.rowcount != 1:
                    latest = None if decided is None else _record(decided)
                    if (
                        latest is not None
                        and latest.pending_approval is not None
                        and latest.pending_approval.decision == command.decision
                        and latest.pending_approval.decided_by == command.decided_by
                    ):
                        return latest
                    raise StorageConflictError("approval changed concurrently")
                session.add(EventRow(run_id=row.run_id, sequence=event_sequence, type="run.approval_decided", payload={"decision": command.decision}, created_at=decided_at))
                return _record(decided)

    async def resume_run(self, command: ResumeRun) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, command.run_id, for_update=True)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_resumable(record)
                assert_approval_decided(record)
                assert_transition(record.status, RunStatus.PENDING)
                resumed_at = datetime.now(timezone.utc)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == command.run_id,
                        RunRow.status == RunStatus.PAUSED.value,
                        RunRow.event_sequence == row.event_sequence,
                    )
                    .values(
                        status=RunStatus.PENDING.value,
                        pending_approval=None,
                        updated_at=resumed_at,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                resumed = await self._run_row(session, command.run_id)
                session.add(EventRow(run_id=row.run_id, sequence=event_sequence, type="run.resumed", payload={}, created_at=resumed_at))
                return _record(resumed)

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, run_id, for_update=True)
                if row is None or row.trace_sequence != expected_sequence:
                    raise StorageConflictError("trace sequence conflict")
                next_sequence = expected_sequence + len(steps)
                sequence_result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == run_id,
                        RunRow.trace_sequence == expected_sequence,
                    )
                    .values(trace_sequence=next_sequence)
                )
                if sequence_result.rowcount != 1:
                    raise StorageConflictError("trace sequence conflict")
                for offset, step in enumerate(steps, 1):
                    session.add(TraceRow(run_id=run_id, sequence=expected_sequence + offset, kind=step.kind, payload=step.payload, created_at=step.created_at))
                return next_sequence

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        async with self.session_factory() as session:
            query = select(TraceRow).where(TraceRow.run_id == run_id, TraceRow.sequence > after_sequence).order_by(asc(TraceRow.sequence))
            if through_sequence is not None:
                query = query.where(TraceRow.sequence <= through_sequence)
            rows = (await session.scalars(query)).all()
        return tuple(RunTraceStep(row.run_id, row.sequence, row.kind, row.payload, row.created_at) for row in rows)

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        async with self.session_factory() as session:
            row = await self._snapshot_row(session, run_id)
            if row is None:
                return None
            return RunSnapshot("run-snapshot.v1", row.run_id, row.revision, tuple(row.resume_messages), row.final_output, RunStatus(row.status), RunUsage(**row.usage), row.trace_end_sequence, row.created_at)

    async def _finish(self, run_id: str, owner: str, fence: int, snapshot: RunSnapshot, status: RunStatus, pending_approval: RunApproval | None = None) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._run_row(session, run_id, for_update=True)
                if row is None:
                    raise StorageError("unknown run")
                record = _record(row)
                assert_owner(record, owner, fence, snapshot.created_at)
                assert_transition(record.status, status)
                event_sequence = row.event_sequence + 1
                result = await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.run_id == run_id,
                        RunRow.status == row.status,
                        RunRow.owner == owner,
                        RunRow.fence == fence,
                        RunRow.snapshot_revision == row.snapshot_revision,
                    )
                    .values(
                        status=status.value,
                        pending_approval=asdict(pending_approval) if pending_approval else None,
                        owner=None,
                        lease_expires_at=None,
                        snapshot_revision=snapshot.revision,
                        trace_sequence=snapshot.trace_end_sequence,
                        updated_at=snapshot.created_at,
                        event_sequence=event_sequence,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("run lifecycle changed concurrently")
                finished = await self._run_row(session, run_id)
                current = await self._snapshot_row(session, run_id, for_update=True)
                if current is not None and snapshot.revision <= current.revision:
                    raise StorageConflictError("snapshot revision is not increasing")
                if current is None:
                    session.add(SnapshotRow(run_id=run_id, revision=snapshot.revision, resume_messages=list(snapshot.resume_messages), final_output=snapshot.final_output, status=status.value, usage=asdict(snapshot.usage), trace_end_sequence=snapshot.trace_end_sequence, created_at=snapshot.created_at, updated_at=snapshot.created_at))
                else:
                    current.revision, current.resume_messages, current.final_output, current.status, current.usage, current.trace_end_sequence, current.updated_at = snapshot.revision, list(snapshot.resume_messages), snapshot.final_output, status.value, asdict(snapshot.usage), snapshot.trace_end_sequence, snapshot.created_at
                session.add(EventRow(run_id=run_id, sequence=event_sequence, type=f"run.{status.value}", payload={}, created_at=snapshot.created_at))
                if finished.session_turn_sequence is not None:
                    turn = await self._turn_row(
                        session,
                        finished.session_id,
                        finished.session_turn_sequence,
                        for_update=True,
                    )
                    if turn is not None:
                        turn.status, turn.assistant_summary = status.value, snapshot.final_output
                        turn.completed_at = snapshot.created_at if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED} else None
                    if status is RunStatus.COMPLETED:
                        owner_row = await self._session_row(session, finished.session_id, for_update=True)
                        if owner_row is not None:
                            owner_row.latest_completed_run_id = run_id
                await session.flush()
                return _record(finished)

    async def pause_run(self, command: PauseRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.PAUSED, command.pending_approval)

    async def complete_run(self, command: CompleteRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.COMPLETED)

    async def fail_run(self, command: FailRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.FAILED)

    async def acknowledge_cancel(self, command: AcknowledgeRunCancel) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.CANCELLED)

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page[RunEvent]:
        async with self.session_factory() as session:
            rows = (await session.scalars(select(EventRow).where(EventRow.run_id == run_id, EventRow.sequence > after_sequence).order_by(asc(EventRow.sequence)).limit(limit + 1))).all()
        values = tuple(RunEvent(row.run_id, row.sequence, row.type, row.payload, row.created_at) for row in rows)
        return Page(values[:limit], len(values) > limit, values[limit - 1].sequence if len(values) > limit else None)

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                session.add(EvaluationRow(evaluation_id=evaluation.evaluation_id, run_id=evaluation.run_id, evaluator=evaluation.evaluator, score=evaluation.score, result=evaluation.result, created_at=evaluation.created_at, updated_at=evaluation.created_at))

    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]:
        async with self.session_factory() as session:
            rows = (await session.scalars(select(EvaluationRow).where(EvaluationRow.run_id == run_id).order_by(asc(EvaluationRow.created_at)))).all()
        return tuple(RunEvaluation(row.evaluation_id, row.run_id, row.evaluator, row.score, row.result, row.created_at) for row in rows)


__all__ = ["SqlAlchemyExecutionBackend"]
