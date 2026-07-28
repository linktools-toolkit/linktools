"""SQLAlchemy ExecutionStore with bounded database-side paging."""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, MetaData, String, Text, and_, asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ...errors import StorageError
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import BIGSERIAL, TABLE_PREFIX
from ..codec import decode, encode
from ..models import Page, RunApproval, RunDefinitionSnapshot, RunEvent, RunKind, RunRecord, RunSnapshot, RunStatus, RunTraceStep, RunUsage, SessionRecord, SessionTurn


class SessionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}sessions"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[str | None] = mapped_column(String(255))
    next_turn_sequence: Mapped[int] = mapped_column(Integer, default=1)
    latest_completed_run_id: Mapped[str | None] = mapped_column(String(255))


class TurnRow(Base):
    __tablename__ = f"{TABLE_PREFIX}session_turns"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunRow(Base):
    __tablename__ = f"{TABLE_PREFIX}runs"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    session_turn_sequence: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40), index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    fence: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class SnapshotRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_snapshots"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class TraceRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_trace_steps"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class EventRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_events"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvaluationRow(Base):
    __tablename__ = f"{TABLE_PREFIX}run_evaluations"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    evaluator: Mapped[str] = mapped_column(String(255), primary_key=True)
    score: Mapped[float | None]
    result: Mapped[dict[str, Any]] = mapped_column(JSON)


class SqlAlchemyExecutionStore:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    def _session(row: SessionRow) -> SessionRecord:
        return SessionRecord(row.id, row.user_id, row.tenant_id, row.next_turn_sequence, row.latest_completed_run_id, row.created_at, row.updated_at)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            async with session.begin():
                row = SessionRow(id=session_id, user_id=user_id, tenant_id=tenant_id, next_turn_sequence=1, created_at=now, updated_at=now)
                session.add(row)
            return self._session(row)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self.session_factory() as session:
            row = await session.get(SessionRow, session_id)
            return None if row is None else self._session(row)

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        async with self.session_factory() as session:
            query = select(TurnRow).where(TurnRow.session_id == session_id)
            if before_sequence is not None:
                query = query.where(TurnRow.sequence < before_sequence)
            rows = (await session.scalars(query.order_by(desc(TurnRow.sequence)).limit(limit + 1))).all()
        has_more = len(rows) > limit
        page_rows = list(reversed(rows[:limit]))
        values = [SessionTurn(row.session_id, row.sequence, row.run_id, row.payload.get("user_prompt"), row.payload.get("assistant_summary"), RunStatus(row.payload["status"]), row.created_at, row.completed_at) for row in page_rows]
        return Page(tuple(values), has_more, rows[limit - 1].sequence if has_more else None)

    async def load_session_context(self, session_id: str) -> tuple[Any, ...]:
        session = await self.get_session(session_id)
        if session is None or session.latest_completed_run_id is None:
            return ()
        snapshot = await self.get_snapshot(session.latest_completed_run_id)
        return () if snapshot is None else snapshot.resume_messages

    @staticmethod
    def _run(row: RunRow) -> RunRecord:
        payload = row.payload
        approval = payload.get("pending_approval")
        cancelled = payload.get("cancel_requested_at")
        cancelled_at = datetime.fromisoformat(cancelled) if isinstance(cancelled, str) else cancelled
        return RunRecord(row.id, row.session_id, RunKind(row.kind), row.session_turn_sequence, payload.get("parent_run_id"), payload.get("root_run_id", row.id), RunStatus(row.status), RunDefinitionSnapshot(**payload["definition"]), RunApproval(**approval) if approval else None, row.owner, row.fence, None, cancelled_at, payload.get("snapshot_revision", 0), payload.get("trace_sequence", 0), row.created_at, row.updated_at, row.payload.get("tenant_id"), row.payload.get("user_id"))

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self.session_factory() as session:
            row = await session.get(RunRow, run_id)
            return None if row is None else self._run(row)

    async def start_run(self, **kwargs: Any) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                owner = await session.get(SessionRow, kwargs["session_id"], with_for_update=True)
                if owner is None:
                    raise StorageError("unknown session")
                kind = RunKind(kwargs.get("kind", RunKind.USER_TURN))
                sequence = owner.next_turn_sequence if kind == RunKind.USER_TURN else None
                now = datetime.now(timezone.utc)
                payload = {"definition": {"agent_id": kwargs["definition"].agent_id, "model": kwargs["definition"].model, "settings": dict(kwargs["definition"].settings)}, "root_run_id": kwargs.get("root_run_id", kwargs["run_id"]), "parent_run_id": kwargs.get("parent_run_id"), "tenant_id": owner.tenant_id, "user_id": owner.user_id, "trace_sequence": 0, "snapshot_revision": 0}
                row = RunRow(id=kwargs["run_id"], session_id=owner.id, kind=kind.value, session_turn_sequence=sequence, status=RunStatus.RUNNING.value, owner=None, fence=0, payload=payload, created_at=now, updated_at=now)
                session.add(row)
                session.add(EventRow(run_id=kwargs["run_id"], sequence=1, type="run.started", payload={}, created_at=now))
                if sequence is not None:
                    session.add(TurnRow(session_id=owner.id, sequence=sequence, run_id=row.id, payload={"user_prompt": kwargs.get("user_prompt", ""), "assistant_summary": None, "status": RunStatus.RUNNING.value}, created_at=now))
                    owner.next_turn_sequence += 1
                await session.flush()
                return self._run(row)

    async def claim_run(self, run_id: str, *, owner: str, expected_fence: int | None = None) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(RunRow, run_id, with_for_update=True)
                if row is None or expected_fence is not None and row.fence != expected_fence:
                    raise StorageError("run fence conflict")
                if row.status not in (RunStatus.PENDING.value, RunStatus.RUNNING.value):
                    raise StorageError(f"cannot claim run in {row.status} state")
                row.owner, row.fence, row.status, row.updated_at = owner, row.fence + 1, RunStatus.RUNNING.value, datetime.now(timezone.utc)
                return self._run(row)

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[Any, ...]) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(RunRow, run_id, with_for_update=True)
                if row is None or row.payload.get("trace_sequence", 0) != expected_sequence:
                    raise StorageError("trace sequence conflict")
                for offset, step in enumerate(steps, 1):
                    session.add(TraceRow(run_id, expected_sequence + offset, step.kind, step.payload, step.created_at))
                row.payload = {**row.payload, "trace_sequence": expected_sequence + len(steps)}
                await session.flush()
                return expected_sequence + len(steps)

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        async with self.session_factory() as session:
            row = await session.get(SnapshotRow, run_id)
            if row is None:
                return None
            payload = row.payload
            return RunSnapshot("run-snapshot.v1", run_id, row.revision, tuple(payload["resume_messages"]), payload.get("final_output"), RunStatus(payload["status"]), RunUsage(**payload["usage"]), payload["trace_end_sequence"], row.created_at)

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        async with self.session_factory() as session:
            query = select(TraceRow).where(TraceRow.run_id == run_id, TraceRow.sequence > after_sequence).order_by(asc(TraceRow.sequence))
            if through_sequence is not None:
                query = query.where(TraceRow.sequence <= through_sequence)
            rows = (await session.scalars(query)).all()
        return tuple(RunTraceStep(row.run_id, row.sequence, row.kind, row.payload, row.created_at) for row in rows)

    async def _finish(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot, status: RunStatus, pending_approval=None) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(RunRow, run_id, with_for_update=True)
                if row is None or row.owner != owner or row.fence != fence:
                    raise StorageError("run ownership conflict")
                if row.status != RunStatus.RUNNING.value:
                    raise StorageError(f"invalid lifecycle transition from {row.status}")
                row.status = status.value
                row.updated_at = datetime.now(timezone.utc)
                approval = asdict(pending_approval) if pending_approval is not None else row.payload.get("pending_approval")
                row.payload = {**row.payload, "snapshot_revision": snapshot.revision, "trace_sequence": snapshot.trace_end_sequence, "pending_approval": approval}
                session.add(SnapshotRow(run_id=run_id, revision=snapshot.revision, payload={"resume_messages": list(snapshot.resume_messages), "final_output": snapshot.final_output, "status": status.value, "usage": {"input_tokens": snapshot.usage.input_tokens, "output_tokens": snapshot.usage.output_tokens, "total_tokens": snapshot.usage.total_tokens}, "trace_end_sequence": snapshot.trace_end_sequence}, created_at=snapshot.created_at))
                last_event = await session.scalar(select(func.max(EventRow.sequence)).where(EventRow.run_id == run_id)) or 0
                session.add(EventRow(run_id=run_id, sequence=last_event + 1, type=f"run.{status.value}", payload={}, created_at=snapshot.created_at))
                if row.session_turn_sequence is not None:
                    turn = await session.get(TurnRow, (row.session_id, row.session_turn_sequence), with_for_update=True)
                    if turn is not None:
                        turn.payload = {**turn.payload, "status": status.value, "assistant_summary": snapshot.final_output}
                        turn.completed_at = snapshot.created_at
                    if status == RunStatus.COMPLETED:
                        session_row = await session.get(SessionRow, row.session_id, with_for_update=True)
                        if session_row is not None:
                            session_row.latest_completed_run_id = run_id
                await session.flush()
                return self._run(row)

    async def complete_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.COMPLETED)

    async def fail_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.FAILED)

    async def acknowledge_cancel(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.CANCELLED)

    async def pause_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot, pending_approval: object | None = None) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.PAUSED, pending_approval=pending_approval)

    async def heartbeat_run(self, run_id: str, *, owner: str, fence: int) -> RunRecord:
        row = await self.get_run(run_id)
        if row is None or row.execution_owner != owner or row.execution_fence != fence:
            raise StorageError("run ownership conflict")
        return row

    async def request_cancel(self, run_id: str, *, owner: str, fence: int) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                db_row = await session.get(RunRow, run_id, with_for_update=True)
                if db_row is None or db_row.owner != owner or db_row.fence != fence:
                    raise StorageError("run ownership conflict")
                if db_row.status not in (RunStatus.RUNNING.value, RunStatus.PAUSED.value):
                    raise StorageError(f"cannot cancel run in {db_row.status} state")
                db_row.payload = {**db_row.payload, "cancel_requested_at": datetime.now(timezone.utc).isoformat()}
                db_row.updated_at = datetime.now(timezone.utc)
                return self._run(db_row)

    async def decide_approval(self, run_id: str, *, approval_id: str, decision: str, decided_by: str) -> RunRecord:
        async with self.session_factory() as session:
            async with session.begin():
                db_row = await session.get(RunRow, run_id, with_for_update=True)
                row = None if db_row is None else self._run(db_row)
                if row is None or row.status is not RunStatus.PAUSED or row.pending_approval is None or row.pending_approval.approval_id != approval_id:
                    raise StorageError("run has no pending approval")
                db_row.payload = {**db_row.payload, "pending_approval": {**asdict(row.pending_approval), "decision": decision, "decided_by": decided_by}}
                last_event = await session.scalar(select(func.max(EventRow.sequence)).where(EventRow.run_id == run_id)) or 0
                session.add(EventRow(run_id=run_id, sequence=last_event + 1, type="run.approval_decided", payload={"decision": decision}, created_at=datetime.now(timezone.utc)))
                await session.flush()
                return self._run(db_row)

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        async with self.session_factory() as session:
            query = select(EventRow).where(EventRow.run_id == run_id, EventRow.sequence > after_sequence).order_by(asc(EventRow.sequence)).limit(limit + 1)
            rows = (await session.scalars(query)).all()
        events = tuple(RunEvent(row.run_id, row.sequence, row.type, row.payload, row.created_at) for row in rows)
        has_more = len(events) > limit
        return Page(events[:limit], has_more, events[limit - 1].sequence if has_more else None)

    async def save_evaluation(self, evaluation) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                session.add(EvaluationRow(run_id=evaluation.run_id, evaluator=evaluation.evaluator, created_at=evaluation.created_at, score=evaluation.score, result=evaluation.result))

    async def list_evaluations(self, run_id: str) -> tuple[Any, ...]:
        async with self.session_factory() as session:
            rows = (await session.scalars(select(EvaluationRow).where(EvaluationRow.run_id == run_id).order_by(asc(EvaluationRow.created_at)))).all()
        from ..models import RunEvaluation
        return tuple(RunEvaluation(row.run_id, row.evaluator, row.score, row.result, row.created_at) for row in rows)
