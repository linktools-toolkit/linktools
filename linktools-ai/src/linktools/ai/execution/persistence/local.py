"""Single-process JSON-file execution persistence."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ...storage.coordination.lease import Lease, assert_active, claim, release, renew
from ...errors import StorageConflictError, StorageCorruptionError, StorageError
from ...storage.local.files import atomic_write_json, read_json
from ...storage.local.locks import KeyedLocks
from ...storage.local.paths import StorageId, safe_child
from ..commands import (
    AcknowledgeRunCancel,
    ClaimRun,
    CompleteRun,
    DecideRunApproval,
    FailRun,
    HeartbeatRun,
    PauseRun,
    RequestRunCancel,
    ResumeRun,
    StartRun,
)
from ..lifecycle import assert_approval_decided, assert_claimable, assert_owner, assert_resumable, assert_transition
from ..run import RunApproval, RunDefinition, RunKind, RunRecord, RunStatus, RunUsage, RunnableType
from ..models import Page, RunEvaluation, RunEvent, RunSnapshot, RunTraceStep, SessionRecord, SessionTurn, NewRunTraceStep


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def _run(raw: dict) -> RunRecord:
    lease_raw = raw.pop("lease", {})
    raw["kind"] = RunKind(raw["kind"])
    raw["status"] = RunStatus(raw["status"])
    definition = raw["definition"]
    definition["runnable_type"] = RunnableType(definition["runnable_type"])
    raw["definition"] = RunDefinition(**definition)
    raw["lease"] = Lease(lease_raw.get("owner"), lease_raw.get("fence", 0), _dt(lease_raw.get("expires_at")))
    if raw.get("pending_approval") is not None:
        raw["pending_approval"] = RunApproval(**raw["pending_approval"])
    raw["cancel_requested_at"] = _dt(raw.get("cancel_requested_at"))
    raw["created_at"] = _dt(raw["created_at"])
    raw["updated_at"] = _dt(raw["updated_at"])
    return RunRecord(**raw)


def _session(raw: dict) -> SessionRecord:
    raw["created_at"] = _dt(raw["created_at"])
    raw["updated_at"] = _dt(raw["updated_at"])
    return SessionRecord(**raw)


def _turn(raw: dict) -> SessionTurn:
    raw["status"] = RunStatus(raw["status"])
    raw["created_at"] = _dt(raw["created_at"])
    raw["completed_at"] = _dt(raw.get("completed_at"))
    return SessionTurn(**raw)


def _snapshot(raw: dict) -> RunSnapshot:
    raw["status"] = RunStatus(raw["status"])
    raw["resume_messages"] = tuple(raw["resume_messages"])
    raw["usage"] = RunUsage(**raw["usage"])
    raw["created_at"] = _dt(raw["created_at"])
    return RunSnapshot(**raw)


class LocalExecutionBackend:
    """Execution store for one process; construction performs no I/O."""

    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._locks = KeyedLocks()

    async def _exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    def _part(self, raw: str) -> StorageId:
        return StorageId.parse(raw)

    def _session_path(self, session_id: str) -> Path:
        sid = self._part(session_id)
        return safe_child(self.root, "execution", "sessions", sid, "session.json")

    def _turn_path(self, session_id: str, sequence: int) -> Path:
        return self._session_path(session_id).parent / "turns" / f"{sequence:020d}.json"

    def _run_dir(self, run_id: str) -> Path:
        return safe_child(self.root, "execution", "runs", self._part(run_id))

    def _run_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _snapshot_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "snapshot.json"

    def _numbered(self, run_id: str, kind: str, sequence: int) -> Path:
        return self._run_dir(run_id) / kind / f"{sequence:020d}.json"

    async def initialize_storage(self) -> None:
        await asyncio.to_thread((self.root / "execution").mkdir, parents=True, exist_ok=True)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        path = self._session_path(session_id)
        async with self._locks.acquire(("session", session_id)):
            if await self._exists(path):
                return _session(dict(await asyncio.to_thread(read_json, path)))
            now = _now()
            value = SessionRecord(session_id, user_id, tenant_id, 1, None, now, now)
            await asyncio.to_thread(atomic_write_json, path, asdict(value))
            return value

    async def get_session(self, session_id: str) -> SessionRecord | None:
        path = self._session_path(session_id)
        if not await self._exists(path):
            return None
        return _session(dict(await asyncio.to_thread(read_json, path)))

    async def start_run(self, command: StartRun) -> RunRecord:
        async with self._locks.acquire(("session", command.session_id)):
            session = await self.get_session(command.session_id)
            if session is None:
                raise StorageError("unknown session")
            if await self._exists(self._run_path(command.run_id)):
                raise StorageConflictError("run already exists")
            now = _now()
            is_root = command.kind is RunKind.USER_TURN
            sequence = session.next_turn_sequence if is_root else None
            record = RunRecord(
                command.run_id,
                command.session_id,
                command.kind,
                command.definition.runnable_id,
                command.definition.runnable_type,
                command.definition,
                RunStatus.PENDING,
                sequence,
                command.parent_run_id,
                command.root_run_id or command.run_id,
                None,
                Lease(),
                None,
                0,
                0,
                1,
                session.tenant_id,
                session.user_id,
                now,
                now,
            )
            await asyncio.to_thread(atomic_write_json, self._run_path(command.run_id), asdict(record))
            await asyncio.to_thread(atomic_write_json, self._numbered(command.run_id, "events", 1), asdict(RunEvent(command.run_id, 1, "run.started", {}, now)))
            if sequence is not None:
                turn = SessionTurn(command.session_id, sequence, command.run_id, command.user_prompt, None, RunStatus.PENDING, now, None)
                await asyncio.to_thread(atomic_write_json, self._turn_path(command.session_id, sequence), asdict(turn))
                await asyncio.to_thread(atomic_write_json, self._session_path(command.session_id), asdict(replace(session, next_turn_sequence=sequence + 1, updated_at=now)))
            return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_path(run_id)
        if not await self._exists(path):
            return None
        return _run(dict(await asyncio.to_thread(read_json, path)))

    async def claim_run(self, command: ClaimRun) -> RunRecord:
        async with self._locks.acquire(("run", command.run_id)):
            record = await self.get_run(command.run_id)
            if record is None:
                raise StorageError("unknown run")
            assert_claimable(record, command.now)
            updated = replace(record, status=RunStatus.RUNNING, lease=claim(record.lease, owner=command.owner, now=command.now, duration=command.duration), updated_at=command.now)
            await asyncio.to_thread(atomic_write_json, self._run_path(command.run_id), asdict(updated))
            return updated

    async def heartbeat_run(self, command: HeartbeatRun) -> RunRecord:
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            updated = replace(record, lease=renew(record.lease, owner=command.owner, fence=command.fence, now=command.now, duration=command.duration), updated_at=command.now)
            await asyncio.to_thread(atomic_write_json, self._run_path(command.run_id), asdict(updated))
            return updated

    async def request_cancel(self, command: RequestRunCancel) -> RunRecord:
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            assert_owner(record, command.owner, command.fence, command.requested_at)
            assert_transition(record.status, RunStatus.CANCELLING)
            updated = replace(record, status=RunStatus.CANCELLING, cancel_requested_at=command.requested_at, lease=release(record.lease), event_sequence=record.event_sequence + 1, updated_at=command.requested_at)
            await self._write_run_event(updated, "run.cancelling", command.requested_at)
            return updated

    async def pause_run(self, command: PauseRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.PAUSED, command.pending_approval)

    async def resume_run(self, command: ResumeRun) -> RunRecord:
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            assert_resumable(record)
            assert_approval_decided(record)
            updated = replace(record, status=RunStatus.PENDING, pending_approval=None, event_sequence=record.event_sequence + 1, updated_at=_now())
            await self._write_run_event(updated, "run.resumed", updated.updated_at)
            return updated

    async def decide_approval(self, command: DecideRunApproval) -> RunRecord:
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            if record.status is not RunStatus.PAUSED or record.pending_approval is None or record.pending_approval.approval_id != command.approval_id:
                raise StorageError("run has no pending approval")
            updated = replace(record, pending_approval=replace(record.pending_approval, decision=command.decision, decided_by=command.decided_by), event_sequence=record.event_sequence + 1, updated_at=_now())
            await self._write_run_event(updated, "run.approval_decided", updated.updated_at)
            return updated

    async def complete_run(self, command: CompleteRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.COMPLETED)

    async def fail_run(self, command: FailRun) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.FAILED)

    async def acknowledge_cancel(self, command: AcknowledgeRunCancel) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.CANCELLED)

    async def _finish(self, run_id: str, owner: str, fence: int, snapshot: RunSnapshot, status: RunStatus, pending_approval: RunApproval | None = None) -> RunRecord:
        record = await self._required_run(run_id)
        session_id = record.session_id
        async with self._locks.acquire(("session", session_id), ("run", run_id)):
            record = await self._required_run(run_id)
            assert_owner(record, owner, fence, _now())
            assert_transition(record.status, status)
            current_snapshot = await self.get_snapshot(run_id)
            if current_snapshot is not None and snapshot.revision <= current_snapshot.revision:
                raise StorageConflictError("snapshot revision is not increasing")
            now = snapshot.created_at
            updated = replace(record, status=status, pending_approval=pending_approval, lease=release(record.lease), snapshot_revision=snapshot.revision, trace_sequence=snapshot.trace_end_sequence, event_sequence=record.event_sequence + 1, updated_at=now)
            await asyncio.to_thread(atomic_write_json, self._snapshot_path(run_id), asdict(snapshot))
            await self._write_run_event(updated, f"run.{status.value}", now)
            if record.session_turn_sequence is not None:
                turn = _turn(dict(await asyncio.to_thread(read_json, self._turn_path(session_id, record.session_turn_sequence))))
                completed = status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
                await asyncio.to_thread(atomic_write_json, self._turn_path(session_id, record.session_turn_sequence), asdict(replace(turn, status=status, assistant_summary=snapshot.final_output, completed_at=now if completed else None)))
                if status is RunStatus.COMPLETED:
                    session = await self.get_session(session_id)
                    if session is not None:
                        await asyncio.to_thread(atomic_write_json, self._session_path(session_id), asdict(replace(session, latest_completed_run_id=run_id, updated_at=now)))
            return updated

    async def _required_run(self, run_id: str) -> RunRecord:
        record = await self.get_run(run_id)
        if record is None:
            raise StorageError(f"unknown run: {run_id}")
        return record

    async def _write_run_event(self, record: RunRecord, event_type: str, created_at: datetime) -> None:
        await asyncio.to_thread(atomic_write_json, self._run_path(record.id), asdict(record))
        await asyncio.to_thread(atomic_write_json, self._numbered(record.id, "events", record.event_sequence), asdict(RunEvent(record.id, record.event_sequence, event_type, {}, created_at)))

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int:
        async with self._locks.acquire(("run", run_id)):
            record = await self._required_run(run_id)
            if record.trace_sequence != expected_sequence:
                raise StorageConflictError("trace sequence conflict")
            sequence = expected_sequence
            for step in steps:
                sequence += 1
                await asyncio.to_thread(atomic_write_json, self._numbered(run_id, "trace", sequence), asdict(RunTraceStep(run_id, sequence, step.kind, step.payload, step.created_at)))
            updated = replace(record, trace_sequence=sequence, updated_at=_now())
            await asyncio.to_thread(atomic_write_json, self._run_path(run_id), asdict(updated))
            return sequence

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        record = await self._required_run(run_id)
        end = record.trace_sequence if through_sequence is None else min(through_sequence, record.trace_sequence)
        values = []
        for sequence in range(after_sequence + 1, end + 1):
            path = self._numbered(run_id, "trace", sequence)
            if not await self._exists(path):
                raise StorageCorruptionError(f"missing trace step: {path}")
            raw = dict(await asyncio.to_thread(read_json, path))
            raw["created_at"] = _dt(raw["created_at"])
            values.append(RunTraceStep(**raw))
        return tuple(values)

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        path = self._snapshot_path(run_id)
        return None if not await self._exists(path) else _snapshot(dict(await asyncio.to_thread(read_json, path)))

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page[SessionTurn]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        session = await self.get_session(session_id)
        if session is None:
            return Page((), False, None)
        end = session.next_turn_sequence - 1 if before_sequence is None else before_sequence - 1
        values = []
        for sequence in range(end, max(0, end - limit - 1), -1):
            path = self._turn_path(session_id, sequence)
            if not await self._exists(path):
                if sequence < session.next_turn_sequence - 1:
                    raise StorageCorruptionError(f"missing session turn: {path}")
                continue
            values.append(_turn(dict(await asyncio.to_thread(read_json, path))))
        has_more = len(values) > limit
        items = tuple(reversed(values[:limit]))
        return Page(items, has_more, values[limit - 1].sequence if has_more else None)

    async def load_session_context(self, session_id: str) -> tuple[object, ...]:
        session = await self.get_session(session_id)
        if session is None or session.latest_completed_run_id is None:
            return ()
        snapshot = await self.get_snapshot(session.latest_completed_run_id)
        return () if snapshot is None else snapshot.resume_messages

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page[RunEvent]:
        record = await self._required_run(run_id)
        values = []
        end = min(record.event_sequence, after_sequence + limit + 1)
        for sequence in range(after_sequence + 1, end + 1):
            path = self._numbered(run_id, "events", sequence)
            if not await self._exists(path):
                raise StorageCorruptionError(f"missing event: {path}")
            raw = dict(await asyncio.to_thread(read_json, path))
            raw["created_at"] = _dt(raw["created_at"])
            values.append(RunEvent(**raw))
        return Page(tuple(values[:limit]), len(values) > limit, values[limit - 1].sequence if len(values) > limit else None)

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        await asyncio.to_thread(atomic_write_json, self._numbered(evaluation.run_id, "evaluations", 0).with_name(f"{StorageId.parse(evaluation.evaluation_id).value}.json"), asdict(evaluation))

    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]:
        directory = self._run_dir(run_id) / "evaluations"
        names = await asyncio.to_thread(lambda: tuple(directory.iterdir()) if directory.exists() else ())
        values = []
        for path in names:
            raw = dict(await asyncio.to_thread(read_json, path))
            raw["created_at"] = _dt(raw["created_at"])
            values.append(RunEvaluation(**raw))
        return tuple(sorted(values, key=lambda item: item.created_at))


__all__ = ["LocalExecutionBackend", "StorageCorruptionError"]
