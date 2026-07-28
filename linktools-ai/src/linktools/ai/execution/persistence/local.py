"""Single-process JSON-file ExecutionStore.

The constructor performs no filesystem I/O. This backend deliberately does not
provide crash durability or cross-process coordination.
"""

import asyncio
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...errors import StorageError
from ..codec import decode, encode
from ..models import (
    Page,
    RunDefinitionSnapshot,
    RunApproval,
    RunEvaluation,
    RunEvent,
    RunKind,
    RunRecord,
    RunSnapshot,
    RunStatus,
    RunTraceStep,
    RunUsage,
    SessionRecord,
    SessionTurn,
)


class StorageCorruptionError(StorageError):
    """A required numbered execution file is missing or malformed."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encode(value))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path) -> Any:
    try:
        return decode(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageCorruptionError(f"missing execution file: {path}") from exc


def _session(raw: dict[str, Any]) -> SessionRecord:
    return SessionRecord(**raw)


def _turn(raw: dict[str, Any]) -> SessionTurn:
    raw["status"] = RunStatus(raw["status"])
    return SessionTurn(**raw)


def _run(raw: dict[str, Any]) -> RunRecord:
    raw["kind"] = RunKind(raw["kind"])
    raw["status"] = RunStatus(raw["status"])
    raw["definition"] = RunDefinitionSnapshot(**raw["definition"])
    if raw.get("pending_approval") is not None:
        raw["pending_approval"] = RunApproval(**raw["pending_approval"])
    return RunRecord(**raw)


def _snapshot(raw: dict[str, Any]) -> RunSnapshot:
    raw["status"] = RunStatus(raw["status"])
    raw["resume_messages"] = tuple(raw["resume_messages"])
    raw["usage"] = RunUsage(**raw["usage"])
    return RunSnapshot(**raw)


class LocalExecutionStore:
    """An aggregate-locked, single-process ExecutionStore."""

    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock(self, kind: str, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault((kind, key), asyncio.Lock())

    def _session_path(self, session_id: str) -> Path:
        return self.root / "execution" / "sessions" / session_id / "session.json"

    def _turn_path(self, session_id: str, sequence: int) -> Path:
        return self._session_path(session_id).parent / "turns" / f"{sequence:020d}.json"

    def _run_dir(self, run_id: str) -> Path:
        return self.root / "execution" / "runs" / run_id

    def _event_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / "events" / f"{sequence:020d}.json"

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        lock = await self._lock("session", session_id)
        async with lock:
            if self._session_path(session_id).exists():
                return _session(await asyncio.to_thread(_read, self._session_path(session_id)))
            now = _now()
            record = SessionRecord(session_id, user_id, tenant_id, 1, None, now, now)
            await asyncio.to_thread(_write, self._session_path(session_id), asdict(record))
            return record

    async def get_session(self, session_id: str) -> SessionRecord | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        return _session(await asyncio.to_thread(_read, path))

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        session = await self.get_session(session_id)
        if session is None:
            return Page((), False, None)
        end = (before_sequence - 1) if before_sequence is not None else session.next_turn_sequence - 1
        sequences = list(range(end, max(0, end - limit - 1), -1))
        values: list[SessionTurn] = []
        for sequence in sequences:
            path = self._turn_path(session_id, sequence)
            if not path.exists():
                if sequence < session.next_turn_sequence - 1:
                    raise StorageCorruptionError(f"missing session turn: {path}")
                continue
            values.append(_turn(await asyncio.to_thread(_read, path)))
        has_more = len(values) > limit
        page_values = list(reversed(values[:limit]))
        return Page(tuple(page_values), has_more, values[limit - 1].sequence if has_more else None)

    async def load_session_context(self, session_id: str) -> tuple[Any, ...]:
        session = await self.get_session(session_id)
        if session is None or session.latest_completed_run_id is None:
            return ()
        snapshot = await self.get_snapshot(session.latest_completed_run_id)
        return () if snapshot is None else snapshot.resume_messages

    async def get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_dir(run_id) / "run.json"
        if not path.exists():
            return None
        return _run(await asyncio.to_thread(_read, path))

    async def start_run(self, **kwargs: Any) -> RunRecord:
        session_id = kwargs["session_id"]
        session_lock = await self._lock("session", session_id)
        async with session_lock:
            session = await self.get_session(session_id)
            if session is None:
                raise StorageError(f"unknown session: {session_id}")
            now = _now()
            kind = kwargs.get("kind", RunKind.USER_TURN)
            kind = RunKind(kind)
            sequence = session.next_turn_sequence if kind == RunKind.USER_TURN else None
            run_id = kwargs["run_id"]
            record = RunRecord(run_id, session_id, kind, sequence, kwargs.get("parent_run_id"), kwargs.get("root_run_id", run_id), RunStatus.RUNNING, kwargs["definition"], None, None, 0, None, None, 0, 0, now, now, session.tenant_id, session.user_id)
            await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(record))
            await asyncio.to_thread(_write, self._event_path(run_id, 1), asdict(RunEvent(run_id, 1, "run.started", {}, now)))
            if sequence is not None:
                turn = SessionTurn(session_id, sequence, run_id, kwargs.get("user_prompt", ""), None, RunStatus.RUNNING, now, None)
                await asyncio.to_thread(_write, self._turn_path(session_id, sequence), asdict(turn))
                session = SessionRecord(session.id, session.user_id, session.tenant_id, sequence + 1, session.latest_completed_run_id, session.created_at, now)
                await asyncio.to_thread(_write, self._session_path(session_id), asdict(session))
            return record

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[Any, ...]) -> int:
        lock = await self._lock("run", run_id)
        async with lock:
            record = await self.get_run(run_id)
            if record is None or record.trace_sequence != expected_sequence:
                raise StorageError("trace sequence conflict")
            for offset, step in enumerate(steps, 1):
                value = RunTraceStep(run_id, expected_sequence + offset, step.kind, step.payload, step.created_at)
                await asyncio.to_thread(_write, self._run_dir(run_id) / "trace" / f"{value.sequence:020d}.json", asdict(value))
            updated = RunRecord(**{**asdict(record), "kind": record.kind, "status": record.status, "definition": record.definition, "pending_approval": record.pending_approval, "trace_sequence": expected_sequence + len(steps), "updated_at": _now()})
            await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(updated))
            return updated.trace_sequence

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        record = await self.get_run(run_id)
        if record is None:
            return ()
        end = through_sequence if through_sequence is not None else record.trace_sequence
        values = []
        for sequence in range(after_sequence + 1, end + 1):
            raw = await asyncio.to_thread(_read, self._run_dir(run_id) / "trace" / f"{sequence:020d}.json")
            values.append(RunTraceStep(**raw))
        return tuple(values)

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        path = self._run_dir(run_id) / "snapshot.json"
        return None if not path.exists() else _snapshot(await asyncio.to_thread(_read, path))

    async def _finish(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot, status: RunStatus, pending_approval: Any = None) -> RunRecord:
        lock = await self._lock("run", run_id)
        async with lock:
            record = await self.get_run(run_id)
            if record is None or record.execution_owner != owner or record.execution_fence != fence:
                raise StorageError("run ownership conflict")
            if record.status is not RunStatus.RUNNING:
                raise StorageError(f"invalid lifecycle transition from {record.status.value}")
            await asyncio.to_thread(_write, self._run_dir(run_id) / "snapshot.json", asdict(snapshot))
            updated = RunRecord(**{**asdict(record), "kind": record.kind, "status": status, "definition": record.definition, "pending_approval": pending_approval if pending_approval is not None else record.pending_approval, "snapshot_revision": snapshot.revision, "trace_sequence": snapshot.trace_end_sequence, "updated_at": _now()})
            await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(updated))
            event_dir = self._run_dir(run_id) / "events"
            existing_events = await asyncio.to_thread(lambda: tuple(event_dir.glob("*.json")))
            event_sequence = max((int(path.stem) for path in existing_events), default=0) + 1
            await asyncio.to_thread(_write, self._event_path(run_id, event_sequence), asdict(RunEvent(run_id, event_sequence, f"run.{status.value}", {}, _now())))
            if record.session_turn_sequence is not None:
                turn_path = self._turn_path(record.session_id, record.session_turn_sequence)
                turn = _turn(await asyncio.to_thread(_read, turn_path))
                turn = SessionTurn(**{**asdict(turn), "status": status, "completed_at": _now(), "assistant_summary": snapshot.final_output})
                await asyncio.to_thread(_write, turn_path, asdict(turn))
                if status == RunStatus.COMPLETED:
                    session = await self.get_session(record.session_id)
                    if session:
                        await asyncio.to_thread(_write, self._session_path(record.session_id), asdict(SessionRecord(**{**asdict(session), "latest_completed_run_id": run_id, "updated_at": _now()})))
            return updated

    async def complete_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.COMPLETED)

    async def fail_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.FAILED)

    async def acknowledge_cancel(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.CANCELLED)

    async def claim_run(self, run_id: str, *, owner: str, expected_fence: int | None = None) -> RunRecord:
        lock = await self._lock("run", run_id)
        async with lock:
            record = await self.get_run(run_id)
            if record is None:
                raise StorageError(f"unknown run: {run_id}")
            if record.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                raise StorageError(f"cannot claim run in {record.status.value} state")
            fence = record.execution_fence + 1
            if expected_fence is not None and record.execution_fence != expected_fence:
                raise StorageError("run fence conflict")
            updated = RunRecord(**{**asdict(record), "kind": record.kind, "status": RunStatus.RUNNING, "definition": record.definition, "pending_approval": record.pending_approval, "execution_owner": owner, "execution_fence": fence, "updated_at": _now()})
            await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(updated))
            return updated

    async def heartbeat_run(self, run_id: str, *, owner: str, fence: int) -> RunRecord:
        record = await self.get_run(run_id)
        if record is None or record.execution_owner != owner or record.execution_fence != fence:
            raise StorageError("run ownership conflict")
        return record

    async def request_cancel(self, run_id: str, *, owner: str, fence: int) -> RunRecord:
        record = await self.get_run(run_id)
        if record is None or record.execution_owner != owner or record.execution_fence != fence:
            raise StorageError("run ownership conflict")
        if record.status not in (RunStatus.RUNNING, RunStatus.PAUSED):
            raise StorageError(f"cannot cancel run in {record.status.value} state")
        updated = RunRecord(**{**asdict(record), "kind": record.kind, "status": record.status, "definition": record.definition, "pending_approval": record.pending_approval, "cancel_requested_at": _now(), "updated_at": _now()})
        await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(updated))
        return updated

    async def pause_run(self, run_id: str, *, owner: str, fence: int, snapshot: RunSnapshot, pending_approval: object | None = None) -> RunRecord:
        return await self._finish(run_id, owner=owner, fence=fence, snapshot=snapshot, status=RunStatus.PAUSED, pending_approval=pending_approval)

    async def decide_approval(self, run_id: str, *, approval_id: str, decision: str, decided_by: str) -> RunRecord:
        lock = await self._lock("run", run_id)
        async with lock:
            record = await self.get_run(run_id)
            approval = record.pending_approval if record is not None else None
            if record is None or record.status is not RunStatus.PAUSED or approval is None or approval.approval_id != approval_id:
                raise StorageError("run has no pending approval")
            updated_approval = RunApproval(approval.approval_id, approval.tool_call_id, approval.tool_name, approval.arguments, decision, decided_by)
            updated = RunRecord(**{**asdict(record), "kind": record.kind, "status": record.status, "definition": record.definition, "pending_approval": updated_approval, "updated_at": _now()})
            await asyncio.to_thread(_write, self._run_dir(run_id) / "run.json", asdict(updated))
            event_dir = self._run_dir(run_id) / "events"
            existing_events = await asyncio.to_thread(lambda: tuple(event_dir.glob("*.json")))
            event_sequence = max((int(path.stem) for path in existing_events), default=0) + 1
            await asyncio.to_thread(_write, self._event_path(run_id, event_sequence), asdict(RunEvent(run_id, event_sequence, "run.approval_decided", {"decision": decision}, _now())))
            return updated

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        paths = [self._run_dir(run_id) / "events" / f"{sequence:020d}.json" for sequence in range(after_sequence + 1, after_sequence + limit + 2)]
        events = tuple(RunEvent(**(await asyncio.to_thread(_read, path))) for path in paths if path.exists())
        has_more = len(events) > limit
        return Page(events[:limit], has_more, events[limit - 1].sequence if has_more else None)

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        stamp = int(evaluation.created_at.timestamp() * 1_000_000_000)
        await asyncio.to_thread(_write, self._run_dir(evaluation.run_id) / "evaluations" / f"{evaluation.evaluator}-{stamp}.json", asdict(evaluation))

    async def list_evaluations(self, run_id: str) -> tuple[RunEvaluation, ...]:
        directory = self._run_dir(run_id) / "evaluations"
        if not directory.exists():
            return ()
        paths = await asyncio.to_thread(lambda: tuple(directory.iterdir()))
        return tuple(RunEvaluation(**(await asyncio.to_thread(_read, path))) for path in paths)
