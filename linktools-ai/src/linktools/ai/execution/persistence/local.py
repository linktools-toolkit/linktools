"""Single-process JSON-file execution persistence."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from ...storage.coordination.lease import Lease, assert_active, claim, release, renew
from ...storage.database import CoordinationScope
from ...errors import StorageConflictError, StorageCorruptionError, StorageError
from ...storage.local.files import (
    atomic_write_bytes,
    atomic_write_json,
    read_bytes,
    read_json,
)
from ...storage.local.locks import KeyedLocks
from ...storage.local.paths import StorageId, safe_child
from ..commands import (
    AbortExecution,
    AcknowledgeCancellation,
    ClaimExecution,
    CompleteExecution,
    DecideApproval,
    FailExecution,
    HeartbeatExecution,
    PauseExecution,
    RequestCancellation,
    ResumeExecution,
    StartExecution,
)
from ..lifecycle import assert_approval_decided, assert_claimable, assert_owner, assert_resumable, assert_transition
from ..domain import ApprovalDecision, RunApproval, RunDefinition, RunError, RunKind, RunRecord, RunStatus, RunUsage, RunnableType
from ..domain import Page
from ..evaluation import RunEvaluation
from ..session import SessionRecord, SessionTurn
from ..snapshots import AgentSnapshotData, RunSnapshot
from ..trace_models import NewRunTraceStep, RunEvent, RunTraceStep


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
    if raw.get("approval") is not None:
        approval = raw["approval"]
        if approval.get("decision") is not None:
            approval["decision"] = ApprovalDecision(approval["decision"])
        approval["decided_at"] = _dt(approval.get("decided_at"))
        raw["approval"] = RunApproval(**approval)
    if raw.get("error") is not None:
        raw["error"] = RunError(**raw["error"])
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

    coordination_scope = CoordinationScope.PROCESS

    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._locks = KeyedLocks()
        self._recovery_locks = KeyedLocks()

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

    def _journal_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "journal"

    def _session_journal_dir(self, session_id: str) -> Path:
        return self._session_path(session_id).parent / "journal"

    @staticmethod
    def _same_json(path: Path, value: object) -> bool:
        try:
            return path.exists() and read_json(path) == value
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _remove(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _cleanup_journal(self, journal_path: Path, journal: dict) -> None:
        transaction_dirs: set[Path] = set()
        for entry in journal["entries"]:
            staged = Path(entry["staged"])
            transaction_dirs.add(staged.parent)
            self._remove(staged)
            backup = entry.get("backup")
            if backup is not None:
                self._remove(Path(backup))
        for path in journal.get("journal_paths", (str(journal_path),)):
            self._remove(Path(path))
        for directory in transaction_dirs:
            try:
                directory.rmdir()
            except (FileNotFoundError, OSError):
                pass

    def _recover_journal(self, journal_path: Path) -> None:
        try:
            journal = dict(read_json(journal_path))
        except (OSError, ValueError, TypeError) as error:
            raise StorageCorruptionError(
                f"invalid execution journal: {journal_path}"
            ) from error
        entries = list(journal.get("entries", ()))
        if journal.get("state") == "PUBLISHED":
            self._cleanup_journal(journal_path, journal)
            return
        if journal.get("state") != "PREPARED" or not entries:
            raise StorageCorruptionError(
                f"invalid execution journal state: {journal_path}"
            )
        manifests = [entry for entry in entries if entry.get("manifest")]
        manifest_published = any(
            self._same_json(Path(entry["target"]), entry["value"])
            for entry in manifests
        )
        if not manifest_published:
            for entry in reversed(entries):
                target = Path(entry["target"])
                backup = entry.get("backup")
                if backup is None:
                    self._remove(target)
                else:
                    backup_path = Path(backup)
                    if not backup_path.exists():
                        raise StorageCorruptionError(
                            f"missing journal backup: {backup_path}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup_path, target)
            self._cleanup_journal(journal_path, journal)
            return
        for entry in entries:
            target = Path(entry["target"])
            if self._same_json(target, entry["value"]):
                continue
            staged = Path(entry["staged"])
            if not staged.exists():
                raise StorageCorruptionError(
                    f"published manifest references missing member: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
        journal["state"] = "PUBLISHED"
        atomic_write_json(journal_path, journal)
        self._cleanup_journal(journal_path, journal)

    async def _recover_directory(self, directory: Path) -> None:
        paths = await asyncio.to_thread(
            lambda: tuple(sorted(directory.glob("*.json")))
            if directory.exists()
            else ()
        )
        for path in paths:
            async with self._recovery_locks.acquire(
                ("journal", path.stem)
            ):
                if await self._exists(path):
                    await asyncio.to_thread(self._recover_journal, path)

    async def _recover_run(self, run_id: str) -> None:
        await self._recover_directory(self._journal_dir(run_id))

    async def _run_journal_sessions(self, run_id: str) -> tuple[str, ...]:
        directory = self._journal_dir(run_id)
        paths = await asyncio.to_thread(
            lambda: tuple(directory.glob("*.json"))
            if directory.exists()
            else ()
        )
        session_ids: set[str] = set()
        for path in paths:
            try:
                journal = dict(await asyncio.to_thread(read_json, path))
            except (OSError, ValueError, TypeError):
                continue
            for entry in journal.get("entries", ()):
                parts = Path(entry["target"]).parts
                if "sessions" in parts:
                    session_ids.add(parts[parts.index("sessions") + 1])
        return tuple(sorted(session_ids))

    async def _recover_session(self, session_id: str) -> None:
        await self._recover_directory(
            self._session_journal_dir(session_id)
        )

    async def _session_journal_runs(
        self, session_id: str
    ) -> tuple[str, ...]:
        directory = self._session_journal_dir(session_id)
        paths = await asyncio.to_thread(
            lambda: tuple(directory.glob("*.json"))
            if directory.exists()
            else ()
        )
        run_ids: set[str] = set()
        for path in paths:
            try:
                journal = dict(await asyncio.to_thread(read_json, path))
            except (OSError, ValueError, TypeError):
                continue
            for entry in journal.get("entries", ()):
                parts = Path(entry["target"]).parts
                if "runs" in parts:
                    run_ids.add(parts[parts.index("runs") + 1])
        return tuple(sorted(run_ids))

    def _commit_files(
        self,
        run_id: str,
        writes: tuple[tuple[Path, object, bool], ...],
    ) -> None:
        transaction_id = uuid4().hex
        journal_dir = self._journal_dir(run_id)
        transaction_dir = journal_dir / transaction_id
        journal_paths = [journal_dir / f"{transaction_id}.json"]
        entries: list[dict] = []
        transaction_dir.mkdir(parents=True, exist_ok=False)
        try:
            for index, (target, value, manifest) in enumerate(writes):
                staged = transaction_dir / f"{index:04d}.staged.json"
                backup = (
                    transaction_dir / f"{index:04d}.backup.json"
                    if target.exists()
                    else None
                )
                atomic_write_json(staged, value)
                if backup is not None:
                    atomic_write_bytes(backup, read_bytes(target))
                entries.append(
                    {
                        "target": str(target),
                        "staged": str(staged),
                        "backup": str(backup) if backup is not None else None,
                        "manifest": manifest,
                        "value": value,
                    }
                )
            journal = {
                "version": "execution-journal.v1",
                "state": "PREPARED",
                "entries": entries,
            }
            session_ids = {
                Path(entry["target"]).parts[
                    Path(entry["target"]).parts.index("sessions") + 1
                ]
                for entry in entries
                if "sessions" in Path(entry["target"]).parts
            }
            journal_paths.extend(
                self._session_journal_dir(session_id)
                / f"{transaction_id}.json"
                for session_id in sorted(session_ids)
            )
            journal["journal_paths"] = [
                str(path) for path in journal_paths
            ]
            for journal_path in journal_paths:
                atomic_write_json(journal_path, journal)
            for entry in sorted(entries, key=lambda item: bool(item["manifest"])):
                target = Path(entry["target"])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(entry["staged"], target)
            journal["state"] = "PUBLISHED"
            for journal_path in journal_paths:
                atomic_write_json(journal_path, journal)
            self._cleanup_journal(journal_paths[0], journal)
        except BaseException:
            # Once PREPARED exists, recovery decides whether to roll back or
            # finish publication. Before that point no target was replaced.
            if not any(path.exists() for path in journal_paths):
                for entry in entries:
                    self._remove(Path(entry["staged"]))
                    backup = entry.get("backup")
                    if backup is not None:
                        self._remove(Path(backup))
                try:
                    transaction_dir.rmdir()
                except OSError:
                    pass
            raise

    async def initialize_storage(self) -> None:
        await asyncio.to_thread((self.root / "execution").mkdir, parents=True, exist_ok=True)

    async def create_session(self, *, session_id: str, user_id: str | None, tenant_id: str | None) -> SessionRecord:
        path = self._session_path(session_id)
        async with self._locks.acquire(("session", session_id)):
            if await self._exists(path):
                existing = _session(
                    dict(await asyncio.to_thread(read_json, path))
                )
                if (
                    existing.user_id != user_id
                    or existing.tenant_id != tenant_id
                ):
                    raise StorageConflictError("session ownership conflict")
                return existing
            now = _now()
            value = SessionRecord(session_id, user_id, tenant_id, 1, None, now, now)
            await asyncio.to_thread(atomic_write_json, path, asdict(value))
            return value

    async def get_session(self, session_id: str) -> SessionRecord | None:
        run_ids = await self._session_journal_runs(session_id)
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in run_ids)),
        ):
            await self._recover_session(session_id)
            return await self._read_session(session_id)

    async def _read_session(self, session_id: str) -> SessionRecord | None:
        path = self._session_path(session_id)
        if not await self._exists(path):
            return None
        return _session(dict(await asyncio.to_thread(read_json, path)))

    async def start_run(self, command: StartExecution) -> RunRecord:
        journal_runs = await self._session_journal_runs(command.session_id)
        async with self._locks.acquire(
            ("session", command.session_id),
            *((("run", run_id) for run_id in journal_runs)),
            ("run", command.run_id),
        ):
            await self._recover_session(command.session_id)
            await self._recover_run(command.run_id)
            session = await self._read_session(command.session_id)
            if session is None:
                raise StorageError("unknown session")
            if await self._exists(self._run_path(command.run_id)):
                raise StorageConflictError("run already exists")
            now = _now()
            is_root = command.kind is RunKind.USER_TURN
            sequence = session.next_turn_sequence if is_root else None
            record = RunRecord(
                id=command.run_id,
                session_id=command.session_id,
                kind=command.kind,
                runnable_id=command.definition.runnable_id,
                runnable_type=command.definition.runnable_type,
                definition=command.definition,
                status=RunStatus.PENDING,
                session_turn_sequence=sequence,
                parent_execution_id=command.parent_execution_id,
                root_execution_id=command.root_execution_id or command.run_id,
                approval=None,
                lease=Lease(),
                cancel_requested_at=None,
                snapshot_revision=0,
                trace_sequence=0,
                event_sequence=1,
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                created_at=now,
                updated_at=now,
                error=None,
                input=command.input,
            )
            writes: list[tuple[Path, object, bool]] = [
                (
                    self._numbered(command.run_id, "events", 1),
                    asdict(RunEvent(command.run_id, 1, "run.started", {}, now)),
                    False,
                ),
            ]
            if sequence is not None:
                turn = SessionTurn(command.session_id, sequence, command.run_id, command.input, None, RunStatus.PENDING, now, None)
                writes.append(
                    (self._turn_path(command.session_id, sequence), asdict(turn), False)
                )
            writes.append((self._run_path(command.run_id), asdict(record), True))
            if sequence is not None:
                writes.append(
                    (
                        self._session_path(command.session_id),
                        asdict(
                            replace(
                                session,
                                next_turn_sequence=sequence + 1,
                                updated_at=now,
                            )
                        ),
                        True,
                    )
                )
            await asyncio.to_thread(
                self._commit_files, command.run_id, tuple(writes)
            )
            return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        sessions = await self._run_journal_sessions(run_id)
        async with self._locks.acquire(
            *((("session", session_id) for session_id in sessions)),
            ("run", run_id),
        ):
            await self._recover_run(run_id)
            return await self._read_run(run_id)

    async def _read_run(self, run_id: str) -> RunRecord | None:
        path = self._run_path(run_id)
        if not await self._exists(path):
            return None
        return _run(dict(await asyncio.to_thread(read_json, path)))

    async def claim_run(self, command: ClaimExecution) -> RunRecord:
        await self.get_run(command.run_id)
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._read_run(command.run_id)
            if record is None:
                raise StorageError("unknown run")
            assert_claimable(record, command.now)
            updated = replace(record, status=RunStatus.RUNNING, lease=claim(record.lease, owner=command.owner, now=command.now, duration=command.duration), updated_at=command.now)
            await asyncio.to_thread(atomic_write_json, self._run_path(command.run_id), asdict(updated))
            return updated

    async def heartbeat_run(self, command: HeartbeatExecution) -> RunRecord:
        await self.get_run(command.run_id)
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            updated = replace(record, lease=renew(record.lease, owner=command.owner, fence=command.fence, now=command.now, duration=command.duration), updated_at=command.now)
            await asyncio.to_thread(atomic_write_json, self._run_path(command.run_id), asdict(updated))
            return updated

    async def request_cancel(self, command: RequestCancellation) -> RunRecord:
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(
            ("session", current.session_id), ("run", command.run_id)
        ):
            record = await self._required_run(command.run_id)
            if record.status in {RunStatus.PENDING, RunStatus.PAUSED}:
                # Direct terminal cancel: PENDING was never claimed (no lease),
                # PAUSED lease was already released by _finish. No owner/fence
                # check needed.
                now = command.requested_at
                updated = replace(record, status=RunStatus.CANCELLED, lease=release(record.lease), cancel_requested_at=now, event_sequence=record.event_sequence + 1, updated_at=now)
                additional: tuple[tuple[Path, object], ...] = ()
                if record.session_turn_sequence is not None:
                    turn = _turn(dict(await asyncio.to_thread(read_json, self._turn_path(record.session_id, record.session_turn_sequence))))
                    additional = (
                        (
                            self._turn_path(
                                record.session_id,
                                record.session_turn_sequence,
                            ),
                            asdict(
                                replace(
                                    turn,
                                    status=RunStatus.CANCELLED,
                                    completed_at=now,
                                )
                            ),
                        ),
                    )
                await self._write_run_event(
                    updated,
                    "run.cancelled",
                    now,
                    additional_immutable=additional,
                )
                return updated
            assert_owner(record, command.owner, command.fence, command.requested_at)
            assert_transition(record.status, RunStatus.CANCELLING)
            # The lease stays active (unlike pause/complete/fail): the same
            # owner+fence must still be able to authorize the terminal
            # acknowledge_cancel/fail_run that follows. A worker that dies
            # mid-cancel is recovered the same way any other stuck RUNNING
            # lease is -- by expiry -- not by releasing it early.
            updated = replace(record, status=RunStatus.CANCELLING, cancel_requested_at=command.requested_at, event_sequence=record.event_sequence + 1, updated_at=command.requested_at)
            await self._write_run_event(updated, "run.cancelling", command.requested_at)
            return updated

    async def pause_run(self, command: PauseExecution) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.PAUSED, command.pending_approval)

    async def resume_run(self, command: ResumeExecution) -> RunRecord:
        await self.get_run(command.run_id)
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            assert_resumable(record)
            assert_approval_decided(record)
            updated = replace(record, status=RunStatus.PENDING, event_sequence=record.event_sequence + 1, updated_at=_now())
            await self._write_run_event(updated, "run.resumed", updated.updated_at)
            return updated

    async def decide_approval(self, command: DecideApproval) -> RunRecord:
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(
            ("session", current.session_id), ("run", command.run_id)
        ):
            record = await self._required_run(command.run_id)
            if record.approval is None or record.approval.approval_id != command.approval_id:
                raise StorageError("run has no pending approval")
            existing = record.approval.decision
            if existing is not None:
                if existing == command.decision and record.approval.decided_by == command.decided_by:
                    return record
                raise StorageConflictError("approval decision conflict")
            if record.status is not RunStatus.PAUSED:
                raise StorageError("run has no pending approval")
            decided_at = _now()
            new_approval = replace(
                record.approval,
                decision=command.decision,
                decided_by=command.decided_by,
                decided_at=decided_at,
            )
            if command.decision == ApprovalDecision.ALLOW:
                updated = replace(record, approval=new_approval, event_sequence=record.event_sequence + 1, updated_at=decided_at)
                await self._write_run_event(
                    updated,
                    "run.approval_decided",
                    updated.updated_at,
                    payload=asdict(new_approval),
                )
                return updated
            now = decided_at
            updated = replace(record, status=RunStatus.CANCELLED, approval=new_approval, lease=release(record.lease), event_sequence=record.event_sequence + 1, updated_at=now)
            additional = ()
            if record.session_turn_sequence is not None:
                turn = _turn(dict(await asyncio.to_thread(read_json, self._turn_path(record.session_id, record.session_turn_sequence))))
                additional = (
                    (
                        self._turn_path(
                            record.session_id, record.session_turn_sequence
                        ),
                        asdict(
                            replace(
                                turn,
                                status=RunStatus.CANCELLED,
                                completed_at=now,
                            )
                        ),
                    ),
                )
            await self._write_run_event(
                updated,
                "run.approval_decided",
                now,
                payload=asdict(new_approval),
                additional_immutable=additional,
            )
            return updated

    async def complete_run(self, command: CompleteExecution) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.COMPLETED)

    async def fail_run(self, command: FailExecution) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.FAILED, error=command.error)

    async def acknowledge_cancel(self, command: AcknowledgeCancellation) -> RunRecord:
        return await self._finish(command.run_id, command.owner, command.fence, command.snapshot, RunStatus.CANCELLED)

    async def abort_run(self, command: AbortExecution) -> RunRecord:
        # Unlike fail_run, there is no snapshot to persist here -- an
        # AbortExecution fires on a programming/config/protocol error, before
        # the engine ever produced a coherent outcome to snapshot.
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(
            ("session", current.session_id), ("run", command.run_id)
        ):
            record = await self._required_run(command.run_id)
            assert_owner(record, command.owner, command.fence, _now())
            assert_transition(record.status, RunStatus.FAILED)
            now = _now()
            updated = replace(record, status=RunStatus.FAILED, error=command.error, lease=release(record.lease), trace_sequence=command.trace_end_sequence, event_sequence=record.event_sequence + 1, updated_at=now)
            additional = ()
            if record.session_turn_sequence is not None:
                turn = _turn(dict(await asyncio.to_thread(read_json, self._turn_path(record.session_id, record.session_turn_sequence))))
                additional = (
                    (
                        self._turn_path(
                            record.session_id, record.session_turn_sequence
                        ),
                        asdict(
                            replace(
                                turn,
                                status=RunStatus.FAILED,
                                completed_at=now,
                            )
                        ),
                    ),
                )
            await self._write_run_event(
                updated,
                "run.aborted",
                now,
                additional_immutable=additional,
            )
            return updated

    async def _finish(self, run_id: str, owner: str, fence: int, snapshot: AgentSnapshotData, status: RunStatus, pending_approval: RunApproval | None = None, error: RunError | None = None) -> RunRecord:
        record = await self.get_run(run_id)
        if record is None:
            raise StorageError(f"unknown run: {run_id}")
        session_id = record.session_id
        async with self._locks.acquire(("session", session_id), ("run", run_id)):
            record = await self._required_run(run_id)
            assert_owner(record, owner, fence, _now())
            assert_transition(record.status, status)
            # The store allocates the snapshot revision (expected + 1); the
            # write lock serializes concurrent commits within this process.
            new_revision = record.snapshot_revision + 1
            now = _now()
            stored_snapshot = RunSnapshot("run-snapshot.v1", run_id, new_revision, snapshot.resume_messages, snapshot.final_output, status, snapshot.usage, snapshot.trace_end_sequence, now)
            updated = replace(
                record,
                status=status,
                approval=(
                    pending_approval
                    if pending_approval is not None
                    else record.approval
                ),
                error=error,
                lease=release(record.lease),
                snapshot_revision=new_revision,
                trace_sequence=snapshot.trace_end_sequence,
                event_sequence=record.event_sequence + 1,
                updated_at=now,
            )
            writes: list[tuple[Path, object, bool]] = [
                (self._snapshot_path(run_id), asdict(stored_snapshot), False),
                (
                    self._numbered(run_id, "events", updated.event_sequence),
                    asdict(
                        RunEvent(
                            run_id,
                            updated.event_sequence,
                            f"run.{status.value}",
                            {},
                            now,
                        )
                    ),
                    False,
                ),
            ]
            if record.session_turn_sequence is not None:
                turn = _turn(dict(await asyncio.to_thread(read_json, self._turn_path(session_id, record.session_turn_sequence))))
                completed = status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
                writes.append(
                    (
                        self._turn_path(session_id, record.session_turn_sequence),
                        asdict(
                            replace(
                                turn,
                                status=status,
                                assistant_summary=snapshot.final_output,
                                completed_at=now if completed else None,
                            )
                        ),
                        False,
                    )
                )
                if status is RunStatus.COMPLETED:
                    session = await self._read_session(session_id)
                    if session is not None:
                        writes.append(
                            (
                                self._session_path(session_id),
                                asdict(
                                    replace(
                                        session,
                                        latest_completed_run_id=run_id,
                                        updated_at=now,
                                    )
                                ),
                                True,
                            )
                        )
            writes.append((self._run_path(run_id), asdict(updated), True))
            # Execution is the first manifest publication point; session is
            # last when present.
            manifests = [item for item in writes if item[2]]
            immutable = [item for item in writes if not item[2]]
            manifests.sort(key=lambda item: item[0] == self._session_path(session_id))
            await asyncio.to_thread(
                self._commit_files,
                run_id,
                tuple((*immutable, *manifests)),
            )
            return updated

    async def _required_run(self, run_id: str) -> RunRecord:
        record = await self._read_run(run_id)
        if record is None:
            raise StorageError(f"unknown run: {run_id}")
        return record

    async def _write_run_event(
        self,
        record: RunRecord,
        event_type: str,
        created_at: datetime,
        *,
        payload: dict | None = None,
        additional_immutable: tuple[tuple[Path, object], ...] = (),
    ) -> None:
        immutable = tuple(
            (path, value, False) for path, value in additional_immutable
        )
        await asyncio.to_thread(
            self._commit_files,
            record.id,
            (
                (
                    self._numbered(
                        record.id, "events", record.event_sequence
                    ),
                    asdict(
                        RunEvent(
                            record.id,
                            record.event_sequence,
                            event_type,
                            payload or {},
                            created_at,
                        )
                    ),
                    False,
                ),
                *immutable,
                (self._run_path(record.id), asdict(record), True),
            ),
        )

    async def append_trace_steps(self, run_id: str, *, expected_sequence: int, steps: tuple[NewRunTraceStep, ...]) -> int:
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
            record = await self._required_run(run_id)
            if record.trace_sequence != expected_sequence:
                raise StorageConflictError("trace sequence conflict")
            sequence = expected_sequence
            writes: list[tuple[Path, object, bool]] = []
            for step in steps:
                sequence += 1
                writes.append(
                    (
                        self._numbered(run_id, "trace", sequence),
                        asdict(
                            RunTraceStep(
                                run_id,
                                sequence,
                                step.kind,
                                step.payload,
                                step.created_at,
                            )
                        ),
                        False,
                    )
                )
            updated = replace(record, trace_sequence=sequence, updated_at=_now())
            writes.append((self._run_path(run_id), asdict(updated), True))
            await asyncio.to_thread(self._commit_files, run_id, tuple(writes))
            return sequence

    async def list_trace_steps(self, run_id: str, *, after_sequence: int = 0, through_sequence: int | None = None) -> tuple[RunTraceStep, ...]:
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
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
        sessions = await self._run_journal_sessions(run_id)
        async with self._locks.acquire(
            *((("session", session_id) for session_id in sessions)),
            ("run", run_id),
        ):
            await self._recover_run(run_id)
            record = await self._read_run(run_id)
            path = self._snapshot_path(run_id)
            exists = await self._exists(path)
            if (
                record is not None
                and record.snapshot_revision > 0
                and not exists
            ):
                raise StorageCorruptionError(
                    f"missing snapshot declared by run manifest: {path}"
                )
            return None if not exists else _snapshot(
                dict(await asyncio.to_thread(read_json, path))
            )

    async def list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page[SessionTurn]:
        run_ids = await self._session_journal_runs(session_id)
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in run_ids)),
        ):
            await self._recover_session(session_id)
            return await self._list_session_turns(
                session_id,
                before_sequence=before_sequence,
                limit=limit,
            )

    async def _list_session_turns(self, session_id: str, *, before_sequence: int | None = None, limit: int = 50) -> Page[SessionTurn]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        session = await self._read_session(session_id)
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
        current = await self.get_session(session_id)
        if current is None or current.latest_completed_run_id is None:
            return ()
        journal_runs = await self._session_journal_runs(session_id)
        run_id = current.latest_completed_run_id
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in journal_runs)),
            ("run", run_id),
        ):
            await self._recover_session(session_id)
            session = await self._read_session(session_id)
            if session is None or session.latest_completed_run_id is None:
                return ()
            run_id = session.latest_completed_run_id
            if run_id != current.latest_completed_run_id:
                raise StorageConflictError(
                    "session latest run changed while loading context"
                )
            await self._recover_run(run_id)
            path = self._snapshot_path(run_id)
            record = await self._read_run(run_id)
            exists = await self._exists(path)
            if (
                record is not None
                    and record.snapshot_revision > 0
                and not exists
            ):
                raise StorageCorruptionError(
                    f"missing snapshot declared by run manifest: {path}"
                )
            snapshot = None if not exists else _snapshot(
                dict(await asyncio.to_thread(read_json, path))
            )
            return () if snapshot is None else snapshot.resume_messages

    async def list_run_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> Page[RunEvent]:
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
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
