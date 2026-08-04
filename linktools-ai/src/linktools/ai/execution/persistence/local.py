#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process JSON-file execution persistence."""

import asyncio
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from linktools.core import environ

from ...storage.coordination.lease import Lease, claim, is_expired, release, renew
from ...storage.database import CoordinationScope
from ...errors import (
    ChildRunAlreadyActiveError,
    ParentLeaseGuardError,
    RunIdentityConflictError,
    StorageConflictError,
    StorageCorruptionError,
    StorageError,
    UsageObservationConflictError,
)
from ...storage.local.files import (
    atomic_write_bytes,
    atomic_write_json,
    read_bytes,
    read_json,
)
from ...storage.local.locks import KeyedLocks
from ...storage.local.paths import StorageId, safe_child
from ..lifecycle import (
    assert_approval_decided,
    assert_claimable,
    assert_owner,
    assert_resumable,
    assert_transition,
)
from ..domain import (
    ApprovalDecision,
    MessageCaptureState,
    RunApproval,
    RunDefinition,
    RunError,
    RunKind,
    RunRecord,
    RunStatus,
    RunUsage,
    RunnableType,
)
from ..domain import Page
from ...evaluation import RunEvaluation
from ..session import SessionRecord, SessionTurn
from ..snapshots import RunSnapshot, is_run_usage_monotonic
from ..trace_models import NewRunTraceStep, RunEvent, RunTraceStep
from ..commands import (
    StartClaimedChildResult,
    StartRunResult,
    run_record_identity,
    start_execution_identity,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import (
        AbortExecution,
        AcknowledgeCancellation,
        ClaimExecution,
        CheckpointExecutionUsage,
        CompleteExecution,
        DecideApproval,
        FailExecution,
        HeartbeatExecution,
        PauseExecution,
        RequestCancellation,
        ResumeExecution,
        StartExecution,
        StartClaimedChildExecution,
    )
    from ..snapshots import AgentSnapshotData
    from ...json import JsonValue


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: object) -> "datetime | None":
    return None if value is None else datetime.fromisoformat(str(value))


def _run(raw: dict) -> RunRecord:
    lease_raw = raw.pop("lease", {})
    raw["kind"] = RunKind(raw["kind"])
    raw["status"] = RunStatus(raw["status"])
    raw["runnable_type"] = RunnableType(raw["runnable_type"])
    definition = raw["definition"]
    definition["runnable_type"] = RunnableType(definition["runnable_type"])
    raw["definition"] = RunDefinition(**definition)
    raw["lease"] = Lease(
        lease_raw.get("owner"),
        lease_raw.get("fence", 0),
        _dt(lease_raw.get("expires_at")),
    )
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
    raw["delta_messages"] = tuple(raw.get("delta_messages") or ())
    raw["capture_state"] = MessageCaptureState(raw.get("capture_state", "complete"))
    raw["created_at"] = _dt(raw["created_at"])
    raw["completed_at"] = _dt(raw.get("completed_at"))
    return SessionTurn(**raw)


def _snapshot(raw: dict) -> RunSnapshot:
    raw["status"] = RunStatus(raw["status"])
    raw["resume_messages"] = tuple(raw["resume_messages"])
    raw["usage"] = RunUsage(**raw["usage"])
    raw["created_at"] = _dt(raw["created_at"])
    return RunSnapshot(**raw)


def _snapshot_json(snapshot: RunSnapshot) -> dict:
    raw = asdict(snapshot)
    usage = raw["usage"]
    if usage["total_cost"] is not None:
        usage["total_cost"] = format(usage["total_cost"], "f")
    return raw


logger = environ.get_logger("ai.execution.persistence.local")


class LocalExecutionBackend:
    """Execution store for one process; construction performs no I/O."""

    coordination_scope = CoordinationScope.PROCESS

    def __init__(self, root: "str | Path" = ".linktools") -> None:
        self.root = Path(root)
        self._locks = KeyedLocks()
        self._recovery_locks = KeyedLocks()

    async def _exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    def _part(self, raw: str) -> StorageId:
        return StorageId.parse(raw)

    def _session_path(self, session_id: str) -> Path:
        sid = self._part(session_id)
        return safe_child(self.root, "sessions", sid, "session.json")

    def _turn_path(self, session_id: str, sequence: int) -> Path:
        return self._session_path(session_id).parent / "turns" / f"{sequence:020d}.json"

    def _run_dir(self, run_id: str) -> Path:
        return safe_child(self.root, "runs", self._part(run_id))

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
        transaction_dirs: "set[Path]" = set()
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
            lambda: (
                tuple(sorted(directory.glob("*.json"))) if directory.exists() else ()
            )
        )
        for path in paths:
            async with self._recovery_locks.acquire(("journal", path.stem)):
                if await self._exists(path):
                    await asyncio.to_thread(self._recover_journal, path)

    async def _recover_run(self, run_id: str) -> None:
        await self._recover_directory(self._journal_dir(run_id))

    async def _run_journal_sessions(self, run_id: str) -> "tuple[str, ...]":
        directory = self._journal_dir(run_id)
        paths = await asyncio.to_thread(
            lambda: tuple(directory.glob("*.json")) if directory.exists() else ()
        )
        session_ids: "set[str]" = set()
        for path in paths:
            try:
                journal = dict(await asyncio.to_thread(read_json, path))
            except (OSError, ValueError, TypeError) as journal_exc:
                logger.warning(
                    "run-journal read failed (lock set may be incomplete): %s: %s",
                    path,
                    journal_exc,
                )
                continue
            for entry in journal.get("entries", ()):
                parts = Path(entry["target"]).parts
                if "sessions" in parts:
                    session_ids.add(parts[parts.index("sessions") + 1])
        return tuple(sorted(session_ids))

    async def _recover_session(self, session_id: str) -> None:
        await self._recover_directory(self._session_journal_dir(session_id))

    async def _session_journal_runs(self, session_id: str) -> "tuple[str, ...]":
        directory = self._session_journal_dir(session_id)
        paths = await asyncio.to_thread(
            lambda: tuple(directory.glob("*.json")) if directory.exists() else ()
        )
        run_ids: "set[str]" = set()
        for path in paths:
            try:
                journal = dict(await asyncio.to_thread(read_json, path))
            except (OSError, ValueError, TypeError) as journal_exc:
                logger.warning(
                    "session-journal read failed (lock set may be incomplete): %s: %s",
                    path,
                    journal_exc,
                )
                continue
            for entry in journal.get("entries", ()):
                parts = Path(entry["target"]).parts
                if "runs" in parts:
                    run_ids.add(parts[parts.index("runs") + 1])
        return tuple(sorted(run_ids))

    def _commit_files(
        self,
        run_id: str,
        writes: "tuple[tuple[Path, object, bool], ...]",
    ) -> None:
        transaction_id = uuid4().hex
        journal_dir = self._journal_dir(run_id)
        transaction_dir = journal_dir / transaction_id
        journal_paths = [journal_dir / f"{transaction_id}.json"]
        entries: "list[dict]" = []
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
                self._session_journal_dir(session_id) / f"{transaction_id}.json"
                for session_id in sorted(session_ids)
            )
            journal["journal_paths"] = [str(path) for path in journal_paths]
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
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def create_session(
        self, *, session_id: str, user_id: "str | None", tenant_id: "str | None"
    ) -> SessionRecord:
        path = self._session_path(session_id)
        async with self._locks.acquire(("session", session_id)):
            if await self._exists(path):
                existing = _session(dict(await asyncio.to_thread(read_json, path)))
                if existing.user_id != user_id or existing.tenant_id != tenant_id:
                    raise StorageConflictError("session ownership conflict")
                return existing
            now = _now()
            value = SessionRecord(session_id, user_id, tenant_id, 1, None, now, now)
            await asyncio.to_thread(atomic_write_json, path, asdict(value))
            return value

    async def get_session(self, session_id: str) -> "SessionRecord | None":
        run_ids = await self._session_journal_runs(session_id)
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in run_ids)),
        ):
            await self._recover_session(session_id)
            return await self._read_session(session_id)

    async def _read_session(self, session_id: str) -> "SessionRecord | None":
        path = self._session_path(session_id)
        if not await self._exists(path):
            return None
        return _session(dict(await asyncio.to_thread(read_json, path)))

    async def list_all_sessions(self) -> "tuple[SessionRecord, ...]":
        """Every persisted session, by scanning sessions/<id>/session.json.
        Local-backend-only enumeration (the SQL backend implements its own)."""
        sessions_dir = self.root / "sessions"

        def _scan() -> "list[SessionRecord]":
            if not sessions_dir.is_dir():
                return []
            out: "list[SessionRecord]" = []
            for entry in sorted(sessions_dir.iterdir()):
                if not entry.is_dir():
                    continue
                path = entry / "session.json"
                if not path.is_file():
                    continue
                try:
                    out.append(_session(dict(read_json(path))))
                except Exception:
                    continue
            return out

        return tuple(await asyncio.to_thread(_scan))

    async def list_all_runs(self) -> "tuple[RunRecord, ...]":
        """Every persisted run, by scanning runs/<id>/run.json."""
        runs_dir = self.root / "runs"

        def _scan() -> "list[RunRecord]":
            if not runs_dir.is_dir():
                return []
            out: "list[RunRecord]" = []
            for entry in sorted(runs_dir.iterdir()):
                if not entry.is_dir():
                    continue
                path = entry / "run.json"
                if not path.is_file():
                    continue
                try:
                    out.append(_run(dict(read_json(path))))
                except Exception as exc:
                    raise StorageCorruptionError(
                        f"invalid persisted run: {path}"
                    ) from exc
            return out

        return tuple(await asyncio.to_thread(_scan))

    async def start_run(self, command: "StartExecution") -> StartRunResult:
        journal_runs = await self._session_journal_runs(command.session_id)
        lock_keys = tuple(("run", run_id) for run_id in journal_runs)
        if command.parent_guard is not None:
            lock_keys += (("run", command.parent_guard.run_id),)
        async with self._locks.acquire(
            ("session", command.session_id),
            *lock_keys,
            ("run", command.run_id),
        ):
            await self._recover_session(command.session_id)
            await self._recover_run(command.run_id)
            if command.parent_guard is not None:
                await self._recover_run(command.parent_guard.run_id)
            session = await self._read_session(command.session_id)
            if session is None:
                raise StorageError("unknown session")
            if command.kind is RunKind.TASK and command.parent_execution_id is not None:
                raise ParentLeaseGuardError(
                    "task_graph child must use atomic child start"
                )
            if (
                command.parent_guard is not None
                and command.parent_execution_id != command.parent_guard.run_id
            ):
                raise ParentLeaseGuardError(
                    "parent lease guard does not match child parent"
                )
            if command.parent_guard is not None:
                parent = await self._read_run(command.parent_guard.run_id)
                now = _now()
                if parent is None:
                    raise ParentLeaseGuardError("parent run does not exist")
                if (
                    parent.status is not RunStatus.RUNNING
                    or parent.lease.owner != command.parent_guard.owner
                    or parent.lease.fence != command.parent_guard.fence
                    or is_expired(parent.lease, now)
                ):
                    raise ParentLeaseGuardError("parent lease guard rejected child")
            existing = await self._read_run(command.run_id)
            identity = start_execution_identity(
                command, tenant_id=session.tenant_id, user_id=session.user_id
            )
            if existing is not None:
                if run_record_identity(existing) == identity:
                    return StartRunResult(existing, created=False)
                raise RunIdentityConflictError(
                    "run id reused with a different start identity"
                )
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
            writes: "list[tuple[Path, object, bool]]" = [
                (
                    self._numbered(command.run_id, "events", 1),
                    asdict(RunEvent(command.run_id, 1, "run.started", {}, now)),
                    False,
                ),
            ]
            if sequence is not None:
                turn = SessionTurn(
                    command.session_id,
                    sequence,
                    command.run_id,
                    command.input,
                    (),
                    RunStatus.PENDING,
                    MessageCaptureState.COMPLETE,
                    now,
                    None,
                )
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
            await asyncio.to_thread(self._commit_files, command.run_id, tuple(writes))
            return StartRunResult(record, created=True)

    async def start_claimed_child(
        self, command: "StartClaimedChildExecution"
    ) -> "StartClaimedChildResult":
        start = command.start
        if start.kind is not RunKind.TASK or start.parent_guard is None:
            raise ParentLeaseGuardError("task child requires parent lease guard")
        journal_runs = await self._session_journal_runs(start.session_id)
        lock_keys = tuple(("run", run_id) for run_id in journal_runs)
        lock_keys += (("run", start.parent_guard.run_id), ("run", start.run_id))
        async with self._locks.acquire(("session", start.session_id), *lock_keys):
            await self._recover_session(start.session_id)
            await self._recover_run(start.run_id)
            await self._recover_run(start.parent_guard.run_id)
            session = await self._read_session(start.session_id)
            if session is None:
                raise StorageError("unknown session")
            if start.parent_execution_id != start.parent_guard.run_id:
                raise ParentLeaseGuardError(
                    "parent lease guard does not match child parent"
                )
            parent = await self._read_run(start.parent_guard.run_id)
            if parent is None:
                raise ParentLeaseGuardError("parent run does not exist")
            if (
                parent.session_id != start.session_id
                or parent.root_execution_id != start.root_execution_id
                or parent.tenant_id != session.tenant_id
                or parent.user_id != session.user_id
            ):
                raise ParentLeaseGuardError("child context does not match parent")
            if (
                parent.status is not RunStatus.RUNNING
                or parent.lease.owner != start.parent_guard.owner
                or parent.lease.fence != start.parent_guard.fence
                or is_expired(parent.lease, command.now)
            ):
                raise ParentLeaseGuardError("parent lease guard rejected child")
            identity = start_execution_identity(
                start, tenant_id=session.tenant_id, user_id=session.user_id
            )
            existing = await self._read_run(start.run_id)
            if existing is not None:
                if run_record_identity(existing) != identity:
                    raise RunIdentityConflictError(
                        "run id reused with a different start identity"
                    )
                if existing.status in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                }:
                    return StartClaimedChildResult(
                        existing, created=False, terminal=True
                    )
                if existing.status is not RunStatus.PENDING:
                    raise ChildRunAlreadyActiveError(existing.id)
                leased = claim(
                    existing.lease,
                    owner=command.child_owner,
                    now=command.now,
                    duration=command.lease_duration,
                )
                updated = replace(
                    existing,
                    status=RunStatus.RUNNING,
                    lease=leased,
                    event_sequence=existing.event_sequence + 1,
                    updated_at=command.now,
                )
                await asyncio.to_thread(
                    self._commit_files,
                    start.run_id,
                    (
                        (
                            self._numbered(
                                start.run_id, "events", updated.event_sequence
                            ),
                            asdict(
                                RunEvent(
                                    start.run_id,
                                    updated.event_sequence,
                                    "run.claimed",
                                    {},
                                    command.now,
                                )
                            ),
                            False,
                        ),
                        (self._run_path(start.run_id), asdict(updated), True),
                    ),
                )
                return StartClaimedChildResult(updated, created=False, terminal=False)
            now = command.now
            leased = claim(
                Lease(),
                owner=command.child_owner,
                now=now,
                duration=command.lease_duration,
            )
            record = RunRecord(
                id=start.run_id,
                session_id=start.session_id,
                kind=RunKind.TASK,
                runnable_id=start.definition.runnable_id,
                runnable_type=start.definition.runnable_type,
                definition=start.definition,
                status=RunStatus.RUNNING,
                session_turn_sequence=None,
                parent_execution_id=start.parent_execution_id,
                root_execution_id=start.root_execution_id or start.run_id,
                approval=None,
                lease=leased,
                cancel_requested_at=None,
                snapshot_revision=0,
                trace_sequence=0,
                event_sequence=2,
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                error=None,
                created_at=now,
                updated_at=now,
                input=start.input,
            )
            writes = (
                (
                    self._numbered(start.run_id, "events", 1),
                    asdict(RunEvent(start.run_id, 1, "run.started", {}, now)),
                    False,
                ),
                (
                    self._numbered(start.run_id, "events", 2),
                    asdict(RunEvent(start.run_id, 2, "run.claimed", {}, now)),
                    False,
                ),
                (self._run_path(start.run_id), asdict(record), True),
            )
            await asyncio.to_thread(self._commit_files, start.run_id, writes)
            logger.debug(
                "child %s started directly RUNNING owner=%s fence=%s",
                start.run_id,
                command.child_owner,
                leased.fence,
            )
            return StartClaimedChildResult(record, created=True, terminal=False)

    async def get_run(self, run_id: str) -> "RunRecord | None":
        sessions = await self._run_journal_sessions(run_id)
        async with self._locks.acquire(
            *((("session", session_id) for session_id in sessions)),
            ("run", run_id),
        ):
            await self._recover_run(run_id)
            return await self._read_run(run_id)

    async def _read_run(self, run_id: str) -> "RunRecord | None":
        path = self._run_path(run_id)
        if not await self._exists(path):
            return None
        return _run(dict(await asyncio.to_thread(read_json, path)))

    async def list_runs_by_ids(
        self, run_ids: "tuple[str, ...]"
    ) -> "tuple[RunRecord, ...]":
        records = []
        for run_id in dict.fromkeys(run_ids):
            record = await self.get_run(run_id)
            if record is not None:
                records.append(record)
        return tuple(records)

    async def assert_active_lease(
        self, run_id: str, *, owner: str, fence: int
    ) -> None:
        record = await self.get_run(run_id)
        if record is None:
            raise StorageError(f"unknown run: {run_id}")
        if (
            record.status not in {RunStatus.RUNNING, RunStatus.CANCELLING}
            or record.lease.owner != owner
            or record.lease.fence != fence
            or is_expired(record.lease, _now())
        ):
            raise StorageConflictError("parent run lease is not active")

    async def claim_run(self, command: "ClaimExecution") -> RunRecord:
        await self.get_run(command.run_id)
        initial = await self._read_run(command.run_id)
        if initial is None:
            raise StorageError("unknown run")
        lock_keys = [("run", command.run_id)]
        if initial.parent_execution_id is not None:
            lock_keys.append(("run", initial.parent_execution_id))
        async with self._locks.acquire(*lock_keys):
            record = await self._read_run(command.run_id)
            if record is None:
                raise StorageError("unknown run")
            if record.kind is RunKind.TASK and record.parent_execution_id is not None:
                guard = command.parent_guard
                if guard is None or guard.run_id != record.parent_execution_id:
                    raise ParentLeaseGuardError(
                        "task child claim requires parent guard"
                    )
                parent = await self._read_run(record.parent_execution_id)
                if (
                    parent is None
                    or parent.status is not RunStatus.RUNNING
                    or parent.lease.owner != guard.owner
                    or parent.lease.fence != guard.fence
                    or is_expired(parent.lease, command.now)
                ):
                    raise ParentLeaseGuardError(
                        "parent lease guard rejected child claim"
                    )
            assert_claimable(record, command.now)
            updated = replace(
                record,
                status=RunStatus.RUNNING,
                lease=claim(
                    record.lease,
                    owner=command.owner,
                    now=command.now,
                    duration=command.duration,
                ),
                updated_at=command.now,
            )
            await asyncio.to_thread(
                atomic_write_json, self._run_path(command.run_id), asdict(updated)
            )
            return updated

    async def claim_run_for_recovery(
        self,
        run_id: str,
        *,
        owner: str,
        now: datetime,
        duration,
    ) -> RunRecord:
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
            record = await self._required_run(run_id)
            if record.status is RunStatus.PENDING:
                target = RunStatus.RUNNING
            elif record.status in {RunStatus.RUNNING, RunStatus.CANCELLING}:
                if record.lease.expires_at is None or not is_expired(record.lease, now):
                    raise StorageConflictError("run recovery lease is still active")
                target = record.status
            else:
                raise StorageConflictError("terminal or paused run cannot be recovered")
            updated = replace(
                record,
                status=target,
                lease=claim(record.lease, owner=owner, now=now, duration=duration),
                updated_at=now,
            )
            await asyncio.to_thread(
                atomic_write_json, self._run_path(run_id), asdict(updated)
            )
            return updated

    async def heartbeat_run(self, command: "HeartbeatExecution") -> RunRecord:
        await self.get_run(command.run_id)
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            updated = replace(
                record,
                lease=renew(
                    record.lease,
                    owner=command.owner,
                    fence=command.fence,
                    now=command.now,
                    duration=command.duration,
                ),
                updated_at=command.now,
            )
            await asyncio.to_thread(
                atomic_write_json, self._run_path(command.run_id), asdict(updated)
            )
            return updated

    async def request_cancel(self, command: "RequestCancellation") -> RunRecord:
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
                updated = replace(
                    record,
                    status=RunStatus.CANCELLED,
                    lease=release(record.lease),
                    cancel_requested_at=now,
                    event_sequence=record.event_sequence + 1,
                    updated_at=now,
                )
                additional: "tuple[tuple[Path, object], ...]" = ()
                if record.session_turn_sequence is not None:
                    turn = _turn(
                        dict(
                            await asyncio.to_thread(
                                read_json,
                                self._turn_path(
                                    record.session_id, record.session_turn_sequence
                                ),
                            )
                        )
                    )
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
                                    capture_state=MessageCaptureState.COMPLETE,
                                    completed_at=now,
                                )
                            ),
                        ),
                    )
                    # Clear the resume checkpoint (no longer resumable) in the
                    # same logical transaction. Read the raw snapshot file
                    # directly -- get_snapshot() would re-acquire the run lock
                    # we already hold (deadlock).
                    snapshot_path = self._snapshot_path(command.run_id)
                    if await self._exists(snapshot_path):
                        raw = dict(await asyncio.to_thread(read_json, snapshot_path))
                        raw["resume_messages"] = []
                        raw["status"] = RunStatus.CANCELLED.value
                        additional = additional + ((snapshot_path, raw),)
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
            updated = replace(
                record,
                status=RunStatus.CANCELLING,
                cancel_requested_at=command.requested_at,
                event_sequence=record.event_sequence + 1,
                updated_at=command.requested_at,
            )
            await self._write_run_event(updated, "run.cancelling", command.requested_at)
            return updated

    async def pause_run(self, command: "PauseExecution") -> RunRecord:
        return await self._finish(
            command.run_id,
            command.owner,
            command.fence,
            command.snapshot,
            RunStatus.PAUSED,
            expected_snapshot_revision=command.expected_snapshot_revision,
            pending_approval=command.pending_approval,
        )

    async def resume_run(self, command: "ResumeExecution") -> RunRecord:
        await self.get_run(command.run_id)
        async with self._locks.acquire(("run", command.run_id)):
            record = await self._required_run(command.run_id)
            assert_resumable(record)
            assert_approval_decided(record)
            updated = replace(
                record,
                status=RunStatus.PENDING,
                event_sequence=record.event_sequence + 1,
                updated_at=_now(),
            )
            await self._write_run_event(updated, "run.resumed", updated.updated_at)
            return updated

    async def decide_approval(self, command: "DecideApproval") -> RunRecord:
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(
            ("session", current.session_id), ("run", command.run_id)
        ):
            record = await self._required_run(command.run_id)
            if (
                record.approval is None
                or record.approval.approval_id != command.approval_id
            ):
                raise StorageError("run has no pending approval")
            existing = record.approval.decision
            if existing is not None:
                if (
                    existing == command.decision
                    and record.approval.decided_by == command.decided_by
                ):
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
                updated = replace(
                    record,
                    approval=new_approval,
                    event_sequence=record.event_sequence + 1,
                    updated_at=decided_at,
                )
                await self._write_run_event(
                    updated,
                    "run.approval_decided",
                    updated.updated_at,
                    payload=asdict(new_approval),
                )
                return updated
            now = decided_at
            updated = replace(
                record,
                status=RunStatus.CANCELLED,
                approval=new_approval,
                lease=release(record.lease),
                event_sequence=record.event_sequence + 1,
                updated_at=now,
            )
            additional = ()
            if record.session_turn_sequence is not None:
                turn = _turn(
                    dict(
                        await asyncio.to_thread(
                            read_json,
                            self._turn_path(
                                record.session_id, record.session_turn_sequence
                            ),
                        )
                    )
                )
                additional = additional + (
                    (
                        self._turn_path(
                            record.session_id, record.session_turn_sequence
                        ),
                        asdict(
                            replace(
                                turn,
                                status=RunStatus.CANCELLED,
                                capture_state=MessageCaptureState.COMPLETE,
                                completed_at=now,
                            )
                        ),
                    ),
                )
                # PAUSED->DENY->CANCELLED: clear the resume checkpoint (same rule
                # as PAUSED->CANCELLED). Read raw to avoid re-acquiring the lock.
                snapshot_path = self._snapshot_path(command.run_id)
                if await self._exists(snapshot_path):
                    raw = dict(await asyncio.to_thread(read_json, snapshot_path))
                    raw["resume_messages"] = []
                    raw["status"] = RunStatus.CANCELLED.value
                    additional = additional + ((snapshot_path, raw),)
            await self._write_run_event(
                updated,
                "run.approval_decided",
                now,
                payload=asdict(new_approval),
                additional_immutable=additional,
            )
            return updated

    async def complete_run(self, command: "CompleteExecution") -> RunRecord:
        return await self._finish(
            command.run_id,
            command.owner,
            command.fence,
            command.snapshot,
            RunStatus.COMPLETED,
            expected_snapshot_revision=command.expected_snapshot_revision,
        )

    async def fail_run(self, command: "FailExecution") -> RunRecord:
        return await self._finish(
            command.run_id,
            command.owner,
            command.fence,
            command.snapshot,
            RunStatus.FAILED,
            error=command.error,
            expected_snapshot_revision=command.expected_snapshot_revision,
        )

    async def acknowledge_cancel(self, command: "AcknowledgeCancellation") -> RunRecord:
        return await self._finish(
            command.run_id,
            command.owner,
            command.fence,
            command.snapshot,
            RunStatus.CANCELLED,
            expected_snapshot_revision=command.expected_snapshot_revision,
        )

    async def abort_run(self, command: "AbortExecution") -> RunRecord:
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(
            ("session", current.session_id), ("run", command.run_id)
        ):
            record = await self._required_run(command.run_id)
            if record.snapshot_revision != command.expected_snapshot_revision:
                raise StorageConflictError("run snapshot revision changed")
            if record.status is not RunStatus.PENDING:
                assert_owner(record, command.owner, command.fence, _now())
            assert_transition(record.status, RunStatus.FAILED)
            previous = await self._read_snapshot_locked(record)
            if previous is not None and not is_run_usage_monotonic(
                previous.usage, command.snapshot.usage
            ):
                raise StorageConflictError("run usage regressed")
            now = _now()
            snapshot_revision = record.snapshot_revision + 1
            updated = replace(
                record,
                status=RunStatus.FAILED,
                error=command.error,
                lease=release(record.lease),
                trace_sequence=command.trace_end_sequence,
                snapshot_revision=snapshot_revision,
                event_sequence=record.event_sequence + 1,
                updated_at=now,
            )
            stored_snapshot = RunSnapshot(
                "run-snapshot.v1",
                command.run_id,
                snapshot_revision,
                (),
                command.snapshot.final_output,
                RunStatus.FAILED,
                command.snapshot.usage,
                command.snapshot.trace_end_sequence,
                now,
            )
            additional = (
                (self._snapshot_path(command.run_id), _snapshot_json(stored_snapshot)),
            )
            if record.session_turn_sequence is not None:
                turn = _turn(
                    dict(
                        await asyncio.to_thread(
                            read_json,
                            self._turn_path(
                                record.session_id, record.session_turn_sequence
                            ),
                        )
                    )
                )
                # The partial snapshot has no trustworthy turn delta.
                additional = additional + (
                    (
                        self._turn_path(
                            record.session_id, record.session_turn_sequence
                        ),
                        asdict(
                            replace(
                                turn,
                                status=RunStatus.FAILED,
                                capture_state=MessageCaptureState.UNAVAILABLE,
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

    async def checkpoint_run_usage(
        self, command: "CheckpointExecutionUsage"
    ) -> RunSnapshot:
        current = await self.get_run(command.run_id)
        if current is None:
            raise StorageError(f"unknown run: {command.run_id}")
        async with self._locks.acquire(("session", current.session_id), ("run", command.run_id)):
            record = await self._required_run(command.run_id)
            assert_owner(record, command.owner, command.fence, _now())
            if record.status not in {RunStatus.RUNNING, RunStatus.CANCELLING}:
                raise StorageConflictError("run is not checkpointable")
            if record.snapshot_revision != command.expected_snapshot_revision:
                latest = await self._read_snapshot_locked(record)
                if latest is not None and (
                    latest.usage == command.usage
                    or is_run_usage_monotonic(command.usage, latest.usage)
                ):
                    return latest
                raise UsageObservationConflictError(
                    "usage checkpoint revision carries a conflicting usage"
                )
            previous = await self._read_snapshot_locked(record)
            previous_usage = (
                previous.usage
                if previous is not None
                else RunUsage(total_cost=Decimal("0"))
            )
            if not is_run_usage_monotonic(previous_usage, command.usage):
                raise StorageConflictError("run usage regressed")
            now = _now()
            revision = record.snapshot_revision + 1
            event_sequence = record.event_sequence + 1
            snapshot = RunSnapshot(
                "run-snapshot.v1",
                record.id,
                revision,
                (),
                None,
                record.status,
                command.usage,
                command.trace_end_sequence,
                now,
            )
            updated = replace(
                record,
                snapshot_revision=revision,
                trace_sequence=command.trace_end_sequence,
                event_sequence=event_sequence,
                updated_at=now,
            )
            writes = (
                (self._snapshot_path(record.id), _snapshot_json(snapshot), False),
                (
                    self._numbered(record.id, "events", event_sequence),
                    asdict(
                        RunEvent(
                            record.id,
                            event_sequence,
                            "run.usage_checkpoint",
                            {},
                            now,
                        )
                    ),
                    False,
                ),
                (self._run_path(record.id), asdict(updated), True),
            )
            await asyncio.to_thread(self._commit_files, record.id, writes)
            logger.debug(
                "run %s usage checkpoint revision=%s trace=%s",
                record.id,
                revision,
                command.trace_end_sequence,
            )
            return snapshot

    async def _finish(
        self,
        run_id: str,
        owner: str,
        fence: int,
        snapshot: "AgentSnapshotData",
        status: RunStatus,
        expected_snapshot_revision: int,
        pending_approval: "RunApproval | None" = None,
        error: "RunError | None" = None,
    ) -> RunRecord:
        record = await self.get_run(run_id)
        if record is None:
            raise StorageError(f"unknown run: {run_id}")
        session_id = record.session_id
        async with self._locks.acquire(("session", session_id), ("run", run_id)):
            record = await self._required_run(run_id)
            if record.snapshot_revision != expected_snapshot_revision:
                raise StorageConflictError("run snapshot revision changed")
            assert_owner(record, owner, fence, _now())
            assert_transition(record.status, status)
            previous = await self._read_snapshot_locked(record)
            if previous is not None and not is_run_usage_monotonic(
                previous.usage, snapshot.usage
            ):
                raise StorageConflictError("run usage regressed")
            # The store allocates the snapshot revision (expected + 1); the
            # write lock serializes concurrent commits within this process.
            new_revision = record.snapshot_revision + 1
            now = _now()
            # RESUME_CHECKPOINT: non-empty ONLY for PAUSED; cleared on terminal.
            checkpoint = (
                snapshot.checkpoint_messages if status is RunStatus.PAUSED else ()
            )
            stored_snapshot = RunSnapshot(
                "run-snapshot.v1",
                run_id,
                new_revision,
                checkpoint,
                snapshot.final_output,
                status,
                snapshot.usage,
                snapshot.trace_end_sequence,
                now,
            )
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
            writes: "list[tuple[Path, object, bool]]" = [
                (self._snapshot_path(run_id), _snapshot_json(stored_snapshot), False),
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
                turn = _turn(
                    dict(
                        await asyncio.to_thread(
                            read_json,
                            self._turn_path(session_id, record.session_turn_sequence),
                        )
                    )
                )
                completed = status in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                }
                # TURN_DELTA: append this run's new_messages() onto whatever the
                # turn already holds (fresh == empty; resumed == m_partial).
                writes.append(
                    (
                        self._turn_path(session_id, record.session_turn_sequence),
                        asdict(
                            replace(
                                turn,
                                status=status,
                                delta_messages=tuple(turn.delta_messages)
                                + tuple(snapshot.delta_messages),
                                capture_state=snapshot.capture_state,
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

    async def _read_snapshot_locked(
        self, record: RunRecord
    ) -> "RunSnapshot | None":
        path = self._snapshot_path(record.id)
        if not await self._exists(path):
            if record.snapshot_revision > 0:
                raise StorageCorruptionError(
                    f"missing snapshot declared by run manifest: {path}"
                )
            return None
        return _snapshot(dict(await asyncio.to_thread(read_json, path)))

    async def _write_run_event(
        self,
        record: RunRecord,
        event_type: str,
        created_at: datetime,
        *,
        payload: "dict | None" = None,
        additional_immutable: "tuple[tuple[Path, object], ...]" = (),
    ) -> None:
        immutable = tuple((path, value, False) for path, value in additional_immutable)
        await asyncio.to_thread(
            self._commit_files,
            record.id,
            (
                (
                    self._numbered(record.id, "events", record.event_sequence),
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

    async def append_trace_steps(
        self,
        run_id: str,
        *,
        expected_sequence: int,
        steps: "tuple[NewRunTraceStep, ...]",
    ) -> int:
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
            record = await self._required_run(run_id)
            if record.trace_sequence != expected_sequence:
                raise StorageConflictError("trace sequence conflict")
            sequence = expected_sequence
            writes: "list[tuple[Path, object, bool]]" = []
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

    async def list_trace_steps(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: "int | None" = None,
    ) -> "tuple[RunTraceStep, ...]":
        await self.get_run(run_id)
        async with self._locks.acquire(("run", run_id)):
            record = await self._required_run(run_id)
            end = (
                record.trace_sequence
                if through_sequence is None
                else min(through_sequence, record.trace_sequence)
            )
            values = []
            for sequence in range(after_sequence + 1, end + 1):
                path = self._numbered(run_id, "trace", sequence)
                if not await self._exists(path):
                    raise StorageCorruptionError(f"missing trace step: {path}")
                raw = dict(await asyncio.to_thread(read_json, path))
                raw["created_at"] = _dt(raw["created_at"])
                values.append(RunTraceStep(**raw))
            return tuple(values)

    async def get_snapshot(self, run_id: str) -> "RunSnapshot | None":
        sessions = await self._run_journal_sessions(run_id)
        async with self._locks.acquire(
            *((("session", session_id) for session_id in sessions)),
            ("run", run_id),
        ):
            await self._recover_run(run_id)
            record = await self._read_run(run_id)
            path = self._snapshot_path(run_id)
            exists = await self._exists(path)
            if record is not None and record.snapshot_revision > 0 and not exists:
                raise StorageCorruptionError(
                    f"missing snapshot declared by run manifest: {path}"
                )
            return (
                None
                if not exists
                else _snapshot(dict(await asyncio.to_thread(read_json, path)))
            )

    async def list_session_turns(
        self, session_id: str, *, before_sequence: "int | None" = None, limit: int = 50
    ) -> "Page[SessionTurn]":
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

    async def _list_session_turns(
        self, session_id: str, *, before_sequence: "int | None" = None, limit: int = 50
    ) -> "Page[SessionTurn]":
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        session = await self._read_session(session_id)
        if session is None:
            return Page((), False, None)
        end = (
            session.next_turn_sequence - 1
            if before_sequence is None
            else before_sequence - 1
        )
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

    async def load_session_context(self, session_id: str) -> "tuple[JsonValue, ...]":
        # Session Context: COMPLETED turns' TURN_DELTA concatenated in sequence
        # order. PAUSED/CANCELLED/FAILED deltas are excluded.
        session = await self.get_session(session_id)
        if session is None:
            return ()
        journal_runs = await self._session_journal_runs(session_id)
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in journal_runs)),
        ):
            await self._recover_session(session_id)
            session = await self._read_session(session_id)
            if session is None:
                return ()
            messages: "list[JsonValue]" = []
            for sequence in range(1, session.next_turn_sequence):
                path = self._turn_path(session_id, sequence)
                if not await self._exists(path):
                    continue
                turn = _turn(dict(await asyncio.to_thread(read_json, path)))
                if turn.status is RunStatus.COMPLETED:
                    messages.extend(turn.delta_messages)
            return tuple(messages)

    async def load_resume_messages(self, execution_id: str) -> "tuple[JsonValue, ...]":
        # Resume Context: the PAUSED run's RESUME_CHECKPOINT only.
        snapshot = await self.get_snapshot(run_id=execution_id)
        return () if snapshot is None else snapshot.resume_messages

    async def get_session_messages(self, session_id: str) -> "tuple[SessionTurn, ...]":
        # Audit History: every turn (any status) with its TURN_DELTA + status +
        # capture_state, in sequence order.
        session = await self.get_session(session_id)
        if session is None:
            return ()
        journal_runs = await self._session_journal_runs(session_id)
        async with self._locks.acquire(
            ("session", session_id),
            *((("run", run_id) for run_id in journal_runs)),
        ):
            await self._recover_session(session_id)
            session = await self._read_session(session_id)
            if session is None:
                return ()
            values: "list[SessionTurn]" = []
            for sequence in range(1, session.next_turn_sequence):
                path = self._turn_path(session_id, sequence)
                if not await self._exists(path):
                    continue
                values.append(_turn(dict(await asyncio.to_thread(read_json, path))))
            return tuple(values)

    async def get_turn(self, session_id: str, sequence: int) -> "SessionTurn | None":
        # O(1) single-turn read by (session_id, sequence).
        path = self._turn_path(session_id, sequence)
        if not await self._exists(path):
            return None
        return _turn(dict(await asyncio.to_thread(read_json, path)))

    async def list_run_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> "Page[RunEvent]":
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
            return Page(
                tuple(values[:limit]),
                len(values) > limit,
                values[limit - 1].sequence if len(values) > limit else None,
            )

    async def save_evaluation(self, evaluation: RunEvaluation) -> None:
        await asyncio.to_thread(
            atomic_write_json,
            self._numbered(evaluation.run_id, "evaluations", 0).with_name(
                f"{StorageId.parse(evaluation.evaluation_id).value}.json"
            ),
            asdict(evaluation),
        )

    async def list_evaluations(self, run_id: str) -> "tuple[RunEvaluation, ...]":
        directory = self._run_dir(run_id) / "evaluations"
        names = await asyncio.to_thread(
            lambda: tuple(directory.iterdir()) if directory.exists() else ()
        )
        values = []
        for path in names:
            raw = dict(await asyncio.to_thread(read_json, path))
            raw["created_at"] = _dt(raw["created_at"])
            values.append(RunEvaluation(**raw))
        return tuple(sorted(values, key=lambda item: item.created_at))


__all__ = ["LocalExecutionBackend", "StorageCorruptionError"]
