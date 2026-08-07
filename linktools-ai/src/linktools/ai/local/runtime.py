#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Agent runtime used by the CLI and ACP composition roots."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from linktools.core import environ

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from ..agent.runner import LocalAgentRunner
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
from ..core.json import JsonValue
from ..core.value import ExecutionEventType, ExecutionProfile, ExecutionStatus, IdempotencyStatus, SessionStatus, StopReason
from ..runtime.persistence import BlobRef, ExecutionRecord, ExecutionTerminalCommit, IdempotencyRecord, ResultRecord, RuntimePersistence, SessionRecord
from ..storage.files import write_bytes_atomic
from .project import LocalProject
from .persistence import build_file_runtime
from .record import LocalExecutionRecord, LocalRecordStore

logger = environ.get_logger("ai.local.runtime")


class TextHandler(Protocol):
    async def __call__(self, text: str) -> None: ...


class EventHandler(Protocol):
    def __call__(self, event: 'dict[str, JsonValue]') -> 'Awaitable[None] | None': ...


@dataclass(frozen=True, slots=True)
class LocalRunResult:
    execution_id: str
    session_id: str
    run_id: str
    output: str


@dataclass(frozen=True, slots=True)
class LocalSession:
    session_id: str
    cwd: Path
    session_revision: int = 0


class LocalAgentRuntime:
    """Run one local Agent while keeping session history under the project root."""

    def __init__(
        self,
        project: LocalProject,
        *,
        runner: LocalAgentRunner,
        persistence: "RuntimePersistence | None" = None,
    ) -> None:
        self.project = project
        self._runner = runner
        self._sessions: "dict[str, LocalSession]" = {}
        self._tasks: "dict[str, asyncio.Task[LocalRunResult]]" = {}
        self._session_runs: "dict[str, set[str]]" = {}
        self._idempotency_tasks: "dict[tuple[str, str], asyncio.Task[LocalRunResult]]" = {}
        self._shutdown_ids: set[str] = set()
        self._records = LocalRecordStore(project.storage_root, project.project_id, work_root=project.root)
        self._history_locks: "dict[str, asyncio.Lock]" = {}
        self._lock = asyncio.Lock()
        self._persistence_runtime = persistence or build_file_runtime(str(project.storage_root), project_id=project.project_id, local_tenant_id=project.project_id)

    @property
    def sessions_root(self) -> Path:
        return self.project.storage_root / ".linktools" / "sessions"

    async def run(
        self,
        session_id: str,
        prompt: str,
        *,
        cwd: "str | Path | None" = None,
        agent_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        on_text: "TextHandler | None" = None,
        on_event: "EventHandler | None" = None,
    ) -> LocalRunResult:
        _validate_session_id(session_id)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        await self._records.initialize()
        await self._persistence_runtime.initialize()
        await self._reconcile_restarted_executions()
        session = await self.open_session(session_id, cwd=cwd, binding_digest=canonical_sha256({"agent_id": agent_id}))
        persisted_session = await self._persistence_runtime.persistence.sessions.get(
            session_id,
            tenant_id=self.project.project_id,
        )
        if persisted_session is None or persisted_session.status is not SessionStatus.OPEN:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT)
        execution_id = uuid4().hex
        created_at = datetime.now(timezone.utc)
        existing_result: LocalRunResult | None = None
        existing_task: asyncio.Task[LocalRunResult] | None = None
        effective_key = idempotency_key or f"local:{session_id}:{execution_id}"
        request_digest = canonical_sha256({"prompt": prompt, "agent_id": agent_id})
        async with self._lock:
            existing = await self._persistence_runtime.persistence.idempotency.get("local.execution", effective_key, tenant_id=self.project.project_id)
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                existing_task = self._tasks.get(existing.execution_id)
                if existing_task is None and existing.status is IdempotencyStatus.COMPLETED:
                    existing_result = await self._load_persisted_result(existing.execution_id, session_id)
                if existing_task is None and existing_result is None and existing.status in {IdempotencyStatus.STARTED, IdempotencyStatus.START_UNKNOWN}:
                    raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
                if existing.status is IdempotencyStatus.RESERVED:
                    raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
                if existing.status is IdempotencyStatus.FAILED:
                    raise _stable_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
                if existing.status is IdempotencyStatus.CANCELLED:
                    raise _stable_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            if existing_result is not None or existing_task is not None:
                task = None
            else:
                now = created_at
                await self._persistence_runtime.persistence.idempotency.reserve(IdempotencyRecord(self.project.project_id, "local.execution", effective_key, request_digest, execution_id, IdempotencyStatus.RESERVED, None, None, now, now))
                await self._persistence_runtime.persistence.executions.create(ExecutionRecord(execution_id, self.project.project_id, session_id, ExecutionProfile.LOCAL_CODING, canonical_sha256({"agent_id": agent_id}), None, execution_id, ExecutionStatus.PENDING_START, 0, 0, 0, None, None, None, {}, now, now))
                await self._records.save(
                    LocalExecutionRecord(
                        self.project.project_id,
                        session_id,
                        session.session_revision,
                        session.cwd.as_posix(),
                        execution_id,
                        "PENDING_START",
                        created_at,
                        created_at,
                        None,
                        effective_key,
                    )
                )
                task: asyncio.Task[LocalRunResult] | None = None
                start_committed = asyncio.Event()
                try:
                    task = asyncio.create_task(self._run(session, prompt, agent_id, on_text, on_event, execution_id, start_committed))
                    self._tasks[execution_id] = task
                    self._session_runs.setdefault(session_id, set()).add(execution_id)
                    self._idempotency_tasks[(session_id, effective_key)] = task
                    started = ExecutionRecord(execution_id, self.project.project_id, session_id, ExecutionProfile.LOCAL_CODING, canonical_sha256({"agent_id": agent_id}), None, execution_id, ExecutionStatus.STARTED, 1, 0, 0, None, None, None, {}, created_at, datetime.now(timezone.utc))
                    await self._persistence_runtime.persistence.executions.compare_and_swap(execution_id, tenant_id=self.project.project_id, expected_snapshot_revision=0, next_record=started)
                    await self._persistence_runtime.persistence.idempotency.compare_and_swap("local.execution", effective_key, expected_status=IdempotencyStatus.RESERVED, next_record=IdempotencyRecord(self.project.project_id, "local.execution", effective_key, request_digest, execution_id, IdempotencyStatus.STARTED, None, None, created_at, datetime.now(timezone.utc)))
                    await self._records.save(
                        LocalExecutionRecord(
                            self.project.project_id,
                            session_id,
                            session.session_revision,
                            session.cwd.as_posix(),
                            execution_id,
                            "STARTED",
                            created_at,
                            datetime.now(timezone.utc),
                            None,
                            effective_key,
                        )
                    )
                    start_committed.set()
                except Exception as error:
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    self._tasks.pop(execution_id, None)
                    session_runs = self._session_runs.get(session_id)
                    if session_runs is not None:
                        session_runs.discard(execution_id)
                        if not session_runs:
                            self._session_runs.pop(session_id, None)
                    self._idempotency_tasks.pop((session_id, effective_key), None)
                    await self._save_record(execution_id, session, created_at, "FAILED_START", "EXECUTION_START_PERSISTENCE_FAILED", effective_key)
                    await self._persist_failure(execution_id, effective_key, request_digest, session_id, created_at, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED.value)
                    raise LinktoolsAIError(ErrorCode.EXECUTION_START_PERSISTENCE_FAILED) from error
        if existing_result is not None:
            return existing_result
        if existing_task is not None:
            return await asyncio.shield(existing_task)
        if task is None:
            raise RuntimeError("local execution task was not created")
        try:
            result = await task
            completed = LocalRunResult(execution_id, result.session_id, result.run_id, result.output)
            await self._save_record(execution_id, session, created_at, "SUCCEEDED", None, effective_key)
            await self._persist_success(completed, effective_key, request_digest, created_at)
            return completed
        except asyncio.CancelledError:
            status = "PROCESS_SHUTDOWN" if execution_id in self._shutdown_ids else "CANCELLED"
            reason = "PROCESS_SHUTDOWN" if status == "PROCESS_SHUTDOWN" else "cancelled"
            await self._save_record(execution_id, session, created_at, status, reason, effective_key)
            await self._persist_cancelled(execution_id, effective_key, request_digest, created_at, reason)
            raise
        except Exception as error:
            error_digest = canonical_sha256({"type": type(error).__name__})
            logger.warning("local execution failed: execution=%s error_digest=%s", execution_id, error_digest)
            await self._save_record(execution_id, session, created_at, "FAILED", "EXECUTION_FAILED", effective_key)
            await self._persist_failure(execution_id, effective_key, request_digest, session_id, created_at, "EXECUTION_FAILED")
            raise
        finally:
            async with self._lock:
                self._tasks.pop(execution_id, None)
                session_runs = self._session_runs.get(session_id)
                if session_runs is not None:
                    session_runs.discard(execution_id)
                    if not session_runs:
                        self._session_runs.pop(session_id, None)
                self._shutdown_ids.discard(execution_id)
                self._idempotency_tasks.pop((session_id, effective_key), None)

    async def cancel(self, session_id: str) -> bool:
        async with self._lock:
            tasks = tuple(
                self._tasks[execution_id]
                for execution_id in self._session_runs.get(session_id, ())
                if execution_id in self._tasks
            )
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def _load_persisted_result(self, execution_id: str, session_id: str) -> LocalRunResult | None:
        result = await self._persistence_runtime.persistence.results.get(execution_id, tenant_id=self.project.project_id)
        if result is None or result.payload_ref is None:
            return None
        blob = await self._persistence_runtime.persistence.blobs.stat(
            BlobRef(self.project.project_id, result.payload_ref, 0, ""),
            tenant_id=self.project.project_id,
        )
        if blob is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = bytearray()
        async for chunk in self._persistence_runtime.persistence.blobs.open(blob, tenant_id=self.project.project_id):
            content.extend(chunk)
        return LocalRunResult(execution_id, session_id, execution_id, bytes(content).decode("utf-8"))

    async def _persist_success(self, completed: LocalRunResult, key: str, request_digest: str, created_at: datetime) -> None:
        blob = await self._persistence_runtime.persistence.blobs.put_bytes(tenant_id=self.project.project_id, data=completed.output.encode("utf-8"))
        current = await self._persistence_runtime.persistence.executions.get(completed.execution_id, tenant_id=self.project.project_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        now = datetime.now(timezone.utc)
        terminal = _terminal_execution(current, ExecutionStatus.SUCCEEDED, now, result_ref=blob.digest, result_digest=blob.digest)
        result = ResultRecord(completed.execution_id, self.project.project_id, ExecutionStatus.SUCCEEDED, "none", 1, "none", blob.digest, blob.digest, StopReason.END_TURN, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(ExecutionTerminalCommit(completed.execution_id, current.snapshot_revision, terminal, result))
        await self._persistence_runtime.persistence.idempotency.compare_and_swap("local.execution", key, expected_status=IdempotencyStatus.STARTED, next_record=IdempotencyRecord(self.project.project_id, "local.execution", key, request_digest, completed.execution_id, IdempotencyStatus.COMPLETED, blob.digest, None, created_at, now))
        await self._append_terminal_event(completed.execution_id, ExecutionEventType.EXECUTION_SUCCEEDED)

    async def _persist_failure(self, execution_id: str, key: str, request_digest: str, session_id: str, created_at: datetime, error_code: str) -> None:
        current = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        terminal = _terminal_execution(current, ExecutionStatus.FAILED, now, error_code=error_code)
        result = ResultRecord(execution_id, self.project.project_id, ExecutionStatus.FAILED, "none", 1, "none", None, None, StopReason.ERROR, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(ExecutionTerminalCommit(execution_id, current.snapshot_revision, terminal, result))
        status = await self._persistence_runtime.persistence.idempotency.get("local.execution", key, tenant_id=self.project.project_id)
        if status is not None and status.status not in {IdempotencyStatus.FAILED, IdempotencyStatus.COMPLETED, IdempotencyStatus.CANCELLED}:
            await self._persistence_runtime.persistence.idempotency.compare_and_swap("local.execution", key, expected_status=status.status, next_record=IdempotencyRecord(self.project.project_id, "local.execution", key, request_digest, execution_id, IdempotencyStatus.FAILED, None, error_code, created_at, now))
        await self._append_terminal_event(execution_id, ExecutionEventType.EXECUTION_FAILED)

    async def _persist_cancelled(self, execution_id: str, key: str, request_digest: str, created_at: datetime, reason: str) -> None:
        current = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        terminal = _terminal_execution(current, ExecutionStatus.CANCELLED, now, error_code=reason)
        result = ResultRecord(execution_id, self.project.project_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(ExecutionTerminalCommit(execution_id, current.snapshot_revision, terminal, result))
        status = await self._persistence_runtime.persistence.idempotency.get("local.execution", key, tenant_id=self.project.project_id)
        if status is not None and status.status not in {IdempotencyStatus.CANCELLED, IdempotencyStatus.COMPLETED, IdempotencyStatus.FAILED}:
            await self._persistence_runtime.persistence.idempotency.compare_and_swap("local.execution", key, expected_status=status.status, next_record=IdempotencyRecord(self.project.project_id, "local.execution", key, request_digest, execution_id, IdempotencyStatus.CANCELLED, None, reason, created_at, now))
        await self._append_terminal_event(execution_id, ExecutionEventType.EXECUTION_CANCELLED)

    async def _append_terminal_event(self, execution_id: str, event_type: ExecutionEventType) -> None:
        current = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        expected = current.event_sequence
        await self._persistence_runtime.persistence.events.append(execution_id, tenant_id=self.project.project_id, expected_sequence=expected, event_type=event_type, payload={})

    async def shutdown(self) -> None:
        async with self._lock:
            execution_ids = tuple(self._tasks)
            tasks = tuple(self._tasks.values())
            self._shutdown_ids.update(execution_ids)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("local runtime shutdown: executions=%s", len(execution_ids))

    async def _reconcile_restarted_executions(self) -> None:
        for record in await self._records.list():
            if record.status != "CANCELLED" or record.stop_reason != "PROCESS_RESTARTED":
                continue
            execution = await self._persistence_runtime.persistence.executions.get(
                record.execution_id,
                tenant_id=self.project.project_id,
            )
            if execution is None or execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                continue
            now = datetime.now(timezone.utc)
            terminal = _terminal_execution(execution, ExecutionStatus.CANCELLED, now, error_code="PROCESS_RESTARTED")
            result = ResultRecord(record.execution_id, self.project.project_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now)
            await self._persistence_runtime.persistence.results.commit_terminal(ExecutionTerminalCommit(record.execution_id, execution.snapshot_revision, terminal, result))
            if record.idempotency_key is not None:
                idem = await self._persistence_runtime.persistence.idempotency.get("local.execution", record.idempotency_key, tenant_id=self.project.project_id)
                if idem is not None and idem.status not in {IdempotencyStatus.COMPLETED, IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED}:
                    await self._persistence_runtime.persistence.idempotency.compare_and_swap(
                        "local.execution",
                        record.idempotency_key,
                        expected_status=idem.status,
                        next_record=IdempotencyRecord(self.project.project_id, "local.execution", record.idempotency_key, idem.request_digest, record.execution_id, IdempotencyStatus.CANCELLED, None, "PROCESS_RESTARTED", idem.created_at, now),
                    )
            current = await self._persistence_runtime.persistence.executions.get(record.execution_id, tenant_id=self.project.project_id)
            if current is not None:
                await self._persistence_runtime.persistence.events.append(record.execution_id, tenant_id=self.project.project_id, expected_sequence=current.event_sequence, event_type=ExecutionEventType.EXECUTION_CANCELLED, payload={"reason": "PROCESS_RESTARTED"})

    async def open_session(self, session_id: str, *, cwd: "str | Path | None" = None, binding_digest: "str | None" = None) -> LocalSession:
        _validate_session_id(session_id)
        await self._records.initialize()
        await self._persistence_runtime.initialize()
        await self._reconcile_restarted_executions()
        existing = self._sessions.get(session_id)
        if existing is not None:
            if binding_digest is not None:
                persisted = await self._persistence_runtime.persistence.sessions.get(
                    session_id,
                    tenant_id=self.project.project_id,
                )
                if persisted is not None and persisted.binding_digest != binding_digest:
                    raise LinktoolsAIError(ErrorCode.SESSION_BINDING_MISMATCH)
            return existing
        session_cwd = Path(cwd or self.project.root).expanduser().resolve()
        try:
            session_cwd.relative_to(self.project.root)
        except ValueError as error:
            raise ValueError("session cwd must be inside the project root") from error
        session = LocalSession(session_id, session_cwd)
        session_file = self._session_file(session_id)
        if session_file.exists():
            await asyncio.to_thread(ModelMessagesTypeAdapter.validate_json, await asyncio.to_thread(session_file.read_bytes))
        digest = binding_digest or canonical_sha256({"agent_id": None})
        persisted = await self._persistence_runtime.persistence.sessions.get(session_id, tenant_id=self.project.project_id)
        if persisted is not None:
            if persisted.binding_digest != digest:
                raise LinktoolsAIError(ErrorCode.SESSION_BINDING_MISMATCH)
            if persisted.cwd is not None:
                persisted_cwd = Path(persisted.cwd).resolve()
                try:
                    persisted_cwd.relative_to(self.project.root)
                except ValueError as error:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                session_cwd = persisted_cwd
            session = LocalSession(session_id, session_cwd, persisted.revision)
        else:
            now = datetime.now(timezone.utc)
            record = SessionRecord(session_id, self.project.project_id, self.project.project_id, digest, SessionStatus.OPEN, 0, 0, session_cwd.as_posix(), {}, now, now, None)
            await self._persistence_runtime.persistence.sessions.create(record)
            session = LocalSession(session_id, session_cwd)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = session
        logger.info(
            "local session opened: session=%s cwd=%s storage=%s",
            session_id,
            session.cwd,
            self.project.storage_root,
        )
        return session

    async def list_sessions(self, *, cwd: "str | None" = None) -> "tuple[LocalSession, ...]":
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        values = []
        for path in sorted(self.sessions_root.glob("*.json")):
            session_id = path.stem
            try:
                await self.open_session(session_id)
            except ValueError:
                continue
            session = self._sessions[session_id]
            if cwd is None or session.cwd.as_posix() == Path(cwd).expanduser().resolve().as_posix():
                values.append(session)
        return tuple(values)

    async def fork_session(self, source_id: str, *, cwd: "str | None" = None) -> LocalSession:
        source = await self.open_session(source_id)
        source_record = await self._persistence_runtime.persistence.sessions.get(
            source_id,
            tenant_id=self.project.project_id,
        )
        if source_record is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        target = await self.open_session(
            uuid4().hex,
            cwd=cwd or source.cwd,
            binding_digest=source_record.binding_digest,
        )
        source_messages = self._load_history(source.session_id)
        await self._save_history(target.session_id, source_messages, [])
        return target

    async def close_session(self, session_id: str, *, force: bool = False) -> None:
        await self._persistence_runtime.initialize()
        record = await self._persistence_runtime.persistence.sessions.get(
            session_id,
            tenant_id=self.project.project_id,
        )
        if record is None:
            raise LinktoolsAIError(ErrorCode.SESSION_NOT_FOUND)
        if record.status is SessionStatus.CLOSED:
            self._sessions.pop(session_id, None)
            return
        async with self._lock:
            active = bool(self._session_runs.get(session_id))
        if active and not force:
            raise LinktoolsAIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
        if record.status is SessionStatus.CLEANUP_REQUIRED and not force:
            raise LinktoolsAIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
        if record.status is SessionStatus.OPEN:
            now = datetime.now(timezone.utc)
            record = await self._persistence_runtime.persistence.sessions.compare_and_swap(
                session_id,
                tenant_id=self.project.project_id,
                expected_revision=record.revision,
                next_record=SessionRecord(
                    record.session_id,
                    record.tenant_id,
                    record.owner_principal_id,
                    record.binding_digest,
                    SessionStatus.CLOSING,
                    record.revision + 1,
                    record.resource_generation,
                    record.cwd,
                    record.metadata,
                    record.created_at,
                    now,
                    record.closed_at,
                ),
            )
        elif record.status is SessionStatus.CLEANUP_REQUIRED:
            now = datetime.now(timezone.utc)
            record = await self._persistence_runtime.persistence.sessions.compare_and_swap(
                session_id,
                tenant_id=self.project.project_id,
                expected_revision=record.revision,
                next_record=SessionRecord(
                    record.session_id,
                    record.tenant_id,
                    record.owner_principal_id,
                    record.binding_digest,
                    SessionStatus.CLOSING,
                    record.revision + 1,
                    record.resource_generation,
                    record.cwd,
                    record.metadata,
                    record.created_at,
                    now,
                    record.closed_at,
                ),
            )
        if active:
            await self.cancel(session_id)
        persisted_active = await self._persistence_runtime.persistence.executions.list_by_session(
            session_id,
            tenant_id=self.project.project_id,
        )
        if any(item.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} for item in persisted_active):
            now = datetime.now(timezone.utc)
            cleanup = SessionRecord(
                record.session_id,
                record.tenant_id,
                record.owner_principal_id,
                record.binding_digest,
                SessionStatus.CLEANUP_REQUIRED,
                record.revision + 1,
                record.resource_generation,
                record.cwd,
                record.metadata,
                record.created_at,
                now,
                record.closed_at,
            )
            await self._persistence_runtime.persistence.sessions.compare_and_swap(
                session_id,
                tenant_id=self.project.project_id,
                expected_revision=record.revision,
                next_record=cleanup,
            )
            raise LinktoolsAIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
        now = datetime.now(timezone.utc)
        await self._persistence_runtime.persistence.sessions.compare_and_swap(
            session_id,
            tenant_id=self.project.project_id,
            expected_revision=record.revision,
            next_record=SessionRecord(
                record.session_id,
                record.tenant_id,
                record.owner_principal_id,
                record.binding_digest,
                SessionStatus.CLOSED,
                record.revision + 1,
                record.resource_generation,
                record.cwd,
                record.metadata,
                record.created_at,
                now,
                now,
            ),
        )
        logger.info("local session closed: session=%s force=%s", session_id, force)
        self._sessions.pop(session_id, None)

    async def _run(
        self,
        session: LocalSession,
        prompt: str,
        agent_id: "str | None",
        on_text: "TextHandler | None",
        on_event: "EventHandler | None",
        execution_id: str,
        start_committed: asyncio.Event,
    ) -> LocalRunResult:
        await start_committed.wait()
        history = await asyncio.to_thread(self._load_history, session.session_id)
        logger.info("local Agent run started: session=%s history=%s", session.session_id, len(history))
        async def on_agent_event(event: 'dict[str, JsonValue]') -> None:
            if event.get("type") == "text" and on_text is not None:
                await _notify_text(on_text, str(event.get("text", "")))
            if on_event is not None:
                await _notify_event(on_event, event)

        result = await self._runner.run(
            agent_id,
            prompt,
            history,
            session.session_id,
            on_event=on_agent_event,
        )
        messages = result.messages
        run_id = result.run_id
        await self._save_history(session.session_id, messages, history)
        logger.info("local Agent run completed: session=%s run=%s", session.session_id, run_id)
        return LocalRunResult(execution_id, session.session_id, run_id, result.output)

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_root / f"{session_id}.json"

    def _load_history(self, session_id: str) -> "list[ModelMessage]":
        path = self._session_file(session_id)
        if not path.exists():
            return []
        return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))

    async def _save_history(
        self,
        session_id: str,
        messages: "list[ModelMessage]",
        base_history: "list[ModelMessage]",
    ) -> None:
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        target = self._session_file(session_id)
        lock = self._history_locks.setdefault(session_id, asyncio.Lock())
        delta = messages[len(base_history):] if messages[:len(base_history)] == base_history else messages
        async with lock:
            current = await asyncio.to_thread(self._load_history, session_id)
            merged = current + delta
            await asyncio.to_thread(write_bytes_atomic, target, ModelMessagesTypeAdapter.dump_json(merged), fsync=True)

    async def _save_record(
        self,
        execution_id: str,
        session: LocalSession,
        created_at: datetime,
        status: str,
        reason: 'str | None',
        idempotency_key: 'str | None' = None,
    ) -> None:
        await self._records.save(
            LocalExecutionRecord(
                self.project.project_id,
                session.session_id,
                session.session_revision,
                session.cwd.as_posix(),
                execution_id,
                status,
                created_at,
                datetime.now(timezone.utc),
                reason,
                idempotency_key,
            )
        )


def _validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("session id must not contain path separators")


def _terminal_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, error_code: "str | None" = None, *, result_ref: "str | None" = None, result_digest: "str | None" = None) -> ExecutionRecord:
    return ExecutionRecord(record.execution_id, record.tenant_id, record.session_id, record.profile, record.binding_digest, record.parent_execution_id, record.root_execution_id, status, record.snapshot_revision + 1, record.event_sequence, record.trace_sequence, result_ref if result_ref is not None else record.result_ref, result_digest if result_digest is not None else record.result_digest, error_code, record.safe_error_details, record.created_at, now)


def _stable_error(error_code: str | None, fallback: ErrorCode) -> LinktoolsAIError:
    try:
        return LinktoolsAIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return LinktoolsAIError(fallback)


async def _notify_text(handler: TextHandler, text: str) -> None:
    try:
        await handler(text)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("local text observer failed")


async def _notify_event(handler: EventHandler, event: 'dict[str, JsonValue]') -> None:
    try:
        pending = handler(event)
        if pending is not None:
            await pending
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("local event observer failed")


__all__ = ["EventHandler", "LocalAgentRuntime", "LocalRunResult", "LocalSession", "TextHandler"]
