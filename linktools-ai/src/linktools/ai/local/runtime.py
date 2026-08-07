#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Agent runtime used by the CLI and ACP composition roots."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from linktools.core import environ

from pydantic_ai_harness.step_persistence import StepStore, continue_run, fork_run

from ..agent.runner import LocalAgentRunner
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256, idempotency_key_hash, step_conversation_id, step_run_id
from ..core.json import JsonValue
from ..core.value import ExecutionEventType, ExecutionLineageKind, ExecutionProfile, ExecutionStatus, IdempotencyStatus, SessionStatus, StopReason
from ..runtime.persistence import BlobRef, ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionTerminalCommit, IdempotencyRecord, IdempotencyTerminalUpdate, ResultRecord, RuntimePersistence, SessionHeadAdvance, SessionRecord
from ..runtime.services import ExecutionRequest
from .project import LocalProject
from .persistence import FileRuntime, build_file_runtime
from .step import DurableFileStepStore

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


class LocalExecutionLauncher:
    """Execute a Runtime execution in the current process and commit its result."""

    def __init__(self, project: LocalProject, runner: LocalAgentRunner, persistence: RuntimePersistence, step_store: StepStore) -> None:
        self._project = project
        self._runner = runner
        self._persistence = persistence
        self._step_store = step_store
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        if execution.session_id is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        task = asyncio.create_task(self._execute(request, execution))
        self._tasks[execution.execution_id] = task
        try:
            await asyncio.shield(task)
        finally:
            self._tasks.pop(execution.execution_id, None)

    async def cancel(self, execution: ExecutionRecord) -> None:
        task = self._tasks.get(execution.execution_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _execute(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        conversation_id = step_conversation_id(namespace=self._project.project_id, tenant_id=execution.tenant_id, execution_id=execution.execution_id)
        run_id = step_run_id(namespace=self._project.project_id, tenant_id=execution.tenant_id, execution_id=execution.execution_id, segment_sequence=1)
        if execution.base_execution_id is not None:
            base = await self._persistence.executions.get(execution.base_execution_id, tenant_id=execution.tenant_id)
            if base is not None and base.agent_run_sequence > 0:
                base_run_id = step_run_id(namespace=self._project.project_id, tenant_id=execution.tenant_id, execution_id=base.execution_id, segment_sequence=base.agent_run_sequence)
                try:
                    history = await (fork_run(self._step_store, run_id=base_run_id) if execution.lineage_kind is ExecutionLineageKind.FORK else continue_run(self._step_store, run_id=base_run_id))
                except LookupError as error:
                    raise LinktoolsAIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
            else:
                history = []
        else:
            history = []
        try:
            result = await self._runner.run(None, request.prompt, history, conversation_id, step_store=self._step_store, step_run_id=run_id, segment_sequence=1)
            blob = await self._persistence.blobs.put_bytes(tenant_id=execution.tenant_id, data=result.output.encode("utf-8"))
            current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = datetime.now(timezone.utc)
            result_record = ResultRecord(execution.execution_id, execution.tenant_id, ExecutionStatus.SUCCEEDED, "none", 1, "none", blob.digest, blob.digest, StopReason.END_TURN, 0, 0, 0, now)
            await self._persistence.results.commit_terminal(await _terminal_commit(self._persistence, current, result_record, ExecutionEventType.EXECUTION_SUCCEEDED, now, advance_head=True))
            logger.info("local execution launcher completed: execution=%s step=%s", execution.execution_id, run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                now = datetime.now(timezone.utc)
                result_record = ResultRecord(execution.execution_id, execution.tenant_id, ExecutionStatus.FAILED, "none", 1, "none", None, None, StopReason.ERROR, 0, 0, 0, now)
                await self._persistence.results.commit_terminal(await _terminal_commit(self._persistence, current, result_record, ExecutionEventType.EXECUTION_FAILED, now, error_code=ErrorCode.EXECUTION_FAILED.value))
            raise


class LocalAgentRuntime:
    """Run one local Agent while keeping session history under the project root."""

    def __init__(
        self,
        project: LocalProject,
        *,
        runner: LocalAgentRunner,
        persistence: "RuntimePersistence | None" = None,
        step_store: "StepStore | None" = None,
    ) -> None:
        self.project = project
        self._runner = runner
        self._sessions: "dict[str, LocalSession]" = {}
        self._tasks: "dict[str, asyncio.Task[LocalRunResult]]" = {}
        self._session_runs: "dict[str, set[str]]" = {}
        self._idempotency_tasks: "dict[tuple[str, str], asyncio.Task[LocalRunResult]]" = {}
        self._shutdown_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._persistence_runtime = persistence or build_file_runtime(str(project.storage_root), project_id=project.project_id, local_tenant_id=project.project_id)
        if step_store is None:
            self._step_store = DurableFileStepStore(project.storage_root, project.project_id, writer_lock=self._persistence_runtime.writer_lock if isinstance(self._persistence_runtime, FileRuntime) else None)
        else:
            self._step_store = step_store

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
        await self._persistence_runtime.initialize()
        if isinstance(self._step_store, DurableFileStepStore):
            await self._step_store.initialize()
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
        effective_key_hash = idempotency_key_hash(effective_key)
        request_digest = canonical_sha256({"prompt": prompt, "agent_id": agent_id})
        source_execution_id = persisted_session.head_execution_id
        lineage_kind = ExecutionLineageKind.SESSION_RESUME if source_execution_id is not None else ExecutionLineageKind.RUN
        async with self._lock:
            existing = await self._persistence_runtime.persistence.idempotency.get("local.execution", effective_key_hash, tenant_id=self.project.project_id)
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
                pending = ExecutionRecord(execution_id, self.project.project_id, session_id, ExecutionProfile.LOCAL_CODING, canonical_sha256({"agent_id": agent_id}), None, execution_id, ExecutionStatus.PENDING_START, 0, 0, None, None, None, {}, now, now, source_execution_id, source_execution_id, lineage_kind, 0)
                reservation = await self._persistence_runtime.persistence.executions.reserve_start(ExecutionStartReservation(pending, IdempotencyRecord(self.project.project_id, "local.execution", effective_key_hash, request_digest, execution_id, IdempotencyStatus.RESERVED, None, None, now, now)))
                if not reservation.created:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                task: asyncio.Task[LocalRunResult] | None = None
                start_committed = asyncio.Event()
                try:
                    claimed = await self._persistence_runtime.persistence.executions.claim_start(ExecutionStartClaim(execution_id, self.project.project_id, 0, 0, "local.execution", effective_key_hash, request_digest, now))
                    task = asyncio.create_task(self._run(session, prompt, agent_id, on_text, on_event, claimed.execution_id, start_committed))
                    self._tasks[execution_id] = task
                    self._session_runs.setdefault(session_id, set()).add(execution_id)
                    self._idempotency_tasks[(session_id, effective_key)] = task
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
            await self._persist_success(completed, effective_key, request_digest, created_at)
            return completed
        except asyncio.CancelledError:
            status = "PROCESS_SHUTDOWN" if execution_id in self._shutdown_ids else "CANCELLED"
            reason = "PROCESS_SHUTDOWN" if status == "PROCESS_SHUTDOWN" else "cancelled"
            await self._persist_cancelled(execution_id, effective_key, request_digest, created_at, reason)
            raise
        except Exception as error:
            error_digest = canonical_sha256({"type": type(error).__name__})
            logger.warning("local execution failed: execution=%s error_digest=%s", execution_id, error_digest)
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
        result = ResultRecord(completed.execution_id, self.project.project_id, ExecutionStatus.SUCCEEDED, "none", 1, "none", blob.digest, blob.digest, StopReason.END_TURN, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(await _terminal_commit(self._persistence_runtime.persistence, current, result, ExecutionEventType.EXECUTION_SUCCEEDED, now, advance_head=True))

    async def _persist_failure(self, execution_id: str, key: str, request_digest: str, session_id: str, created_at: datetime, error_code: str) -> None:
        current = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        result = ResultRecord(execution_id, self.project.project_id, ExecutionStatus.FAILED, "none", 1, "none", None, None, StopReason.ERROR, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(await _terminal_commit(self._persistence_runtime.persistence, current, result, ExecutionEventType.EXECUTION_FAILED, now, error_code=error_code))

    async def _persist_cancelled(self, execution_id: str, key: str, request_digest: str, created_at: datetime, reason: str) -> None:
        current = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        result = ResultRecord(execution_id, self.project.project_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now)
        await self._persistence_runtime.persistence.results.commit_terminal(await _terminal_commit(self._persistence_runtime.persistence, current, result, ExecutionEventType.EXECUTION_CANCELLED, now, error_code=reason))

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

    async def close(self) -> None:
        await self.shutdown()
        if isinstance(self._step_store, DurableFileStepStore):
            await self._step_store.close()
        await self._persistence_runtime.close()

    async def open_session(self, session_id: str, *, cwd: "str | Path | None" = None, binding_digest: "str | None" = None) -> LocalSession:
        _validate_session_id(session_id)
        await self._persistence_runtime.initialize()
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
            record = SessionRecord(session_id, self.project.project_id, self.project.project_id, digest, SessionStatus.OPEN, 0, 0, session_cwd.as_posix(), {}, now, now, None, ExecutionProfile.LOCAL_CODING, None)
            await self._persistence_runtime.persistence.sessions.create(record)
            session = LocalSession(session_id, session_cwd)
        self._sessions[session_id] = session
        logger.info(
            "local session opened: session=%s cwd=%s storage=%s",
            session_id,
            session.cwd,
            self.project.storage_root,
        )
        return session

    async def list_sessions(self, *, cwd: "str | None" = None) -> "tuple[LocalSession, ...]":
        await self._persistence_runtime.initialize()
        records = await self._persistence_runtime.persistence.sessions.list(tenant_id=self.project.project_id, owner_principal_id=self.project.project_id)
        values = []
        for record in records:
            session = await self.open_session(record.session_id)
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
                next_record=replace(record, status=SessionStatus.CLOSING, revision=record.revision + 1, updated_at=now),
            )
        elif record.status is SessionStatus.CLEANUP_REQUIRED:
            now = datetime.now(timezone.utc)
            record = await self._persistence_runtime.persistence.sessions.compare_and_swap(
                session_id,
                tenant_id=self.project.project_id,
                expected_revision=record.revision,
                next_record=replace(record, status=SessionStatus.CLOSING, revision=record.revision + 1, updated_at=now),
            )
        if active:
            await self.cancel(session_id)
        persisted_active = await self._persistence_runtime.persistence.executions.list_by_session(
            session_id,
            tenant_id=self.project.project_id,
        )
        if any(item.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} for item in persisted_active):
            now = datetime.now(timezone.utc)
            cleanup = replace(record, status=SessionStatus.CLEANUP_REQUIRED, revision=record.revision + 1, updated_at=now)
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
            next_record=replace(record, status=SessionStatus.CLOSED, revision=record.revision + 1, updated_at=now, closed_at=now),
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
        conversation_id = step_conversation_id(namespace=self.project.project_id, tenant_id=self.project.project_id, execution_id=execution_id)
        deterministic_step_run_id = step_run_id(namespace=self.project.project_id, tenant_id=self.project.project_id, execution_id=execution_id, segment_sequence=1)
        execution = await self._persistence_runtime.persistence.executions.get(execution_id, tenant_id=self.project.project_id)
        if execution is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_run_id = deterministic_step_run_id
        if execution.base_execution_id is not None:
            base = await self._persistence_runtime.persistence.executions.get(execution.base_execution_id, tenant_id=self.project.project_id)
            if base is None or base.agent_run_sequence < 1 or base.status is not ExecutionStatus.SUCCEEDED:
                raise LinktoolsAIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            if base.agent_run_sequence > 0:
                history_run_id = step_run_id(namespace=self.project.project_id, tenant_id=self.project.project_id, execution_id=base.execution_id, segment_sequence=base.agent_run_sequence)
            try:
                history = await (fork_run(self._step_store, run_id=history_run_id) if execution.lineage_kind is ExecutionLineageKind.FORK else continue_run(self._step_store, run_id=history_run_id))
            except LookupError as error:
                raise LinktoolsAIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        else:
            history = []
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
            conversation_id,
            step_store=self._step_store,
            step_run_id=deterministic_step_run_id,
            segment_sequence=1,
            on_event=on_agent_event,
        )
        run_id = result.run_id
        logger.info("local Agent run completed: session=%s run=%s", session.session_id, run_id)
        return LocalRunResult(execution_id, session.session_id, run_id, result.output)



def _validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("session id must not contain path separators")


async def _terminal_commit(
    persistence: RuntimePersistence,
    current: ExecutionRecord,
    result: ResultRecord,
    event_type: ExecutionEventType,
    now: datetime,
    *,
    advance_head: bool = False,
    error_code: "str | None" = None,
) -> ExecutionTerminalCommit:
    terminal = _terminal_execution(current, result.status, now, error_code=error_code, result_ref=result.payload_ref, result_digest=result.payload_digest)
    records = await persistence.idempotency.list_by_execution(current.execution_id, tenant_id=current.tenant_id)
    if len(records) > 1:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity = records[0] if records else None
    next_status = {
        ExecutionStatus.SUCCEEDED: IdempotencyStatus.COMPLETED,
        ExecutionStatus.FAILED: IdempotencyStatus.FAILED,
        ExecutionStatus.CANCELLED: IdempotencyStatus.CANCELLED,
    }[result.status]
    idempotency = None if identity is None else IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, next_status, identity.request_digest, result.payload_digest, terminal.error_code)
    lineage = ExecutionLineageKind(current.lineage_kind)
    session_head = None
    if advance_head and result.status is ExecutionStatus.SUCCEEDED and current.session_id is not None and lineage in {ExecutionLineageKind.SESSION_RESUME, ExecutionLineageKind.RETRY}:
        session_head = SessionHeadAdvance(current.session_id, current.base_execution_id, current.execution_id)
    return ExecutionTerminalCommit(current.snapshot_revision, current.event_sequence, terminal, result, event_type, {}, idempotency=idempotency, session_head=session_head)


def _terminal_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, error_code: "str | None" = None, *, result_ref: "str | None" = None, result_digest: "str | None" = None) -> ExecutionRecord:
    return ExecutionRecord(record.execution_id, record.tenant_id, record.session_id, record.profile, record.binding_digest, record.parent_execution_id, record.root_execution_id, status, record.snapshot_revision + 1, record.event_sequence + 1, result_ref if result_ref is not None else record.result_ref, result_digest if result_digest is not None else record.result_digest, error_code, record.safe_error_details, record.created_at, now, record.source_execution_id, record.base_execution_id, record.lineage_kind, record.agent_run_sequence)


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


__all__ = ["EventHandler", "LocalAgentRuntime", "LocalExecutionLauncher", "LocalRunResult", "LocalSession", "TextHandler"]
