#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace application facade and the in-process execution launcher."""

import asyncio
import hashlib
import secrets
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from linktools.core import environ
from pydantic_ai.models import Model
from pydantic_ai_harness.step_persistence import continue_run, fork_run

from ..agent import WorkspaceAgentResult, WorkspaceAgentRunner
from ..core import ErrorCode, AIError
from ..core import canonical_sha256, step_conversation_id, step_run_id
from ..core import JsonValue
from ..core import TenantAuthorizationPolicy
from ..core import ExecutionEventType, ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, SessionStatus, StopReason
from ..runtime import BlobRef, ExecutionRecord, ExecutionTerminalCommit, IdempotencyTerminalUpdate, ResultRecord, RuntimeBackend, SessionHeadAdvance
from ..runtime import CancelEffectOutcome
from ..runtime import (
    CancelExecutionRequest,
    CreateSessionRequest,
    ExecutionRequest,
    ForkSessionRequest,
    ListSessionRequest,
    ResumeSessionRequest,
    RuntimeServices,
    SessionView,
)
from ._assembly import (
    RuntimePersistenceConfig,
    RuntimeResources,
    build_runtime_services,
    open_runtime_resources,
)
from ..adapter import StepExecutionHistoryReader
from ..storage import FilesystemWriterLock
from ..workspace import Workspace, trusted_workspace_principal


_logger = environ.get_logger("ai.app.workbench")


class TextHandler(Protocol):
    async def __call__(self, text: str) -> None: ...


class EventHandler(Protocol):
    def __call__(self, event: dict[str, JsonValue]) -> "Awaitable[None] | None": ...


@dataclass(frozen=True, slots=True)
class WorkspaceSession:
    session_id: str
    cwd: Path
    revision: int


@dataclass(frozen=True, slots=True)
class WorkspaceRunResult:
    execution_id: str
    session_id: str
    run_id: str
    output: str


@dataclass(frozen=True, slots=True)
class _LaunchContext:
    agent_id: str | None
    on_text: TextHandler | None
    on_event: EventHandler | None


class WorkspaceExecutionLauncher:
    """Own process-local agent tasks and submit only Runtime atomic commands."""

    def __init__(self, workspace: Workspace, runner: WorkspaceAgentRunner, resources: RuntimeResources) -> None:
        self._workspace = workspace
        self._runner = runner
        self._resources = resources
        self._contexts: dict[tuple[str, str, str], _LaunchContext] = {}
        self._tasks: dict[str, asyncio.Task[WorkspaceAgentResult]] = {}
        self._results: dict[str, WorkspaceAgentResult] = {}
        self._accepting = True

    def register_context(
        self,
        tenant_id: str,
        session_id: str,
        idempotency_key: str,
        context: _LaunchContext,
    ) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        key = (tenant_id, session_id, hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest())
        existing = self._contexts.get(key)
        if existing is not None and existing != context:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._contexts[key] = context

    def active_execution_ids(self) -> tuple[str, ...]:
        """Return the stable snapshot of process-local execution handles."""
        return tuple(sorted(self._tasks))

    def release_context(self, tenant_id: str, session_id: str, idempotency_key: str) -> None:
        key = (tenant_id, session_id, hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest())
        self._contexts.pop(key, None)

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        if not self._accepting or execution.session_id is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        key = (execution.tenant_id, execution.session_id, hashlib.sha256(str(request.idempotency_key).encode("utf-8")).hexdigest())
        context = self._contexts.get(key)
        if context is None:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        existing = self._tasks.get(execution.execution_id)
        if existing is not None:
            return
        task = asyncio.create_task(self._execute(request, execution, context))
        self._tasks[execution.execution_id] = task
        _logger.info("workspace launch registered: execution=%s segment=%s", execution.execution_id, execution.agent_run_sequence)

    async def cancel(self, execution: ExecutionRecord) -> "CancelEffectOutcome":
        task = self._tasks.get(execution.execution_id)
        if task is None:
            _logger.info("workspace cancellation effect unknown: execution=%s reason=task_missing", execution.execution_id)
            return CancelEffectOutcome.UNKNOWN
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        outcome = CancelEffectOutcome.CONFIRMED if task.cancelled() else CancelEffectOutcome.UNKNOWN
        _logger.info("workspace cancellation effect resolved: execution=%s outcome=%s", execution.execution_id, outcome.value)
        return outcome

    async def wait(self, execution_id: str) -> WorkspaceAgentResult:
        task = self._tasks.get(execution_id)
        if task is not None:
            try:
                result = await asyncio.shield(task)
            finally:
                self._tasks.pop(execution_id, None)
                self._results.pop(execution_id, None)
            return result
        result = self._results.pop(execution_id, None)
        if result is not None:
            return result
        raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)

    async def load_result(self, execution_id: str, *, tenant_id: str) -> WorkspaceAgentResult:
        record = await self._resources.domain.executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        result = await self._resources.domain.results.get(execution_id, tenant_id=tenant_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        output = ""
        if result.payload_ref is not None:
            blob = await self._resources.domain.blobs.stat(
                BlobRef(tenant_id, result.payload_ref, 0, ""),
                tenant_id=tenant_id,
            )
            if blob is not None:
                chunks = bytearray()
                async for chunk in self._resources.domain.blobs.open(blob, tenant_id=tenant_id):
                    chunks.extend(chunk)
                output = bytes(chunks).decode("utf-8")
        run_id = await self._durable_run_id(record, tenant_id=tenant_id)
        return WorkspaceAgentResult(run_id, output, [])

    async def _durable_run_id(self, record: ExecutionRecord, *, tenant_id: str) -> str:
        after_sequence = 0
        while True:
            page = await self._resources.domain.events.list(
                record.execution_id,
                tenant_id=tenant_id,
                after_sequence=after_sequence,
                limit=200,
            )
            for event in page.items:
                if event.event_type is not ExecutionEventType.EXECUTION_SUCCEEDED or not isinstance(event.payload, Mapping):
                    continue
                value = event.payload.get("run_id")
                if isinstance(value, str) and value:
                    return value
            if page.next_cursor is None or not page.items:
                break
            after_sequence = page.items[-1].sequence
        return step_run_id(
            namespace=self._workspace.workspace_id,
            tenant_id=tenant_id,
            execution_id=record.execution_id,
            segment_sequence=record.agent_run_sequence,
        )

    async def shutdown(self) -> None:
        self._accepting = False
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._contexts.clear()
        self._results.clear()

    async def reconcile(self) -> None:
        """Cancel claimed segments proven to have no durable Harness run."""
        sessions = await self._resources.domain.sessions.list(tenant_id=self._workspace.workspace_id)
        for session in sessions:
            executions = await self._resources.domain.executions.list_by_session(session.session_id, tenant_id=session.tenant_id)
            for execution in executions:
                if execution.status is not ExecutionStatus.STARTED or execution.agent_run_sequence < 1:
                    continue
                run_id = step_run_id(namespace=self._workspace.workspace_id, tenant_id=execution.tenant_id, execution_id=execution.execution_id, segment_sequence=execution.agent_run_sequence)
                if await self._resources.steps.get_run(run_id=run_id) is not None:
                    continue
                current = await self._resources.domain.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
                if current is None or current.status is not ExecutionStatus.STARTED:
                    continue
                now = datetime.now(timezone.utc)
                terminal = _terminal_record(current, ExecutionStatus.CANCELLED, now, error_code=ErrorCode.EXECUTION_CANCELLED.value, safe_error_details={"reason": "PROCESS_RESTARTED_BEFORE_AGENT"})
                idempotency = await self._terminal_idempotency(current, IdempotencyStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value)
                await self._resources.domain.results.commit_terminal(
                    ExecutionTerminalCommit(
                        expected_revision=current.revision,
                        expected_event_sequence=current.event_sequence,
                        execution=terminal,
                        result=ResultRecord(current.execution_id, current.tenant_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now),
                        terminal_event_type=ExecutionEventType.EXECUTION_CANCELLED,
                        terminal_event_payload={"reason": "PROCESS_RESTARTED_BEFORE_AGENT"},
                        idempotency=idempotency,
                    )
                )
                _logger.warning("reconciled abandoned workspace execution: execution=%s", execution.execution_id)

    async def _execute(self, request: ExecutionRequest, execution: ExecutionRecord, context: _LaunchContext) -> WorkspaceAgentResult:
        if execution.session_id is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        segment_sequence = execution.agent_run_sequence
        conversation_id = step_conversation_id(
            namespace=self._workspace.workspace_id,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
        )
        run_id = step_run_id(
            namespace=self._workspace.workspace_id,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
            segment_sequence=segment_sequence,
        )
        history = await self._history(execution)

        async def on_event(event: dict[str, JsonValue]) -> None:
            if context.on_text is not None and event.get("type") == "text":
                pending = context.on_text(str(event.get("text", "")))
                if pending is not None:
                    await pending
            if context.on_event is not None:
                pending = context.on_event(event)
                if pending is not None:
                    await pending

        try:
            result = await self._runner.run(
                context.agent_id,
                request.prompt,
                history,
                conversation_id,
                step_store=self._resources.steps,
                step_run_id=run_id,
                segment_sequence=segment_sequence,
                on_event=on_event,
            )
            await self._commit_success(execution, result)
            self._results[execution.execution_id] = result
            _logger.info("workspace execution completed: execution=%s run=%s", execution.execution_id, result.run_id)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._commit_failure(execution, error)
            _logger.error("workspace execution failed: execution=%s", execution.execution_id, exc_info=environ.debug)
            raise
        finally:
            key = (execution.tenant_id, execution.session_id, hashlib.sha256(str(request.idempotency_key).encode("utf-8")).hexdigest())
            self._contexts.pop(key, None)

    async def _history(self, execution: ExecutionRecord) -> list[object]:
        if execution.base_execution_id is None:
            return []
        base = await self._resources.domain.executions.get(execution.base_execution_id, tenant_id=execution.tenant_id)
        if base is None or base.agent_run_sequence == 0:
            return []
        base_run_id = step_run_id(
            namespace=self._workspace.workspace_id,
            tenant_id=execution.tenant_id,
            execution_id=base.execution_id,
            segment_sequence=base.agent_run_sequence,
        )
        try:
            session = await self._resources.domain.sessions.get(execution.session_id or "", tenant_id=execution.tenant_id)
            if execution.lineage_kind is ExecutionLineageKind.FORK or (session is not None and base.session_id != session.session_id):
                return list(await fork_run(self._resources.steps, run_id=base_run_id))
            return list(await continue_run(self._resources.steps, run_id=base_run_id))
        except LookupError as error:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error

    async def _commit_success(self, execution: ExecutionRecord, result: WorkspaceAgentResult) -> None:
        now = datetime.now(timezone.utc)
        blob = await self._resources.domain.blobs.put_bytes(tenant_id=execution.tenant_id, data=result.output.encode("utf-8"))
        current = await self._resources.domain.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal = _terminal_record(current, ExecutionStatus.SUCCEEDED, now, result_ref=blob.digest, result_digest=blob.digest)
        idempotency = await self._terminal_idempotency(current, IdempotencyStatus.COMPLETED, blob.digest, None)
        head = None
        if current.session_id is not None:
            if current.lineage_kind is ExecutionLineageKind.SESSION_RESUME:
                expected_head_execution_id = current.base_execution_id
            elif current.lineage_kind is ExecutionLineageKind.RETRY:
                if current.source_execution_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                expected_head_execution_id = current.source_execution_id
            else:
                expected_head_execution_id = None
            if current.lineage_kind in {ExecutionLineageKind.SESSION_RESUME, ExecutionLineageKind.RETRY}:
                head = SessionHeadAdvance(current.session_id, expected_head_execution_id, current.execution_id)
        await self._resources.domain.results.commit_terminal(
            ExecutionTerminalCommit(
                expected_revision=current.revision,
                expected_event_sequence=current.event_sequence,
                execution=terminal,
                result=ResultRecord(current.execution_id, current.tenant_id, ExecutionStatus.SUCCEEDED, "text", 1, "text", blob.digest, blob.digest, StopReason.END_TURN, 0, 0, 0, now),
                terminal_event_type=ExecutionEventType.EXECUTION_SUCCEEDED,
                terminal_event_payload={"run_id": result.run_id},
                idempotency=idempotency,
                session_head=head,
            )
        )

    async def _commit_failure(self, execution: ExecutionRecord, error: Exception) -> None:
        current = await self._resources.domain.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        error_code = ErrorCode.EXECUTION_FAILED.value
        terminal = _terminal_record(current, ExecutionStatus.FAILED, now, error_code=error_code)
        idempotency = await self._terminal_idempotency(current, IdempotencyStatus.FAILED, None, error_code)
        await self._resources.domain.results.commit_terminal(
            ExecutionTerminalCommit(
                expected_revision=current.revision,
                expected_event_sequence=current.event_sequence,
                execution=terminal,
                result=ResultRecord(current.execution_id, current.tenant_id, ExecutionStatus.FAILED, "text", 1, "text", None, None, StopReason.ERROR, 0, 0, 0, now),
                terminal_event_type=ExecutionEventType.EXECUTION_FAILED,
                terminal_event_payload={"error_code": error_code},
                idempotency=idempotency,
            )
        )

    async def _terminal_idempotency(self, execution: ExecutionRecord, status: IdempotencyStatus, result_digest: str | None, error_code: str | None) -> IdempotencyTerminalUpdate | None:
        records = await self._resources.domain.idempotency.list_by_execution(execution.execution_id, tenant_id=execution.tenant_id)
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not records:
            return None
        identity = records[0]
        return IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, status, identity.request_digest, result_digest, error_code)


class WorkspaceAgentRuntime:
    """Translate workspace operations into Runtime service requests."""

    def __init__(self, workspace: Workspace, *, runner: WorkspaceAgentRunner, resources: RuntimeResources, services: RuntimeServices, launcher: WorkspaceExecutionLauncher) -> None:
        self.workspace = workspace
        self._runner = runner
        self._resources = resources
        self._services = services
        self._launcher = launcher
        self._principal = trusted_workspace_principal(workspace.workspace_id)
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_complete = False

    async def open_session(self, session_id: str, *, cwd: str | Path | None = None, agent_id: str | None = None) -> WorkspaceSession:
        async with self._lifecycle_lock:
            self._ensure_open()
            return await self._open_session(session_id, cwd=cwd, agent_id=agent_id)

    async def _open_session(self, session_id: str, *, cwd: str | Path | None = None, agent_id: str | None = None) -> WorkspaceSession:
        _validate_identifier(session_id)
        normalized_cwd = self._normalize_cwd(cwd)
        binding_digest = await self._runner.binding_digest(agent_id)
        try:
            view = await self._services.session.get(session_id, principal=self._principal)
        except AIError as error:
            if error.code is not ErrorCode.AUTHORIZATION_DENIED:
                raise
            view = await self._services.session.create(
                binding_digest,
                CreateSessionRequest(
                    self._principal,
                    session_id,
                    canonical_sha256(["workspace-session-create", self.workspace.workspace_id, session_id, binding_digest, normalized_cwd.as_posix()]),
                    normalized_cwd.as_posix(),
                ),
            )
        if view.status is not SessionStatus.OPEN or view.binding_digest != binding_digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        if view.cwd is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        persisted_cwd = self._normalize_cwd(view.cwd)
        if cwd is not None and persisted_cwd != normalized_cwd:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if persisted_cwd != Path(view.cwd):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return WorkspaceSession(view.session_id, persisted_cwd, view.revision)

    async def run(self, session_id: str, prompt: str, *, cwd: str | Path | None = None, agent_id: str | None = None, idempotency_key: str | None = None, on_text: TextHandler | None = None, on_event: EventHandler | None = None) -> WorkspaceRunResult:
        if not prompt.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        effective_key = idempotency_key or secrets.token_urlsafe(32)
        binding_digest = await self._runner.binding_digest(agent_id)
        _validate_identifier(session_id)
        try:
            async with self._lifecycle_lock:
                self._ensure_open()
                self._launcher.register_context(self._principal.tenant_id, session_id, effective_key, _LaunchContext(agent_id, on_text, on_event))
                session = await self._open_session(session_id, cwd=cwd, agent_id=agent_id)
                handle = await self._services.session.resume(binding_digest, session_id, ResumeSessionRequest(self._principal, prompt, effective_key))
            try:
                result = await self._launcher.wait(handle.execution_id)
            except AIError as error:
                if error.code is not ErrorCode.EXECUTION_START_UNKNOWN:
                    raise
                result = await self._launcher.load_result(handle.execution_id, tenant_id=self._principal.tenant_id)
            return WorkspaceRunResult(handle.execution_id, session.session_id, result.run_id, result.output)
        finally:
            self._launcher.release_context(self._principal.tenant_id, session_id, effective_key)

    async def cancel(self, session_id: str) -> bool:
        async with self._lifecycle_lock:
            self._ensure_open()
            loaded = await self._services.session.load(session_id, principal=self._principal)
            if not loaded.active_execution_ids:
                return False
            for execution_id in loaded.active_execution_ids:
                await self._cancel_execution(execution_id)
            return True

    async def list_sessions(self, *, cwd: str | Path | None = None) -> tuple[WorkspaceSession, ...]:
        normalized = None if cwd is None else self._normalize_cwd(cwd)
        result: list[WorkspaceSession] = []
        for view in await self._session_views():
            if view.cwd is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            persisted = self._normalize_cwd(view.cwd)
            if persisted != Path(view.cwd):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if normalized is None or normalized == persisted:
                result.append(WorkspaceSession(view.session_id, persisted, view.revision))
        return tuple(result)

    async def fork_session(self, source_id: str, *, cwd: str | Path | None = None, agent_id: str | None = None) -> WorkspaceSession:
        async with self._lifecycle_lock:
            self._ensure_open()
            source = await self._open_session(source_id, agent_id=agent_id)
            target_id = str(uuid4())
            target_cwd = None if cwd is None else self._normalize_cwd(cwd).as_posix()
            binding_digest = await self._runner.binding_digest(agent_id)
            view = await self._services.session.fork(binding_digest, source_id, ForkSessionRequest(self._principal, target_id, secrets.token_urlsafe(32), target_cwd))
            return WorkspaceSession(view.session_id, self._normalize_cwd(view.cwd or source.cwd), view.revision)

    async def close_session(self, session_id: str, *, force: bool = False) -> None:
        async with self._lifecycle_lock:
            self._ensure_open()
            request_id = canonical_sha256(["workspace-close", self.workspace.workspace_id, session_id, force])
            from ..runtime import CloseSessionRequest
            await self._services.session.close(session_id, CloseSessionRequest(self._principal, request_id, force))

    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            if self._shutdown_complete:
                return
            self._closed = True
            failures: list[Exception] = []
            for execution_id in self._launcher.active_execution_ids():
                try:
                    await self._cancel_execution(execution_id)
                except Exception as error:
                    failures.append(error)
            try:
                for session in await self._session_views():
                    for execution_id in session.active_execution_ids:
                        try:
                            await self._cancel_execution(execution_id)
                        except Exception as error:
                            failures.append(error)
                remaining = tuple(execution_id for session in await self._session_views() for execution_id in session.active_execution_ids)
                if remaining or failures:
                    raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
            finally:
                await self._launcher.shutdown()
            self._shutdown_complete = True

    async def _session_views(self) -> tuple[SessionView, ...]:
        cursor = None
        values: list[SessionView] = []
        while True:
            page = await self._services.session.list(ListSessionRequest(self._principal, cursor, 200))
            values.extend(page.items)
            if page.next_cursor is None:
                return tuple(values)
            if page.next_cursor == cursor or (not page.items and page.next_cursor is not None):
                raise AIError(ErrorCode.CURSOR_INVALID)
            cursor = page.next_cursor

    async def _cancel_execution(self, execution_id: str) -> None:
        request_id = canonical_sha256(["workspace-cancel", self.workspace.workspace_id, execution_id])
        await self._services.execution.cancel(execution_id, CancelExecutionRequest(self._principal, request_id, True))

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _normalize_cwd(self, cwd: str | Path | None) -> Path:
        candidate = Path(cwd or self.workspace.root).expanduser().resolve()
        try:
            candidate.relative_to(self.workspace.root)
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        return candidate


@asynccontextmanager
async def open_workspace_runtime(workspace: Workspace, *, config: RuntimePersistenceConfig | None = None, runner: WorkspaceAgentRunner | None = None, model: "str | Model | None" = None, base_url: str | None = None, api_key: str | None = None, grant_key: bytes | None = None) -> AsyncIterator[WorkspaceAgentRuntime]:
    persistence_config = config or RuntimePersistenceConfig.filesystem(str(workspace.storage_root), workspace_id=workspace.workspace_id)
    authorization = TenantAuthorizationPolicy()
    key = grant_key or hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()
    local_lock = None
    if persistence_config.backend is RuntimeBackend.SQLITE:
        path = Path(str(persistence_config.location))
        local_lock = FilesystemWriterLock(path.with_name(f"{path.name}.local.lock"))
        await local_lock.acquire()
    try:
        workspace_runner = runner or WorkspaceAgentRunner(workspace.root, workspace.config, model=model, base_url=base_url, api_key=api_key)
        async with open_runtime_resources(persistence_config) as resources:
            launcher = WorkspaceExecutionLauncher(workspace, workspace_runner, resources)
            await launcher.reconcile()
            history_reader = StepExecutionHistoryReader(persistence_config.namespace, resources.domain, resources.steps)
            services = build_runtime_services(
                resources.domain,
                authorization,
                grant_key=key,
                history_reader=history_reader,
                schema_digest=resources.domain.atomic_domain_id,
                execution_launcher=launcher,
            )
            runtime = WorkspaceAgentRuntime(workspace, runner=workspace_runner, resources=resources, services=services, launcher=launcher)
            _logger.info("workspace runtime opened: workspace=%s backend=%s", workspace.workspace_id, persistence_config.backend)
            try:
                yield runtime
            finally:
                await runtime.shutdown()
    finally:
        if local_lock is not None:
            await local_lock.release()


def _terminal_record(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, result_ref: str | None = None, result_digest: str | None = None, error_code: str | None = None, safe_error_details: dict[str, JsonValue] | None = None) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=record.execution_id,
        tenant_id=record.tenant_id,
        session_id=record.session_id,
        binding_digest=record.binding_digest,
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        source_execution_id=record.source_execution_id,
        base_execution_id=record.base_execution_id,
        lineage_kind=record.lineage_kind,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + 1,
        agent_run_sequence=record.agent_run_sequence,
        result_ref=result_ref,
        result_digest=result_digest,
        error_code=error_code,
        safe_error_details={} if safe_error_details is None else safe_error_details,
        created_at=record.created_at,
        updated_at=now,
    )


def _validate_identifier(value: str) -> None:
    if not value.strip() or value in {".", ".."} or any(char in value for char in "/\\\x00"):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


__all__ = ["EventHandler", "TextHandler", "WorkspaceAgentRuntime", "WorkspaceExecutionLauncher", "WorkspaceRunResult", "WorkspaceSession", "open_workspace_runtime"]
