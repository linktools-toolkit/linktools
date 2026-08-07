#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from ..core import Page, Principal
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
from ..core.principal import AuthorizationAction, AuthorizationPolicy, ResourceRef
from ..core.value import ExecutionEventType, ExecutionProfile, ExecutionStatus, IdempotencyStatus, OperationKind, OperationStatus, ResourceKind, StopReason, TraceKind
from linktools.core import environ
from .persistence import (
    ExecutionEventRecord,
    ExecutionRecord,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    OperationLedgerRecord,
    ResultRecord,
    RuntimePersistence,
)
from ..agent.context import AgentBinding
from .services import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionView,
    ForkExecutionRequest,
    RetryExecutionRequest,
    TraceItem,
    TranscriptItem,
)

_logger = environ.get_logger("ai.runtime.execution")


class ExecutionQueryApi(Protocol):
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...
    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TraceItem]': ...
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TranscriptItem]': ...


class ExecutionApi(ExecutionQueryApi, Protocol):
    async def run(self, request: ExecutionRequest) -> ExecutionHandle: ...
    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle: ...
    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle: ...
    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult: ...


class ExecutionLauncher(Protocol):
    async def start(self, binding: AgentBinding, request: ExecutionRequest, execution: ExecutionRecord) -> None: ...
    async def cancel(self, execution: ExecutionRecord) -> None: ...


class DefaultExecutionService:
    """Apply execution authorization and state transitions before launching work."""

    def __init__(
        self,
        persistence: RuntimePersistence,
        authorization: AuthorizationPolicy,
        *,
        launcher: "ExecutionLauncher | None" = None,
        operation_ids: "Callable[[], str] | None" = None,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)

    async def run(self, binding: AgentBinding, request: ExecutionRequest) -> ExecutionHandle:
        return await self._start(binding, request, scope="execution.run")

    async def run_for_session(self, binding: AgentBinding, session_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not session_id.strip():
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(binding, request, session_id=session_id, scope="session.resume")

    async def _start(self, binding: AgentBinding, request: ExecutionRequest, *, session_id: "str | None" = None, scope: str = "execution.run") -> ExecutionHandle:
        execution_id = self._operation_ids()
        resource = ResourceRef(ResourceKind.EXECUTION, execution_id, request.principal.tenant_id)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXECUTION_RUN, resource)
        key = request.idempotency_key or execution_id
        request_digest = _request_digest(request, binding)
        existing = await self._persistence.idempotency.get(scope, key, tenant_id=request.principal.tenant_id)
        if existing is not None:
            if existing.request_digest != request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.START_UNKNOWN:
                raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.RESERVED:
                raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
            if existing.status is IdempotencyStatus.CANCELLED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            return ExecutionHandle(existing.execution_id, request.requested_profile)
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id,
            request.principal.tenant_id,
            session_id,
            request.requested_profile,
            binding.digest,
            None,
            execution_id,
            ExecutionStatus.PENDING_START,
            0,
            0,
            0,
            None,
            None,
            None,
            {},
            now,
            now,
        )
        await self._persistence.idempotency.reserve(
            IdempotencyRecord(
                request.principal.tenant_id,
                scope,
                key,
                request_digest,
                execution_id,
                IdempotencyStatus.RESERVED,
                None,
                None,
                now,
                now,
            )
        )
        try:
            await self._persistence.executions.create(execution)
            started = _next_execution(execution, ExecutionStatus.STARTED, now)
            await self._persistence.executions.compare_and_swap(execution_id, tenant_id=request.principal.tenant_id, expected_snapshot_revision=0, next_record=started)
            await self._persistence.idempotency.compare_and_swap(
                scope,
                key,
                expected_status=IdempotencyStatus.RESERVED,
                next_record=IdempotencyRecord(request.principal.tenant_id, scope, key, request_digest, execution_id, IdempotencyStatus.STARTED, None, None, now, now),
            )
            await self._persistence.events.append(execution_id, tenant_id=request.principal.tenant_id, expected_sequence=0, event_type=ExecutionEventType.EXECUTION_STARTED, payload={})
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                raise
            await _best_effort_execution_failure(
                self._persistence,
                execution_id,
                request.principal.tenant_id,
                key,
                request_digest,
                scope,
                status=ExecutionStatus.FAILED,
                error_code=ErrorCode.EXECUTION_START_PERSISTENCE_FAILED.value,
            )
            raise LinktoolsAIError(ErrorCode.EXECUTION_START_PERSISTENCE_FAILED) from error
        if self._launcher is not None:
            try:
                launch_record = await self._persistence.executions.get(
                    execution_id,
                    tenant_id=request.principal.tenant_id,
                )
                if launch_record is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._launcher.start(binding, request, launch_record)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                await _best_effort_execution_failure(
                    self._persistence,
                    execution_id,
                    request.principal.tenant_id,
                    key,
                    request_digest,
                    scope,
                    status=ExecutionStatus.START_UNKNOWN,
                    error_code=ErrorCode.EXECUTION_START_UNKNOWN.value,
                )
                _logger.error("execution start outcome unknown: execution=%s", execution_id)
                raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN) from error
        _logger.info("execution started: execution=%s scope=%s profile=%s", execution_id, scope, request.requested_profile)
        return ExecutionHandle(execution_id, request.requested_profile)

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return ExecutionView(execution.execution_id, execution.status, execution.profile)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        result = await self._persistence.results.get(execution_id, tenant_id=principal.tenant_id)
        if result is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        return ExecutionResult(execution.execution_id, result.status, result.payload_ref or "")

    async def retry(self, binding: AgentBinding, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding.digest:
            raise LinktoolsAIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        retry_request = ExecutionRequest(
            f"retry:{execution_id}",
            request.principal,
            previous.profile,
            request.idempotency_key,
        )
        return await self._start(binding, retry_request, session_id=previous.session_id, scope="execution.retry")

    async def fork(self, binding: AgentBinding, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding.digest:
            raise LinktoolsAIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        fork_request = ExecutionRequest(
            f"fork:{execution_id}",
            request.principal,
            previous.profile,
            request.idempotency_key,
        )
        return await self._start(binding, fork_request, session_id=previous.session_id, scope="execution.fork")

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        execution = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_CANCEL)
        operation_digest = canonical_sha256(
            {
                "action": "execution.cancel",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "execution_id": execution_id,
                "force": request.force,
            }
        )
        operation = await self._persistence.operations.get(
            request.cancel_request_id,
            tenant_id=request.principal.tenant_id,
        )
        if operation is not None:
            if operation.request_digest != operation_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                return CancelExecutionResult(execution_id, execution.status is ExecutionStatus.CANCELLED)
        else:
            now = datetime.now(timezone.utc)
            operation = OperationLedgerRecord(
                request.cancel_request_id,
                request.principal.tenant_id,
                ResourceKind.EXECUTION,
                execution_id,
                execution_id,
                OperationKind.EXECUTION_CANCEL,
                OperationStatus.PENDING,
                operation_digest,
                None,
                None,
                None,
                True,
                await self._persistence.operations.next_sequence(
                    ResourceKind.EXECUTION,
                    execution_id,
                    tenant_id=request.principal.tenant_id,
                ),
                now,
                now,
            )
            operation = await self._persistence.operations.create(operation)
        if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=request.principal.tenant_id,
                expected_status=operation.status,
                next_record=_operation_result(operation, execution.execution_id, execution.result_digest),
            )
            return CancelExecutionResult(execution_id, execution.status is ExecutionStatus.CANCELLED)
        now = datetime.now(timezone.utc)
        cancelling = _next_execution(execution, ExecutionStatus.CANCELLING, now)
        try:
            await self._persistence.executions.compare_and_swap(
                execution_id,
                tenant_id=request.principal.tenant_id,
                expected_snapshot_revision=execution.snapshot_revision,
                next_record=cancelling,
            )
            await self._persistence.events.append(
                execution_id,
                tenant_id=request.principal.tenant_id,
                expected_sequence=cancelling.event_sequence,
                event_type=ExecutionEventType.CANCEL_REQUESTED,
                payload={},
            )
            if self._launcher is not None:
                await self._launcher.cancel(cancelling)
            cancelling_current = await self._persistence.executions.get(
                execution_id,
                tenant_id=request.principal.tenant_id,
            )
            if cancelling_current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            terminal = _next_execution(cancelling_current, ExecutionStatus.CANCELLED, datetime.now(timezone.utc), error_code=None)
            result = ResultRecord(execution_id, request.principal.tenant_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, datetime.now(timezone.utc))
            await self._persistence.results.commit_terminal(ExecutionTerminalCommit(operation.operation_id, cancelling_current.snapshot_revision, terminal, result))
            current = await self._persistence.executions.get(execution_id, tenant_id=request.principal.tenant_id)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._persistence.events.append(
                execution_id,
                tenant_id=request.principal.tenant_id,
                expected_sequence=current.event_sequence,
                event_type=ExecutionEventType.EXECUTION_CANCELLED,
                payload={},
            )
            await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=request.principal.tenant_id,
                expected_status=operation.status,
                next_record=_operation_result(operation, execution_id, None),
            )
            _logger.info("execution cancelled: execution=%s operation=%s", execution_id, operation.operation_id)
        except asyncio.CancelledError:
            raise
        except BaseException:
            try:
                current = await self._persistence.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                if current is not None and current.status is OperationStatus.PENDING:
                    await self._persistence.operations.compare_and_swap(
                        operation.operation_id,
                        tenant_id=request.principal.tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=_operation_failure(current, ErrorCode.STORAGE_UNAVAILABLE.value),
                    )
            except Exception:
                pass
            raise
        return CancelExecutionResult(execution_id, True)

    async def trace(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TraceItem]:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        after = int(cursor or 0)
        page = await self._persistence.traces.list(execution_id, tenant_id=principal.tenant_id, after_sequence=after, limit=limit)
        return Page(tuple(TraceItem(item.execution_id, item.sequence, item.payload) for item in page.items), page.next_cursor)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TranscriptItem]:
        await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        page = await self._persistence.events.list(execution_id, tenant_id=principal.tenant_id, after_sequence=int(cursor or 0), limit=limit)
        items = tuple(TranscriptItem(item.execution_id, item.sequence, str(item.payload.get("text", "")) if isinstance(item.payload, dict) else "") for item in page.items)
        return Page(items, page.next_cursor)

    async def _load_authorized(self, execution_id: str, principal: Principal, action: AuthorizationAction) -> ExecutionRecord:
        header = await self._persistence.executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._persistence.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _request_digest(request: ExecutionRequest, binding: AgentBinding) -> str:
    return canonical_sha256({"prompt": request.prompt, "profile": request.requested_profile.value, "binding": binding.digest})


def _operation_result(
    operation: OperationLedgerRecord,
    result_ref: str,
    result_digest: str | None,
) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.kind,
        OperationStatus.SUCCEEDED,
        operation.request_digest,
        result_ref,
        result_digest,
        None,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _operation_failure(operation: OperationLedgerRecord, error_code: str) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.kind,
        OperationStatus.FAILED,
        operation.request_digest,
        None,
        None,
        error_code,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _next_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, error_code: "str | None" = None) -> ExecutionRecord:
    return ExecutionRecord(record.execution_id, record.tenant_id, record.session_id, record.profile, record.binding_digest, record.parent_execution_id, record.root_execution_id, status, record.snapshot_revision + 1, record.event_sequence, record.trace_sequence, record.result_ref, record.result_digest, error_code, record.safe_error_details, record.created_at, now)


async def _best_effort_execution_failure(
    persistence: RuntimePersistence,
    execution_id: str,
    tenant_id: str,
    key: str,
    digest: str,
    scope: str,
    *,
    status: ExecutionStatus,
    error_code: str,
) -> None:
    try:
        current = await persistence.executions.get(execution_id, tenant_id=tenant_id)
        if current is None:
            return
        record = _next_execution(current, status, datetime.now(timezone.utc), error_code=error_code)
        if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            await persistence.executions.compare_and_swap(record.execution_id, tenant_id=record.tenant_id, expected_snapshot_revision=current.snapshot_revision, next_record=record)
        current_key = await persistence.idempotency.get(scope, key, tenant_id=record.tenant_id)
        if current_key is not None and current_key.status not in {IdempotencyStatus.COMPLETED, IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED, IdempotencyStatus.START_UNKNOWN}:
            next_status = IdempotencyStatus.START_UNKNOWN if status is ExecutionStatus.START_UNKNOWN else IdempotencyStatus.FAILED
            await persistence.idempotency.compare_and_swap(scope, key, expected_status=current_key.status, next_record=IdempotencyRecord(record.tenant_id, scope, key, digest, record.execution_id, next_status, None, record.error_code, record.created_at, record.updated_at))
    except Exception:
        pass


def _stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> LinktoolsAIError:
    try:
        return LinktoolsAIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return LinktoolsAIError(fallback)


__all__ = ["DefaultExecutionService", "ExecutionApi", "ExecutionLauncher", "ExecutionQueryApi"]
