#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from ..core import Page, Principal
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256, idempotency_key_hash
from ..core.principal import AuthorizationAction, AuthorizationPolicy, ResourceRef
from ..core.value import ExecutionEventType, ExecutionLineageKind, ExecutionProfile, ExecutionStatus, IdempotencyStatus, OperationKind, OperationStatus, ResourceKind, StopReason
from linktools.core import environ
from .persistence import (
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit, IdempotencyTerminalUpdate, OperationTerminalUpdate,
    IdempotencyRecord,
    OperationLedgerInput,
    OperationLedgerRecord,
    ResultRecord,
    RuntimePersistence,
)
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
    ExecutionHistoryReader,
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
    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None: ...
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
        service_profile: "ExecutionProfile",
        history_reader: ExecutionHistoryReader,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._service_profile = service_profile
        self._history_reader = history_reader

    async def run(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle:
        return await self._start(binding_digest, request, scope="execution.run")

    async def run_for_session(self, binding_digest: str, session_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not session_id.strip():
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(binding_digest, request, session_id=session_id, scope="session.resume")

    async def _start(self, binding_digest: str, request: ExecutionRequest, *, session_id: "str | None" = None, source_execution_id: "str | None" = None, base_execution_id: "str | None" = None, parent_execution_id: "str | None" = None, root_execution_id: "str | None" = None, lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN, scope: str = "execution.run") -> ExecutionHandle:
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        if request.requested_profile is not self._service_profile:
            raise LinktoolsAIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if self._launcher is None:
            raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        execution_id = self._operation_ids()
        resource = ResourceRef(ResourceKind.EXECUTION, execution_id, request.principal.tenant_id)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXECUTION_RUN, resource)
        key = request.idempotency_key or execution_id
        key_hash = idempotency_key_hash(key)
        request_digest = _request_digest(request, binding_digest)
        existing = await self._persistence.idempotency.get(scope, key_hash, tenant_id=request.principal.tenant_id)
        if existing is not None:
            if existing.request_digest != request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.START_UNKNOWN:
                raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.RESERVED:
                pending = await self._persistence.executions.get(existing.execution_id, tenant_id=request.principal.tenant_id)
                if pending is not None and pending.status is ExecutionStatus.STARTED:
                    return ExecutionHandle(existing.execution_id, request.requested_profile)
                if pending is None or pending.status is not ExecutionStatus.PENDING_START:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    await self._persistence.executions.claim_start(ExecutionStartClaim(existing.execution_id, request.principal.tenant_id, pending.snapshot_revision, pending.event_sequence, scope, key_hash, request_digest, datetime.now(timezone.utc)))
                except LinktoolsAIError as error:
                    current = await self._persistence.executions.get(existing.execution_id, tenant_id=request.principal.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(existing.execution_id, request.requested_profile)
                await self._launch_claim(request, existing.execution_id, request.principal.tenant_id, scope, key_hash)
                return ExecutionHandle(existing.execution_id, request.requested_profile)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
            if existing.status is IdempotencyStatus.CANCELLED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            return ExecutionHandle(existing.execution_id, request.requested_profile)
        now = datetime.now(timezone.utc)
        if session_id is not None and source_execution_id is None:
            session = await self._persistence.sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if session is None:
                raise LinktoolsAIError(ErrorCode.SESSION_NOT_FOUND)
            source_execution_id = session.head_execution_id
            base_execution_id = source_execution_id
            lineage_kind = ExecutionLineageKind.SESSION_RESUME
        execution = ExecutionRecord(
            execution_id=execution_id,
            tenant_id=request.principal.tenant_id,
            session_id=session_id,
            profile=request.requested_profile,
            binding_digest=binding_digest,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id or execution_id,
            status=ExecutionStatus.PENDING_START,
            snapshot_revision=0,
            event_sequence=0,
            result_ref=None,
            result_digest=None,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            source_execution_id=source_execution_id,
            base_execution_id=base_execution_id,
            lineage_kind=lineage_kind,
            agent_run_sequence=0,
        )
        reservation = await self._persistence.executions.reserve_start(
            ExecutionStartReservation(
                execution,
                IdempotencyRecord(request.principal.tenant_id, scope, key_hash, request_digest, execution_id, IdempotencyStatus.RESERVED, None, None, now, now),
            )
        )
        if not reservation.created:
            if reservation.idempotency.request_digest != request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if reservation.execution.status is ExecutionStatus.PENDING_START and reservation.idempotency.status is IdempotencyStatus.RESERVED:
                try:
                    await self._persistence.executions.claim_start(ExecutionStartClaim(reservation.execution.execution_id, reservation.execution.tenant_id, reservation.execution.snapshot_revision, reservation.execution.event_sequence, scope, key_hash, request_digest, now))
                except LinktoolsAIError as error:
                    current = await self._persistence.executions.get(reservation.execution.execution_id, tenant_id=reservation.execution.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(reservation.execution.execution_id, request.requested_profile)
                await self._launch_claim(request, reservation.execution.execution_id, reservation.execution.tenant_id, scope, key_hash)
            elif reservation.execution.status is ExecutionStatus.STARTED and reservation.idempotency.status is IdempotencyStatus.STARTED:
                return ExecutionHandle(reservation.execution.execution_id, request.requested_profile)
            elif reservation.execution.status is ExecutionStatus.START_UNKNOWN or reservation.idempotency.status is IdempotencyStatus.START_UNKNOWN:
                raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN)
            else:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(reservation.execution.execution_id, request.requested_profile)
        execution_id = reservation.execution.execution_id
        try:
            await self._persistence.executions.claim_start(
                ExecutionStartClaim(execution_id, request.principal.tenant_id, 0, 0, scope, key_hash, request_digest, now)
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                raise
            raise LinktoolsAIError(ErrorCode.EXECUTION_START_PERSISTENCE_FAILED) from error
        await self._launch_claim(request, execution_id, request.principal.tenant_id, scope, key_hash)
        _logger.info("execution started: execution=%s scope=%s profile=%s", execution_id, scope, request.requested_profile)
        return ExecutionHandle(execution_id, request.requested_profile)

    async def _launch_claim(self, request: ExecutionRequest, execution_id: str, tenant_id: str, scope: str, key_hash: str) -> None:
        if self._launcher is None:
            raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            launch_record = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
            if launch_record is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._launcher.start(request, launch_record)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            current = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
            if current is not None and current.status is ExecutionStatus.STARTED:
                await self._persistence.executions.mark_start_unknown(ExecutionStartUnknownCommit(execution_id, tenant_id, current.snapshot_revision, scope, key_hash, datetime.now(timezone.utc)))
            _logger.error("execution start outcome unknown: execution=%s", execution_id)
            raise LinktoolsAIError(ErrorCode.EXECUTION_START_UNKNOWN) from error

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return ExecutionView(execution.execution_id, execution.status, execution.profile)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        result = await self._persistence.results.get(execution_id, tenant_id=principal.tenant_id)
        if result is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        return ExecutionResult(execution.execution_id, result.status, result.payload_ref or "")

    async def retry(self, binding_digest: str, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest or previous.profile is not self._service_profile:
            raise LinktoolsAIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        retry_request = ExecutionRequest(
            request.prompt,
            request.principal,
            previous.profile,
            request.idempotency_key,
        )
        return await self._start(binding_digest, retry_request, session_id=previous.session_id, scope="execution.retry", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.RETRY, base_execution_id=previous.base_execution_id or previous.execution_id)

    async def fork(self, binding_digest: str, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest or previous.profile is not self._service_profile:
            raise LinktoolsAIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        fork_request = ExecutionRequest(
            request.prompt,
            request.principal,
            previous.profile,
            request.idempotency_key,
        )
        return await self._start(binding_digest, fork_request, session_id=previous.session_id, scope="execution.fork", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.FORK, base_execution_id=previous.base_execution_id or previous.execution_id)

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
        operation_id = idempotency_key_hash(request.cancel_request_id)
        operation = await self._persistence.operations.get(
            operation_id,
            tenant_id=request.principal.tenant_id,
        )
        if operation is not None:
            if operation.request_digest != operation_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                return CancelExecutionResult(execution_id, execution.status is ExecutionStatus.CANCELLED)
        else:
            now = datetime.now(timezone.utc)
            operation = OperationLedgerInput(
                operation_id,
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
                now,
                now,
            )
            operation = await self._persistence.operations.append(operation)
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
            now = datetime.now(timezone.utc)
            terminal = _next_execution(cancelling_current, ExecutionStatus.CANCELLED, now, error_code=None, terminal_event=True)
            result = ResultRecord(execution_id, request.principal.tenant_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now)
            idempotency_records = await self._persistence.idempotency.list_by_execution(execution_id, tenant_id=request.principal.tenant_id)
            if len(idempotency_records) > 1:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            identity = idempotency_records[0] if idempotency_records else None
            idempotency = None if identity is None else IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, IdempotencyStatus.CANCELLED, identity.request_digest, None, None)
            operation_update = OperationTerminalUpdate(operation.operation_id, operation.status, OperationStatus.SUCCEEDED, None, None, None)
            await self._persistence.results.commit_terminal(ExecutionTerminalCommit(cancelling_current.snapshot_revision, cancelling_current.event_sequence, terminal, result, ExecutionEventType.EXECUTION_CANCELLED, {}, idempotency=idempotency, operation=operation_update))
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
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.trace(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TranscriptItem]:
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.transcript(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    async def _load_authorized(self, execution_id: str, principal: Principal, action: AuthorizationAction) -> ExecutionRecord:
        header = await self._persistence.executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._persistence.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _request_digest(request: ExecutionRequest, binding_digest: str) -> str:
    return canonical_sha256({"prompt": request.prompt, "profile": request.requested_profile.value, "binding": binding_digest})


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


def _next_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, error_code: "str | None" = None, agent_run_sequence: int | None = None, terminal_event: bool = False) -> ExecutionRecord:
    return ExecutionRecord(record.execution_id, record.tenant_id, record.session_id, record.profile, record.binding_digest, record.parent_execution_id, record.root_execution_id, status, record.snapshot_revision + 1, record.event_sequence + (1 if terminal_event else 0), record.result_ref, record.result_digest, error_code, record.safe_error_details, record.created_at, now, record.source_execution_id, record.base_execution_id, record.lineage_kind, record.agent_run_sequence if agent_run_sequence is None else agent_run_sequence)


def _stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> LinktoolsAIError:
    try:
        return LinktoolsAIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return LinktoolsAIError(fallback)


__all__ = ["DefaultExecutionService", "ExecutionApi", "ExecutionLauncher", "ExecutionQueryApi"]
