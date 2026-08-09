#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol

from ..core import Page, Principal
from ..errors import ErrorCode, AIError
from ..core import canonical_sha256, idempotency_key_hash
from ..core import AuthorizationAction, AuthorizationPolicy, ResourceRef
from ..core import ExecutionEventType, ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, OperationKind, OperationStatus, ResourceKind, StopReason
from linktools.core import environ
from ._persistence import (
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionCancelRequestCommit,
    ExecutionTerminalCommit, IdempotencyTerminalUpdate, OperationTerminalUpdate,
    IdempotencyRecord,
    OperationLedgerInput,
    OperationLedgerRecord,
    ResultRecord,
    RuntimePersistence,
)
from ._services import (
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
    async def cancel(self, execution: ExecutionRecord) -> "CancelEffectOutcome": ...


class CancelEffectOutcome(StrEnum):
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"


class DefaultExecutionService:
    """Apply execution authorization and state transitions before launching work."""

    def __init__(
        self,
        persistence: RuntimePersistence,
        authorization: AuthorizationPolicy,
        *,
        launcher: "ExecutionLauncher | None" = None,
        operation_ids: "Callable[[], str] | None" = None,
        history_reader: ExecutionHistoryReader,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._history_reader = history_reader

    async def run(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle:
        return await self._start(binding_digest, request, scope="execution.run")

    async def run_for_session(self, binding_digest: str, session_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(binding_digest, request, session_id=session_id, scope="session.resume")

    async def _start(self, binding_digest: str, request: ExecutionRequest, *, session_id: "str | None" = None, source_execution_id: "str | None" = None, base_execution_id: "str | None" = None, parent_execution_id: "str | None" = None, root_execution_id: "str | None" = None, lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN, scope: str = "execution.run", allow_legacy_memory_namespace: bool = False) -> ExecutionHandle:
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.idempotency_key is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        if not allow_legacy_memory_namespace and (not isinstance(request.memory_namespace, str) or request.memory_namespace == ""):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if session_id is not None and source_execution_id is None:
            session = await self._persistence.sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)
            base_execution_id = session.head_execution_id
            lineage_kind = ExecutionLineageKind.SESSION_RESUME
        execution_id = self._operation_ids()
        resource = ResourceRef(ResourceKind.EXECUTION, execution_id, request.principal.tenant_id)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXECUTION_RUN, resource)
        key_hash = idempotency_key_hash(request.idempotency_key)
        request_digest = _request_digest(request, binding_digest, session_id=session_id, source_execution_id=source_execution_id, base_execution_id=base_execution_id, parent_execution_id=parent_execution_id, root_execution_id=root_execution_id, lineage_kind=lineage_kind)
        existing = await self._persistence.idempotency.get(scope, key_hash, tenant_id=request.principal.tenant_id)
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.RESERVED:
                pending = await self._persistence.executions.get(existing.execution_id, tenant_id=request.principal.tenant_id)
                if pending is not None and pending.status is ExecutionStatus.STARTED:
                    return ExecutionHandle(existing.execution_id)
                if pending is None or pending.status is not ExecutionStatus.PENDING_START:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    await self._persistence.executions.claim_start(ExecutionStartClaim(existing.execution_id, request.principal.tenant_id, pending.revision, pending.event_sequence, scope, key_hash, request_digest, datetime.now(timezone.utc)))
                except AIError as error:
                    current = await self._persistence.executions.get(existing.execution_id, tenant_id=request.principal.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(existing.execution_id)
                await self._launch_claim(request, existing.execution_id, request.principal.tenant_id, scope, key_hash)
                return ExecutionHandle(existing.execution_id)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
            if existing.status is IdempotencyStatus.CANCELLED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            return ExecutionHandle(existing.execution_id)
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=execution_id,
            tenant_id=request.principal.tenant_id,
            session_id=session_id,
            binding_digest=binding_digest,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id or execution_id,
            status=ExecutionStatus.PENDING_START,
            revision=0,
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
            memory_namespace=request.memory_namespace,
        )
        reservation = await self._persistence.executions.reserve_start(
            ExecutionStartReservation(
                execution,
                IdempotencyRecord(request.principal.tenant_id, scope, key_hash, request_digest, execution_id, IdempotencyStatus.RESERVED, None, None, now, now),
            )
        )
        if not reservation.created:
            if reservation.idempotency.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if reservation.execution.status is ExecutionStatus.PENDING_START and reservation.idempotency.status is IdempotencyStatus.RESERVED:
                try:
                    await self._persistence.executions.claim_start(ExecutionStartClaim(reservation.execution.execution_id, reservation.execution.tenant_id, reservation.execution.revision, reservation.execution.event_sequence, scope, key_hash, request_digest, now))
                except AIError as error:
                    current = await self._persistence.executions.get(reservation.execution.execution_id, tenant_id=reservation.execution.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(reservation.execution.execution_id)
                await self._launch_claim(request, reservation.execution.execution_id, reservation.execution.tenant_id, scope, key_hash)
            elif reservation.execution.status is ExecutionStatus.STARTED and reservation.idempotency.status is IdempotencyStatus.STARTED:
                return ExecutionHandle(reservation.execution.execution_id)
            elif reservation.execution.status is ExecutionStatus.START_UNKNOWN or reservation.idempotency.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(reservation.execution.execution_id)
        execution_id = reservation.execution.execution_id
        try:
            await self._persistence.executions.claim_start(
                ExecutionStartClaim(execution_id, request.principal.tenant_id, 0, 0, scope, key_hash, request_digest, now)
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                raise
            raise AIError(ErrorCode.EXECUTION_START_PERSISTENCE_FAILED) from error
        await self._launch_claim(request, execution_id, request.principal.tenant_id, scope, key_hash)
        _logger.info("execution started: execution=%s scope=%s", execution_id, scope)
        return ExecutionHandle(execution_id)

    async def _launch_claim(self, request: ExecutionRequest, execution_id: str, tenant_id: str, scope: str, key_hash: str) -> None:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            launch_record = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
            if launch_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._launcher.start(request, launch_record)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            current = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
            if current is not None and current.status is ExecutionStatus.STARTED:
                identity = await self._persistence.idempotency.get(scope, key_hash, tenant_id=tenant_id)
                if identity is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._persistence.executions.mark_start_unknown(ExecutionStartUnknownCommit(execution_id, tenant_id, current.revision, current.event_sequence, scope, key_hash, identity.request_digest, datetime.now(timezone.utc)))
            _logger.error("execution start outcome unknown: execution=%s", execution_id)
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN) from error

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return ExecutionView(execution.execution_id, execution.status)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        result = await self._persistence.results.get(execution_id, tenant_id=principal.tenant_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return ExecutionResult(execution.execution_id, result.status, result.payload_ref or "")

    async def retry(self, binding_digest: str, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        retry_request = ExecutionRequest(
            prompt=request.prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_namespace=previous.memory_namespace,
        )
        return await self._start(binding_digest, retry_request, session_id=previous.session_id, scope="execution.retry", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.RETRY, base_execution_id=previous.base_execution_id, allow_legacy_memory_namespace=True)

    async def fork(self, binding_digest: str, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        fork_request = ExecutionRequest(
            prompt=request.prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_namespace=previous.memory_namespace,
        )
        return await self._start(binding_digest, fork_request, session_id=previous.session_id, scope="execution.fork", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.FORK, base_execution_id=previous.execution_id, allow_legacy_memory_namespace=True)

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
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                return CancelExecutionResult(execution_id, execution.status is ExecutionStatus.CANCELLED)
            if operation.status is OperationStatus.EFFECT_UNKNOWN:
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                return CancelExecutionResult(execution_id, False)
            if operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(operation.error_code)
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
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
        try:
            try:
                cancelling = await self._persistence.executions.request_cancel(
                    ExecutionCancelRequestCommit(
                        execution_id=execution_id,
                        tenant_id=request.principal.tenant_id,
                        expected_revision=execution.revision,
                        expected_event_sequence=execution.event_sequence,
                        operation_id=operation.operation_id,
                        requested_at=datetime.now(timezone.utc),
                    )
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            if self._launcher is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            outcome = await self._launcher.cancel(cancelling)
            if outcome is CancelEffectOutcome.UNKNOWN:
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                current = await self._persistence.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.PENDING:
                    try:
                        current = await self._persistence.operations.compare_and_swap(
                            operation.operation_id,
                            tenant_id=request.principal.tenant_id,
                            expected_status=OperationStatus.PENDING,
                            next_record=_operation_status(current, OperationStatus.EFFECT_UNKNOWN),
                        )
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_CONFLICT:
                            raise
                        resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                        if resolved is not None:
                            return resolved
                        current = await self._persistence.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                        if current is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.EFFECT_UNKNOWN:
                    _logger.warning("execution cancellation effect unknown: execution=%s operation=%s", execution_id, operation.operation_id)
                    return CancelExecutionResult(execution_id, False)
                if current.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                    latest = await self._persistence.executions.get(execution_id, tenant_id=request.principal.tenant_id)
                    if latest is None or latest.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return CancelExecutionResult(execution_id, latest.status is ExecutionStatus.CANCELLED)
                if current.status is OperationStatus.FAILED:
                    raise _stable_operation_error(current.error_code)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            cancelling_current = await self._persistence.executions.get(execution_id, tenant_id=request.principal.tenant_id)
            if cancelling_current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
            if cancelling_current.status is not ExecutionStatus.CANCELLING:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            now = datetime.now(timezone.utc)
            terminal = _next_execution(cancelling_current, ExecutionStatus.CANCELLED, now, error_code=None, terminal_event=True)
            result = ResultRecord(execution_id, request.principal.tenant_id, ExecutionStatus.CANCELLED, "none", 1, "none", None, None, StopReason.CANCELLED, 0, 0, 0, now)
            idempotency_records = await self._persistence.idempotency.list_by_execution(execution_id, tenant_id=request.principal.tenant_id)
            if len(idempotency_records) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            identity = idempotency_records[0] if idempotency_records else None
            idempotency = None if identity is None else IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, IdempotencyStatus.CANCELLED, identity.request_digest, None, None)
            current_operation = await self._persistence.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
            if current_operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current_operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                latest = await self._persistence.executions.get(execution_id, tenant_id=request.principal.tenant_id)
                if latest is None or latest.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return CancelExecutionResult(execution_id, latest.status is ExecutionStatus.CANCELLED)
            if current_operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(current_operation.error_code)
            if current_operation.status not in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            operation_update = OperationTerminalUpdate(operation.operation_id, current_operation.status, OperationStatus.SUCCEEDED, execution_id, cancelling_current.result_digest, None)
            try:
                await self._persistence.results.commit_terminal(
                    ExecutionTerminalCommit(
                        expected_revision=cancelling_current.revision,
                        expected_event_sequence=cancelling_current.event_sequence,
                        execution=terminal,
                        result=result,
                        terminal_event_type=ExecutionEventType.EXECUTION_CANCELLED,
                        terminal_event_payload={},
                        idempotency=idempotency,
                        operation=operation_update,
                    )
                )
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            _logger.info("execution cancelled: execution=%s operation=%s", execution_id, operation.operation_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
            if isinstance(error, AIError) and error.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                raise
            error_code = error.code.value if isinstance(error, AIError) else ErrorCode.STORAGE_UNAVAILABLE.value
            try:
                current = await self._persistence.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                if current is not None and current.status is OperationStatus.PENDING:
                    await self._persistence.operations.compare_and_swap(
                        operation.operation_id,
                        tenant_id=request.principal.tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=_operation_failure(current, error_code),
                    )
            except Exception:
                _logger.error(
                    "execution cancellation ledger update failed: execution=%s operation=%s",
                    execution_id,
                    operation.operation_id,
                    exc_info=environ.debug,
                )
            raise
        return CancelExecutionResult(execution_id, True)

    async def _resolve_cancel_race(
        self,
        execution_id: str,
        tenant_id: str,
        operation: OperationLedgerRecord,
    ) -> "CancelExecutionResult | None":
        current = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
        if current is None or current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return None
        current_operation = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
        if current_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for attempt in range(2):
            if current_operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                break
            if current_operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(current_operation.error_code)
            if current_operation.status not in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            try:
                current_operation = await self._persistence.operations.compare_and_swap(
                    operation.operation_id,
                    tenant_id=tenant_id,
                    expected_status=current_operation.status,
                    next_record=_operation_result(current_operation, current.execution_id, current.result_digest),
                )
                break
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                reloaded = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = reloaded
                if attempt == 1 and current_operation.status in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info(
            "execution cancellation race resolved: execution=%s status=%s operation=%s",
            execution_id,
            current.status.value,
            operation.operation_id,
        )
        return CancelExecutionResult(execution_id, current.status is ExecutionStatus.CANCELLED)


    async def trace(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TraceItem]:
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.trace(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TranscriptItem]:
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.transcript(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    async def _load_authorized(self, execution_id: str, principal: Principal, action: AuthorizationAction) -> ExecutionRecord:
        header = await self._persistence.executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._persistence.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _request_digest(
    request: ExecutionRequest,
    binding_digest: str,
    *,
    session_id: str | None,
    source_execution_id: str | None,
    base_execution_id: str | None,
    parent_execution_id: str | None,
    root_execution_id: str | None,
    lineage_kind: ExecutionLineageKind,
) -> str:
    return canonical_sha256(
        {
            "prompt": request.prompt,
            "binding_digest": binding_digest,
            "scope": session_id or "execution",
            "principal_id": request.principal.principal_id,
            "tenant_id": request.principal.tenant_id,
            "session_id": session_id,
            "source_execution_id": source_execution_id,
            "base_execution_id": base_execution_id,
            "parent_execution_id": parent_execution_id,
            "root_identity": root_execution_id or "$self",
            "lineage_kind": lineage_kind.value,
            "memory_namespace_digest": None if request.memory_namespace is None else canonical_sha256(request.memory_namespace),
        }
    )


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


def _operation_status(operation: OperationLedgerRecord, status: OperationStatus) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.kind,
        status,
        operation.request_digest,
        operation.result_ref,
        operation.result_digest,
        operation.error_code,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _next_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, error_code: "str | None" = None, agent_run_sequence: int | None = None, terminal_event: bool = False) -> ExecutionRecord:
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
        event_sequence=record.event_sequence + (1 if terminal_event else 0),
        agent_run_sequence=record.agent_run_sequence if agent_run_sequence is None else agent_run_sequence,
        result_ref=record.result_ref,
        result_digest=record.result_digest,
        error_code=error_code,
        safe_error_details=record.safe_error_details,
        created_at=record.created_at,
        updated_at=now,
        memory_namespace=record.memory_namespace,
    )


def _stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    try:
        return AIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return AIError(fallback)


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["CancelEffectOutcome", "DefaultExecutionService", "ExecutionApi", "ExecutionLauncher", "ExecutionQueryApi"]
