#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution lifecycle decisions independent of Runtime composition and drivers."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol
from contextlib import AbstractContextManager
from linktools.core import environ
from ..core import (AuthorizationAction, AuthorizationPolicy, Principal, ResourceRef,
    ResourceKind, ExecutionStatus, ExecutionEventType, IdempotencyStatus, JsonValue,
    OperationKind, OperationStatus, OperationLedgerInput, OperationLedgerRecord,
    StopReason, UsageMetrics, canonical_sha256, principal_identity_payload,
    idempotency_key_digest as compute_idempotency_key_digest)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..storage import ObjectRef
from ._records import (ExecutionRecord, ResultRecord, IdempotencyRecord,
    ExecutionCancelRequestCommit, ExecutionTerminalCommit, ExecutionTerminalCommitResult,
    IdempotencyTerminalUpdate, OperationTerminalUpdate)
from ._api import CancelExecutionRequest, CancelExecutionResult, ExecutionResult

_logger = environ.get_logger("ai.execution")
_TerminalMetricDelivery = Callable[[], Awaitable[None]]

class CancelEffectOutcome(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"


class ExecutionDriver(Protocol):
    async def commit_cancel_checkpoint(self, commit: ExecutionCancelRequestCommit,
        *, expected_status: ExecutionStatus) -> ExecutionRecord: ...
    async def cancel(self, execution: ExecutionRecord, *, force: bool = False) -> CancelEffectOutcome: ...
    async def cancel_children(self, parent_execution_id: str, principal: Principal) -> None: ...
    async def abort_start(self, execution: ExecutionRecord) -> None: ...
    async def persist_cancel_intent(self, execution: ExecutionRecord,
        operation: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def verify_terminal_projection(self, execution: ExecutionRecord,
        status: ExecutionStatus, required_agent_run_id: str | None) -> None: ...
    async def commit_terminal_checkpoint(self, commit: ExecutionTerminalCommit, *,
        session_id: str | None, expected_execution: ExecutionRecord | None = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None
    ) -> ExecutionTerminalCommitResult: ...


class ExecutionManagementPort(Protocol):
    """Storage observation and cross-domain atomic transitions supplied by the host."""
    @property
    def embedded_results(self) -> bool: ...
    def source_reads(self) -> AbstractContextManager[None]: ...
    async def write_output(self, content: bytes, *, tenant_id: str) -> ObjectRef: ...
    @property
    def driver(self) -> ExecutionDriver | None: ...
    @property
    def has_pending_cancel_promotion(self) -> bool: ...
    async def promote_pending_cancel(self, execution_id: str) -> None: ...
    async def read_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None: ...
    async def read_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def read_source_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    def has_local_pending(self, execution_id: str, *, tenant_id: str) -> bool: ...
    async def read_source_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None: ...
    async def read_remote_pending(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None: ...
    async def signal_remote_cancel(self, execution_id: str, *, tenant_id: str) -> None: ...
    async def get_operation(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None: ...
    async def append_operation(self, operation: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def compare_and_swap_operation(self, operation_id: str, *, tenant_id: str,
        expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord: ...
    async def list_idempotency(self, resource_kind: ResourceKind, resource_id: str,
        *, tenant_id: str) -> tuple[IdempotencyRecord, ...]: ...
    async def get_idempotency(self, scope: str, key_digest: str, *, tenant_id: str) -> IdempotencyRecord | None: ...
    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None: ...
    async def result_record(self, execution: ExecutionRecord, *, tenant_id: str) -> ResultRecord | None: ...
    async def read_output(self, reference: ObjectRef) -> bytes: ...


class ExecutionManager:
    """Own durable admission, authorization, result validation and cancellation."""
    def __init__(self, port: ExecutionManagementPort, authorization: AuthorizationPolicy) -> None:
        self._port = port
        self._authorization = authorization

    def require_driver(self) -> ExecutionDriver:
        driver = self._port.driver
        if driver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return driver

    async def authorize_execution(self, execution_id: str, principal: Principal,
                                  action: AuthorizationAction) -> ResourceRef:
        header = await self._port.read_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        return header

    async def resolve_admission(self, scope: str, key_digest: str, request_digest: str,
                                principal: Principal) -> tuple[IdempotencyRecord, ExecutionRecord] | None:
        identity = await self._port.get_idempotency(scope, key_digest, tenant_id=principal.tenant_id)
        if identity is None:
            return None
        if identity.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if (identity.scope != scope or identity.idempotency_key_digest != key_digest
                or identity.resource_kind is not ResourceKind.EXECUTION):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if identity.status is IdempotencyStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        header = await self._port.read_source_header(identity.resource_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_RUN, header)
        execution = await self._port.read_execution(identity.resource_id, tenant_id=principal.tenant_id)
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self.authorize_loaded_execution(execution, principal, AuthorizationAction.EXECUTION_RUN,
                                             owner_principal_id=header.owner_principal_id)
        if execution.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        return identity, execution

    async def _load_cancel_execution(
        self, execution_id: str, request: CancelExecutionRequest,
    ) -> ExecutionRecord:
        execution = await self.load_authorized(
            execution_id, request.principal, AuthorizationAction.EXECUTION_CANCEL,
        )
        tenant_id = request.principal.tenant_id
        if self._port.has_local_pending(execution_id, tenant_id=tenant_id) or execution.status not in {
            ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.CANCELLING,
        }:
            return execution

        async def source_execution() -> ExecutionRecord | None:
            source = await self._port.read_source_execution(execution_id, tenant_id=tenant_id)
            if source is not None:
                await self.authorize_loaded_execution(
                    source, request.principal, AuthorizationAction.EXECUTION_CANCEL,
                    owner_principal_id=execution.principal_id,
                )
            return source

        source = await source_execution()
        if source is not None:
            return source
        pending = await self._port.read_remote_pending(execution_id, tenant_id=tenant_id)
        if pending is not None:
            await self.authorize_loaded_execution(
                pending, request.principal, AuthorizationAction.EXECUTION_CANCEL,
                owner_principal_id=execution.principal_id,
            )
            await self._port.signal_remote_cancel(execution_id, tenant_id=tenant_id)
        poll_backoff = 0.05
        while True:
            source = await source_execution()
            if source is not None:
                return source
            pending = await self._port.read_remote_pending(execution_id, tenant_id=tenant_id)
            if pending is None:
                source = await source_execution()
                if source is not None:
                    return source
                raise AIError(ErrorCode.EXECUTION_NOT_READY)
            # The owner promotes and applies the ordinary cancellation protocol.
            # A shared cancellation request is never a terminal source receipt.
            await asyncio.sleep(poll_backoff)
            poll_backoff = min(1.0, poll_backoff * 2)

    async def _cancel(
        self, execution_id: str, request: CancelExecutionRequest,
        *, capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> CancelExecutionResult:
        execution = await self._load_cancel_execution(execution_id, request)
        if self._port.has_pending_cancel_promotion:
            await self._port.promote_pending_cancel(execution_id)
            execution = await self.load_authorized(
                execution_id, request.principal, AuthorizationAction.EXECUTION_CANCEL,
            )
        operation_digest = canonical_sha256(
            {
                "action": "execution.cancel",
                "principal": principal_identity_payload(request.principal),
                "execution_id": execution_id,
                "force": request.force,
            }
        )
        operation_id = compute_idempotency_key_digest(request.idempotency_key)
        operation = await self._port.get_operation(
            operation_id,
            tenant_id=request.principal.tenant_id,
        )
        if operation is not None:
            if operation.request_digest != operation_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.CANCELLED,
            }:
                settled = await self._port.read_execution(execution_id, tenant_id=request.principal.tenant_id)
                if settled is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return CancelExecutionResult(
                    execution_id, settled.status is ExecutionStatus.CANCELLED
                )
            if operation.status is OperationStatus.EFFECT_UNKNOWN:
                resolved = await self._resolve_cancel_race(
                    execution_id, request.principal.tenant_id, operation
                )
                if resolved is not None:
                    return resolved
            if operation.status is OperationStatus.FAILED:
                raise stable_operation_error(operation.error_code)
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
            operation = await self._port.append_operation(operation)
        latest = await self._port.read_execution(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = latest
        prepared_before_claim = execution.status is ExecutionStatus.PENDING_START
        if execution.status is ExecutionStatus.FINALIZING:
            resolved = await self._resolve_cancel_race(
                execution_id,
                request.principal.tenant_id,
                operation,
            )
            if resolved is not None:
                return resolved
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            resolved = await self._resolve_cancel_race(
                execution_id, request.principal.tenant_id, operation
            )
            if resolved is not None:
                return resolved
        if self._port.driver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        metric_delivery: _TerminalMetricDelivery | None = None

        def capture_terminal_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

        try:
            if execution.status is ExecutionStatus.CANCELLING:
                cancelling = execution
            else:
                try:
                    cancelling = await self._port.driver.commit_cancel_checkpoint(
                        ExecutionCancelRequestCommit(
                            execution_id=execution_id,
                            expected_revision=execution.revision,
                            expected_event_seq=execution.event_seq,
                            operation_id=operation.operation_id,
                            requested_at=datetime.now(timezone.utc),
                        ),
                        expected_status=execution.status,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    current = await self._port.read_execution(
                        execution_id, tenant_id=request.principal.tenant_id,
                    )
                    if current is not None and current.status is ExecutionStatus.CANCELLING:
                        cancelling = current
                    else:
                        resolved = await self._resolve_cancel_race(
                            execution_id, request.principal.tenant_id, operation
                        )
                        if resolved is not None:
                            return resolved
                        raise
            if cancelling.status is not ExecutionStatus.CANCELLING:
                resolved = await self._resolve_cancel_race(
                    execution_id,
                    request.principal.tenant_id,
                    operation,
                )
                if resolved is not None:
                    return resolved
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if self._port.driver is not None:
                await self._port.driver.cancel_children(
                    cancelling.execution_id, request.principal
                )
            outcome = await self._port.driver.cancel(cancelling, force=request.force)
            if outcome is CancelEffectOutcome.UNKNOWN:
                recovery_current = await self._port.read_execution(
                    execution_id,
                    tenant_id=request.principal.tenant_id,
                )
                if recovery_current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if recovery_current.status is ExecutionStatus.RECOVERY_REQUIRED:
                    return await self._cancel_recovery_required(
                        recovery_current,
                        request,
                        capture_metric=capture_metric,
                    )
                resolved = await self._resolve_cancel_race(
                    execution_id, request.principal.tenant_id, operation
                )
                if resolved is not None:
                    return resolved
                current = await self._port.get_operation(
                    operation.operation_id, tenant_id=request.principal.tenant_id
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.PENDING:
                    try:
                        current = await self._port.compare_and_swap_operation(
                            operation.operation_id,
                            tenant_id=request.principal.tenant_id,
                            expected_status=OperationStatus.PENDING,
                            next_record=operation_status(
                                current, OperationStatus.EFFECT_UNKNOWN
                            ),
                        )
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_CONFLICT:
                            raise
                        resolved = await self._resolve_cancel_race(
                            execution_id, request.principal.tenant_id, operation
                        )
                        if resolved is not None:
                            return resolved
                        current = await self._port.get_operation(
                            operation.operation_id,
                            tenant_id=request.principal.tenant_id,
                        )
                        if current is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.EFFECT_UNKNOWN:
                    _logger.warning(
                        "execution cancellation effect unknown: execution=%s operation=%s",
                        execution_id,
                        operation.operation_id,
                    )
                    return CancelExecutionResult(execution_id, False)
                if current.status in {
                    OperationStatus.SUCCEEDED,
                    OperationStatus.CANCELLED,
                }:
                    latest = await self._port.read_execution(
                        execution_id, tenant_id=request.principal.tenant_id
                    )
                    if latest is None or latest.status not in {
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    }:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return CancelExecutionResult(
                        execution_id, latest.status is ExecutionStatus.CANCELLED
                    )
                if current.status is OperationStatus.FAILED:
                    raise stable_operation_error(current.error_code)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            cancelling_current = await self._port.read_execution(
                execution_id, tenant_id=request.principal.tenant_id
            )
            if cancelling_current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            resolved = await self._resolve_cancel_race(
                execution_id, request.principal.tenant_id, operation
            )
            if resolved is not None:
                return resolved
            if cancelling_current.status is not ExecutionStatus.CANCELLING:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            now = datetime.now(timezone.utc)
            terminal = next_execution(
                cancelling_current,
                ExecutionStatus.CANCELLED,
                now,
                error_code=ErrorCode.EXECUTION_CANCELLED.value,
                safe_error_details={},
                terminal_event=True,
            )
            result = ResultRecord(
                None,
                StopReason.CANCELLED,
                UsageMetrics(),
                now,
            )
            idempotency_records = await self._port.list_idempotency(
                ResourceKind.EXECUTION,
                execution_id,
                tenant_id=request.principal.tenant_id,
            )
            if len(idempotency_records) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            identity = idempotency_records[0] if idempotency_records else None
            idempotency = (
                None
                if identity is None
                else IdempotencyTerminalUpdate(
                    identity.scope,
                    identity.idempotency_key_digest,
                    identity.status,
                    IdempotencyStatus.CANCELLED,
                    identity.request_digest,
                    None,
                    None,
                )
            )
            current_operation = await self._port.get_operation(
                operation.operation_id, tenant_id=request.principal.tenant_id
            )
            if current_operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current_operation.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.CANCELLED,
            }:
                latest = await self._port.read_execution(
                    execution_id, tenant_id=request.principal.tenant_id
                )
                if latest is None or latest.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return CancelExecutionResult(
                    execution_id, latest.status is ExecutionStatus.CANCELLED
                )
            if current_operation.status is OperationStatus.FAILED:
                raise stable_operation_error(current_operation.error_code)
            if current_operation.status not in {
                OperationStatus.PENDING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            operation_update = OperationTerminalUpdate(
                operation.operation_id,
                current_operation.status,
                OperationStatus.SUCCEEDED,
                execution_id,
                None,
                None,
            )
            await self.require_driver().verify_terminal_projection(
                cancelling_current, ExecutionStatus.CANCELLED, None
            )
            terminal_commit = ExecutionTerminalCommit(
                expected_revision=cancelling_current.revision,
                expected_event_seq=cancelling_current.event_seq,
                execution=terminal,
                result=result,
                terminal_event_type=ExecutionEventType.EXECUTION_CANCELLED,
                terminal_event_payload={
                    "error_code": ErrorCode.EXECUTION_CANCELLED.value,
                    "safe_error_details": {},
                },
                idempotency=idempotency,
                operation=operation_update,
            )
            if self._port.driver is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                await self._port.driver.commit_terminal_checkpoint(
                    terminal_commit,
                    session_id=cancelling_current.session_id,
                    expected_execution=cancelling_current,
                    capture_metric=capture_terminal_metric,
                )
            except AIError as error:
                if error.code not in {
                    ErrorCode.STORAGE_CONFLICT,
                    ErrorCode.EXECUTION_RESULT_CONFLICT,
                }:
                    raise
                resolved = await self._resolve_cancel_race(
                    execution_id, request.principal.tenant_id, operation
                )
                if resolved is not None:
                    return resolved
                raise
            if prepared_before_claim:
                try:
                    await self._port.driver.abort_start(terminal)
                except Exception:
                    _logger.error(
                        "pending cancellation cleanup failed: execution=%s",
                        execution_id,
                        exc_info=environ.debug,
                    )
            _logger.info(
                "execution cancelled: execution=%s operation=%s",
                execution_id,
                operation.operation_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            recovery_current = await self._port.read_execution(
                execution_id,
                tenant_id=request.principal.tenant_id,
            )
            if (
                recovery_current is not None
                and recovery_current.status is ExecutionStatus.RECOVERY_REQUIRED
            ):
                return await self._cancel_recovery_required(
                    recovery_current,
                    request,
                    capture_metric=capture_metric,
                )
            resolved = await self._resolve_cancel_race(
                execution_id, request.principal.tenant_id, operation
            )
            if resolved is not None:
                return resolved
            if isinstance(error, AIError) and error.code in {
                ErrorCode.STORAGE_CONFLICT,
                ErrorCode.EXECUTION_RESULT_CONFLICT,
            }:
                raise

            error_code = (
                error.code.value
                if isinstance(error, AIError)
                else ErrorCode.INTERNAL_ERROR.value
            )
            try:
                current = await self._port.get_operation(
                    operation.operation_id, tenant_id=request.principal.tenant_id
                )
                if current is not None and current.status is OperationStatus.PENDING:
                    await self._port.compare_and_swap_operation(
                        operation.operation_id,
                        tenant_id=request.principal.tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=operation_failure(current, error_code),
                    )
            except Exception:
                _logger.exception(
                    "execution cancellation ledger update failed: execution=%s operation=%s",
                    execution_id,
                    operation.operation_id,
                )
            raise
        finally:
            if metric_delivery is not None:
                if capture_metric is not None:
                    capture_metric(metric_delivery)
                else:
                    await metric_delivery()
        return CancelExecutionResult(execution_id, True)


    async def cancel_authorized(
        self,
        execution_id: str,
        request: CancelExecutionRequest,
        *,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> CancelExecutionResult:
        execution = await self._load_cancel_execution(execution_id, request)
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            return await self._cancel_recovery_required(
                execution, request, capture_metric=capture_metric,
            )
        return await self._cancel(execution_id, request, capture_metric=capture_metric)


    async def _cancel_recovery_required(
        self,
        execution: ExecutionRecord,
        request: CancelExecutionRequest,
        *,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> CancelExecutionResult:
        operation_digest = canonical_sha256(
            {
                "action": "execution.cancel",
                "principal": principal_identity_payload(request.principal),
                "execution_id": execution.execution_id,
                "force": request.force,
            }
        )
        operation_id = compute_idempotency_key_digest(request.idempotency_key)
        now = datetime.now(timezone.utc)
        candidate = OperationLedgerInput(
            operation_id,
            request.principal.tenant_id,
            ResourceKind.EXECUTION,
            execution.execution_id,
            execution.execution_id,
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
        try:
            current = await self.require_driver().persist_cancel_intent(
                execution,
                candidate,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            latest = await self.load_authorized(
                execution.execution_id,
                request.principal,
                AuthorizationAction.EXECUTION_RECOVER,
            )
            if latest.status is not ExecutionStatus.RECOVERY_REQUIRED:
                return await self._cancel(
                    execution.execution_id, request, capture_metric=capture_metric,
                )
            raise
        if (
            current.request_digest != operation_digest
            or current.resource_kind is not ResourceKind.EXECUTION
            or current.resource_id != execution.execution_id
            or current.execution_id != execution.execution_id
            or current.operation_kind is not OperationKind.EXECUTION_CANCEL
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if current.status is OperationStatus.FAILED:
            try:
                code = ErrorCode(current.error_code or "")
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise AIError(code)
        if request.force:
            outcome = await self.require_driver().cancel(execution, force=True)
            if outcome is CancelEffectOutcome.CONFIRMED:
                return await self._cancel(execution.execution_id, request, capture_metric=capture_metric)
        return CancelExecutionResult(execution.execution_id, False)


    async def _resolve_cancel_race(
        self,
        execution_id: str,
        tenant_id: str,
        operation: OperationLedgerRecord,
    ) -> "CancelExecutionResult | None":
        current = await self._port.read_execution(execution_id, tenant_id=tenant_id)
        if current is None or current.status not in {
            ExecutionStatus.FINALIZING,
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return None
        current_operation = await self._port.get_operation(
            operation.operation_id, tenant_id=tenant_id
        )
        if current_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for attempt in range(2):
            if current_operation.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.CANCELLED,
            }:
                break
            if current_operation.status is OperationStatus.FAILED:
                raise stable_operation_error(current_operation.error_code)
            if current_operation.status not in {
                OperationStatus.PENDING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is ExecutionStatus.FINALIZING:
                result_ref = current.execution_id
                result_digest = None
            else:
                result = await self._port.get_result(
                    execution_id, tenant_id=tenant_id
                )
                result_ref = current.execution_id
                result_digest = (
                    None
                    if result is None or result.output is None
                    else result.output.digest
                )
            try:
                current_operation = await self._port.compare_and_swap_operation(
                    operation.operation_id,
                    tenant_id=tenant_id,
                    expected_status=current_operation.status,
                    next_record=operation_result(
                        current_operation, result_ref, result_digest
                    ),
                )
                break
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                reloaded = await self._port.get_operation(
                    operation.operation_id, tenant_id=tenant_id
                )
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = reloaded
                if attempt == 1 and current_operation.status in {
                    OperationStatus.PENDING,
                    OperationStatus.EFFECT_UNKNOWN,
                }:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.debug(
            "execution cancellation race resolved: execution=%s status=%s operation=%s",
            execution_id,
            current.status.value,
            operation.operation_id,
        )
        return CancelExecutionResult(
            execution_id, current.status is ExecutionStatus.CANCELLED
        )


    async def result_from_execution(
        self, execution: ExecutionRecord, *, tenant_id: str
    ) -> ExecutionResult:
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise execution_recovery_error(execution)
        if execution.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.EXECUTION_NOT_READY)
        result = await self._port.result_record(execution, tenant_id=tenant_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        error_code, safe_details = terminal_error(execution)
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            if result.output is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                return ExecutionResult(
                    execution.execution_id,
                    execution.status,
                    None,
                    result.usage,
                    error_code,
                    safe_details,
                    execution.error_diagnostics,
                )
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if result.output.kind == "inline":
                decoded = result.output.decode()
                if not isinstance(
                    decoded, (str, dict, list, int, float, bool, type(None))
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                output = decoded
            else:
                reference = result.output.ref
                if reference is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = await self._port.read_output(reference)
                if result.output.encoding == "utf-8":
                    output = payload.decode("utf-8")
                else:
                    output = json.loads(payload.decode("utf-8"))
            return ExecutionResult(
                execution.execution_id,
                execution.status,
                output,
                result.usage,
                error_code,
                safe_details,
            )
        except AIError:
            raise
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


    async def load_authorized(
        self, execution_id: str, principal: Principal, action: AuthorizationAction
    ) -> ExecutionRecord:
        header = await self.authorize_execution(execution_id, principal, action)
        record = await self._port.read_execution(execution_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self.authorize_loaded_execution(
            record, principal, action, owner_principal_id=header.owner_principal_id,
        )
        return record


    async def authorize_loaded_execution(
        self, execution: ExecutionRecord, principal: Principal, action: AuthorizationAction,
        *, owner_principal_id: str | None,
    ) -> None:
        if owner_principal_id is not None and execution.principal_id != owner_principal_id:
            # A source winner retains its own authorization policy, even when
            # an earlier remote observation described a different principal.
            header = await self._port.read_source_header(execution.execution_id, tenant_id=principal.tenant_id)
            if header is None:
                header = ResourceRef(
                    ResourceKind.EXECUTION, execution.execution_id, principal.tenant_id,
                    execution.principal_id,
                )
            await self._authorization.authorize(principal, action, header)


def terminal_error(
    execution: ExecutionRecord,
) -> "tuple[str | None, Mapping[str, JsonValue]]":
    details = dict(execution.safe_error_details)
    if execution.status is ExecutionStatus.SUCCEEDED:
        if execution.error_code is not None or details:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None, {}
    if execution.status is ExecutionStatus.CANCELLED:
        if execution.error_code != ErrorCode.EXECUTION_CANCELLED.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return execution.error_code, details
    if execution.status is not ExecutionStatus.FAILED or execution.error_code is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(execution.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if code is ErrorCode.EXECUTION_CANCELLED:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return code.value, details


def execution_recovery_error(execution: ExecutionRecord) -> AIError:
    if (
        execution.status is not ExecutionStatus.RECOVERY_REQUIRED
        or execution.error_code is None
        or execution.error_diagnostics is not None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(execution.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return AIError(code, safe_details=execution.safe_error_details)


def operation_result(
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
        operation.operation_kind,
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


def operation_failure(
    operation: OperationLedgerRecord, error_code: str
) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.operation_kind,
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


def operation_status(
    operation: OperationLedgerRecord, status: OperationStatus
) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.operation_kind,
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


def next_execution(
    record: ExecutionRecord,
    status: ExecutionStatus,
    now: datetime,
    *,
    error_code: "str | None" = None,
    safe_error_details: "Mapping[str, JsonValue] | None" = None,
    error_diagnostics: "ErrorDiagnostics | None" = None,
    agent_run_seq: int | None = None,
    terminal_event: bool = False,
) -> ExecutionRecord:
    return replace(
        record,
        status=status,
        revision=record.revision + 1,
        event_seq=record.event_seq + (1 if terminal_event else 0),
        agent_run_seq=record.agent_run_seq
        if agent_run_seq is None
        else agent_run_seq,
        error_code=error_code,
        safe_error_details=(
            record.safe_error_details
            if safe_error_details is None
            else dict(safe_error_details)
        ),
        error_diagnostics=error_diagnostics,
        updated_at=now,
    )


def terminal_idempotency_matches(
    identity: IdempotencyRecord,
    execution: ExecutionRecord,
) -> bool:
    if execution.started_at is None or execution.result is None:
        return False
    expected_status = (
        ExecutionStatus.SUCCEEDED
        if identity.status is IdempotencyStatus.COMPLETED
        else ExecutionStatus.FAILED
        if identity.status is IdempotencyStatus.FAILED
        else ExecutionStatus.CANCELLED
        if identity.status is IdempotencyStatus.CANCELLED
        else None
    )
    if expected_status is None or execution.status is not expected_status:
        return False
    expected_digest = (
        None if execution.result.output is None else execution.result.output.digest
    )
    return (
        identity.resource_kind is ResourceKind.EXECUTION
        and identity.resource_id == execution.execution_id
        and identity.result_digest == expected_digest
        and identity.error_code == execution.error_code
    )


def stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    if error_code is None:
        return AIError(fallback)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ['ExecutionManager', 'ExecutionManagementPort', 'ExecutionDriver', 'CancelEffectOutcome', 'terminal_error', 'execution_recovery_error', 'operation_result', 'operation_failure', 'operation_status', 'next_execution', 'terminal_idempotency_matches', 'stable_idempotency_error', 'stable_operation_error']
