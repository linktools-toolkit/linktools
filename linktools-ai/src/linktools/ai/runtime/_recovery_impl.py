#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery-aware Runtime implementations composed by the local factory."""

import asyncio
import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from ..core import (
    AuthorizationAction,
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceKind,
    StopReason,
    ToolOperationStatus,
    canonical_json_bytes,
    canonical_sha256,
    idempotency_key_digest,
    normalize_json_value,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload, payload_fits_inline
from ._execution import DefaultExecutionService
from ._local import LocalExecutionBackend
from ._message import encode_model_messages
from ._object import RuntimeObjectKeyFactory, put_runtime_object
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectApplied,
    ToolEffectFailed,
    ToolEffectNotApplied,
    ToolEffectResolutionResult,
)
from .service_api import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionResult,
)
from .state._contracts import (
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRecord,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)
from .state._recovery_commands import RuntimeRecoveryCommands


class RecoveryLocalExecutionBackend(LocalExecutionBackend):
    """Extend the local backend with quiescent execution recovery."""

    def _recovery_commands_for(self, execution_id: str) -> RuntimeRecoveryCommands:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return RuntimeRecoveryCommands(
            self._execution.executions,
            self._execution.events,
            self._recovery.operations,
            self._tool_operations,
            background_tasks=self._execution_task_set(execution_id),
        )

    async def _commit_failure(
        self,
        execution: ExecutionRecord,
        error: Exception,
        *,
        run_id: str | None = None,
    ) -> ExecutionRecord:
        unknown = _tool_effect_unknown_cause(error)
        if unknown is None:
            return await super()._commit_failure(execution, error, run_id=run_id)
        return await self._commit_recovery_required(execution, unknown)

    async def _commit_recovery_required(
        self,
        execution: ExecutionRecord,
        error: AIError,
    ) -> ExecutionRecord:
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return current
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.CANCELLING,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        effects = await self.recovery_effects(
            current.execution_id,
            tenant_id=current.tenant_id,
        )
        details: dict[str, object] = {
            "execution_id": current.execution_id,
            "phase": "tool_effect",
        }
        requested_operation_id = error.safe_details.get("operation_id")
        selected = None
        if isinstance(requested_operation_id, str):
            selected = next(
                (
                    value
                    for value in effects
                    if value.operation_id == requested_operation_id
                ),
                None,
            )
            if selected is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif len(effects) == 1:
            selected = effects[0]
        if selected is not None:
            details.update(
                {
                    "operation_id": selected.operation_id,
                    "step_run_id": selected.step_run_id,
                    "tool_call_id": selected.tool_call_id,
                    "tool_name": selected.tool_name,
                    "fence": selected.fence,
                }
            )
        elif not effects:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            details["unknown_effect_count"] = len(effects)

        async with self._audit_lock(current.execution_id):
            pending = tuple(
                self._pending_audit_events.get(current.execution_id, ())
            )
            committed = await self._recovery_commands_for(
                current.execution_id
            ).commit_recovery_required(
                current,
                error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                safe_error_details=details,
                audit_events=pending,
            )
            if pending:
                self._pending_audit_events.pop(current.execution_id, None)
            self._confirm_committed_events(
                current.execution_id,
                pending_count=len(pending),
                durable_sequence=committed.event_sequence,
            )
            event_payload = {
                "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                "safe_error_details": details,
            }
            self._live_broker.publish_event(
                current.execution_id,
                ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                event_payload,
                durable_sequence=committed.event_sequence,
            )
            self._live_broker.complete(current.execution_id)
        return committed

    async def _reconcile_checkpoint(self, checkpoint) -> None:
        execution = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if execution is not None and execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            self._validate_recovery_identity(execution, checkpoint.input)
            if (
                checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
                or checkpoint.state
                not in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.WAITING,
                }
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        await super()._reconcile_checkpoint(checkpoint)

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        records = await self._tool_operations.list_by_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        return tuple(
            ExecutionRecoveryEffect(
                operation_id=record.tool_operation_id,
                execution_id=record.execution_id,
                step_run_id=record.step_run_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                fence=record.fence,
                status=record.status.value,
            )
            for record in records
            if record.status is ToolOperationStatus.EFFECT_UNKNOWN
        )

    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult:
        current = await self._execution.executions.get(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        operation = await self._tool_operations.get_operation(
            request.operation_id,
            tenant_id=current.tenant_id,
        )
        if operation is None or operation.execution_id != execution_id:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        result_payload = None
        error_code = None
        error_payload = None
        if isinstance(request.resolution, ToolEffectApplied):
            target_status = ToolOperationStatus.COMPLETED
            result_payload = await self._tool_result_payload(
                current,
                request.operation_id,
                request.resolution.result,
            )
        elif isinstance(request.resolution, ToolEffectNotApplied):
            target_status = ToolOperationStatus.PENDING
        elif isinstance(request.resolution, ToolEffectFailed):
            target_status = ToolOperationStatus.FAILED
            error_code = ErrorCode.TOOL_EXECUTION_FAILED.value
            error_payload = await self._tool_resolution_error_payload(
                current,
                request.resolution,
            )
        else:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        resolution_kind = type(request.resolution).__name__
        payload_digest = (
            result_payload.digest
            if result_payload is not None
            else error_payload.digest
            if error_payload is not None
            else None
        )
        request_digest = canonical_sha256(
            {
                "kind": "tool_effect_resolution",
                "execution_id": execution_id,
                "operation_id": request.operation_id,
                "expected_fence": request.expected_fence,
                "resolution": resolution_kind,
                "payload_digest": payload_digest,
            }
        )
        resolution_operation_id = canonical_sha256(
            {
                "scope": "execution.tool_effect.resolve",
                "tenant_id": current.tenant_id,
                "execution_id": execution_id,
                "idempotency_key_digest": idempotency_key_digest(
                    request.idempotency_key
                ),
            }
        )
        result_digest = canonical_sha256(
            {
                "operation_id": request.operation_id,
                "fence": request.expected_fence,
                "status": target_status.value,
                "payload_digest": payload_digest,
            }
        )
        now = datetime.now(timezone.utc)
        resolution_operation = OperationLedgerInput(
            resolution_operation_id,
            current.tenant_id,
            ResourceKind.TOOL_OPERATION,
            request.operation_id,
            execution_id,
            OperationKind.TOOL_EFFECT_RESOLVE,
            OperationStatus.SUCCEEDED,
            request_digest,
            request.operation_id,
            result_digest,
            None,
            True,
            now,
            now,
        )
        resolved = await self._recovery_commands_for(
            execution_id
        ).resolve_tool_effect(
            resolution_operation,
            expected_fence=request.expected_fence,
            target_status=target_status,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )
        return ToolEffectResolutionResult(
            operation_id=resolved.tool_operation_id,
            status=resolved.status.value,
            fence=resolved.fence,
        )

    async def recover_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord:
        current = await self._execution.executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        unresolved = await self.recovery_effects(
            execution_id,
            tenant_id=tenant_id,
        )
        if unresolved:
            first = unresolved[0]
            raise AIError(
                ErrorCode.TOOL_EFFECT_UNKNOWN,
                safe_details={
                    "execution_id": execution_id,
                    "operation_id": first.operation_id,
                    "fence": first.fence,
                    "phase": "execution_recover",
                },
            )
        checkpoint = await self._recovery.checkpoints.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state
            not in {
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }
            or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        pending = await self._execution.operations.list_pending(
            ResourceKind.EXECUTION,
            execution_id,
            tenant_id=tenant_id,
            limit=257,
        )
        cancel_operations = tuple(
            value
            for value in pending
            if value.operation_kind is OperationKind.EXECUTION_CANCEL
        )
        resumed = await self._recovery_commands_for(
            execution_id
        ).commit_resumed(current)
        self._live_broker.publish_event(
            execution_id,
            ExecutionEventType.EXECUTION_RESUMED,
            {},
            durable_sequence=resumed.event_sequence,
        )
        if cancel_operations:
            operation = cancel_operations[0]
            cancelling = await self.commit_cancel_checkpoint(
                ExecutionCancelRequestCommit(
                    execution_id,
                    tenant_id,
                    resumed.revision,
                    resumed.event_sequence,
                    operation.operation_id,
                    datetime.now(timezone.utc),
                ),
                expected_status=ExecutionStatus.STARTED,
            )
            terminal = await self._commit_terminal(
                cancelling,
                ExecutionStatus.CANCELLED,
                None,
                ErrorCode.EXECUTION_CANCELLED.value,
                StopReason.CANCELLED,
                run_id=checkpoint.step_run_id,
            )
            for pending_operation in cancel_operations:
                latest = await self._execution.operations.get(
                    pending_operation.operation_id,
                    tenant_id=tenant_id,
                )
                if latest is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if latest.status in {
                    OperationStatus.SUCCEEDED,
                    OperationStatus.CANCELLED,
                }:
                    continue
                if latest.status not in {
                    OperationStatus.PENDING,
                    OperationStatus.RUNNING,
                }:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._execution.operations.compare_and_swap(
                    latest.operation_id,
                    tenant_id=tenant_id,
                    expected_status=latest.status,
                    next_record=latest.__class__(
                        latest.operation_id,
                        latest.tenant_id,
                        latest.resource_kind,
                        latest.resource_id,
                        latest.execution_id,
                        latest.operation_kind,
                        OperationStatus.SUCCEEDED,
                        latest.request_digest,
                        execution_id,
                        None,
                        None,
                        latest.compactable,
                        latest.sequence,
                        latest.created_at,
                        datetime.now(timezone.utc),
                    ),
                )
            return terminal
        await super()._reconcile_checkpoint(checkpoint)
        latest = await self._execution.executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return latest

    async def _tool_result_payload(
        self,
        execution: ExecutionRecord,
        operation_id: str,
        result: object,
    ) -> StoredPayload:
        normalized = normalize_json_value(result)
        encoded = encode_model_messages(
            (
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "runtime",
                            normalized,
                            tool_call_id=operation_id,
                        )
                    ]
                ),
            )
        )
        return await self._recovery_payload(execution, encoded)

    async def _tool_resolution_error_payload(
        self,
        execution: ExecutionRecord,
        resolution: ToolEffectFailed,
    ) -> StoredPayload:
        encoded = json.dumps(
            {
                "kind": "error",
                "code": ErrorCode.TOOL_EXECUTION_FAILED.value,
                "safe_details": {
                    "phase": "tool_effect_resolution",
                    "reason": resolution.reason,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return await self._recovery_payload(execution, encoded)

    async def _recovery_payload(
        self,
        execution: ExecutionRecord,
        data: bytes,
    ) -> StoredPayload:
        inline = StoredPayload.inline_bytes(data)
        if payload_fits_inline(inline, self._payload_policy):
            return inline
        return StoredPayload.object(
            await put_runtime_object(
                self._recovery_objects,
                RuntimeObjectKeyFactory(self._namespace),
                self._recovery_domain(),
                execution.tenant_id,
                data,
            )
        )

    @staticmethod
    def _recovery_domain():
        from .state._plan import RuntimeDomain

        return RuntimeDomain.RECOVERY


class RecoveryExecutionService(DefaultExecutionService):
    """Expose the explicit recovery control plane through ExecutionService."""

    def _recovery_backend(self) -> RecoveryLocalExecutionBackend:
        backend = self._backend
        if not isinstance(backend, RecoveryLocalExecutionBackend):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return backend

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        execution = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            return ()
        return await self._recovery_backend().recovery_effects(
            execution_id,
            tenant_id=principal.tenant_id,
        )

    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult:
        await self._load_authorized(
            execution_id,
            request.principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        return await self._recovery_backend().resolve_tool_effect(
            execution_id,
            request,
        )

    async def recover(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionHandle:
        execution = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._recovery_backend().recover_execution(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        return ExecutionHandle(execution_id)

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        execution = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_READ,
        )
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise _execution_recovery_error(execution)
        return await super().result(execution_id, principal=principal)

    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def wait_once() -> ExecutionResult:
            while True:
                execution = await self._load_authorized(
                    execution_id,
                    principal,
                    AuthorizationAction.EXECUTION_READ,
                )
                if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
                    raise _execution_recovery_error(execution)
                if execution.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    return await super(RecoveryExecutionService, self).result(
                        execution_id,
                        principal=principal,
                    )
                if self._backend is not None:
                    failure = self._backend.worker_failure(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                    if failure is not None:
                        raise failure
                waiter = self._local_waiter
                if waiter is not None and waiter.owns_execution(
                    execution_id,
                    tenant_id=principal.tenant_id,
                ):
                    await waiter.wait_terminal(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                else:
                    await asyncio.sleep(1.0)

        try:
            return await asyncio.wait_for(wait_once(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT) from error

    async def _cancel(
        self,
        execution_id: str,
        request: CancelExecutionRequest,
    ) -> CancelExecutionResult:
        execution = await self._load_authorized(
            execution_id,
            request.principal,
            AuthorizationAction.EXECUTION_CANCEL,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            return await super()._cancel(execution_id, request)
        operation_digest = canonical_sha256(
            {
                "action": "execution.cancel",
                "principal": principal_identity_payload(request.principal),
                "execution_id": execution_id,
                "force": request.force,
            }
        )
        operation_id = idempotency_key_digest(request.idempotency_key)
        current = await self._state.operations.get(
            operation_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            now = datetime.now(timezone.utc)
            candidate = OperationLedgerInput(
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
            try:
                current = await self._state.operations.append(candidate)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                current = await self._state.operations.get(
                    operation_id,
                    tenant_id=request.principal.tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        if (
            current.request_digest != operation_digest
            or current.resource_kind is not ResourceKind.EXECUTION
            or current.resource_id != execution_id
            or current.execution_id != execution_id
            or current.operation_kind is not OperationKind.EXECUTION_CANCEL
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if current.status is OperationStatus.FAILED:
            try:
                code = ErrorCode(current.error_code or "")
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise AIError(code)
        return CancelExecutionResult(execution_id, False)


def _tool_effect_unknown_cause(error: BaseException) -> AIError | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, AIError) and current.code is ErrorCode.TOOL_EFFECT_UNKNOWN:
            return current
        current = current.__cause__ or current.__context__
    return None


def _execution_recovery_error(execution: ExecutionRecord) -> AIError:
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


__all__ = ["RecoveryExecutionService", "RecoveryLocalExecutionBackend"]
