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
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Principal,
    ResourceKind,
    StopReason,
    ToolOperationStatus,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload, payload_fits_inline
from ._execution import DefaultExecutionService, _consumed_query
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
    ExecutionRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)
from .state._plan import RuntimeDomain
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
            execution_operations=self._execution.operations,
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
        effects = await self.recovery_effects(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if not effects:
            return await super()._commit_failure(execution, error, run_id=run_id)
        return await self._commit_recovery_required(execution, unknown, effects)

    async def _commit_recovery_required(
        self,
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[ExecutionRecoveryEffect, ...],
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
        details: dict[str, JsonValue] = {
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
            payload = {
                "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                "safe_error_details": details,
            }
            self._live_broker.publish_event(
                current.execution_id,
                ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                payload,
                durable_sequence=committed.event_sequence,
            )
            self._live_broker.complete(current.execution_id)
        return committed

    async def _reconcile_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
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
        if (
            execution is not None
            and execution.status
            in {ExecutionStatus.STARTED, ExecutionStatus.CANCELLING}
            and checkpoint.handoff_phase is RecoveryHandoffPhase.NONE
            and checkpoint.state
            in {
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }
        ):
            self._validate_recovery_identity(execution, checkpoint.input)
            effects = await self.recovery_effects(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if effects:
                first = effects[0]
                await self._commit_recovery_required(
                    execution,
                    AIError(
                        ErrorCode.TOOL_EFFECT_UNKNOWN,
                        safe_details={
                            "execution_id": execution.execution_id,
                            "operation_id": first.operation_id,
                            "fence": first.fence,
                            "phase": "startup_reconcile",
                        },
                    ),
                    effects,
                )
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
                idempotency_key_digest=record.idempotency_key_digest,
                replay_safe=record.replay_safe,
                error_code=record.error_code,
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
        tool = await self._tool_operations.get_operation(
            request.operation_id,
            tenant_id=current.tenant_id,
        )
        if tool is None or tool.execution_id != execution_id:
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
            error_payload = await self._tool_resolution_error_payload(current)
        else:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

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
                "resolution": type(request.resolution).__name__,
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
        ledger = OperationLedgerInput(
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
            ledger,
            expected_fence=request.expected_fence,
            target_status=target_status,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )
        return ToolEffectResolutionResult(
            operation_id=resolved.tool_operation_id,
            execution_id=resolved.execution_id,
            status=resolved.status,
            fence=resolved.fence,
        )

    async def persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord:
        return await self._recovery_commands_for(
            execution.execution_id
        ).commit_cancel_intent(execution, operation)

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
        cancel_operations = await self._pending_cancel_operations(
            execution_id,
            tenant_id=tenant_id,
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
            return await self._complete_recovered_cancel(
                resumed,
                checkpoint,
                cancel_operations,
            )
        await super()._reconcile_checkpoint(checkpoint)
        latest = await self._execution.executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return latest

    async def _pending_cancel_operations(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[OperationLedgerRecord, ...]:
        values = await self._execution.operations.list_pending(
            ResourceKind.EXECUTION,
            execution_id,
            tenant_id=tenant_id,
            limit=257,
        )
        cancel = tuple(
            value
            for value in values
            if value.operation_kind is OperationKind.EXECUTION_CANCEL
        )
        if len(cancel) > 256:
            raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
        return cancel

    async def _complete_recovered_cancel(
        self,
        resumed: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        operations: tuple[OperationLedgerRecord, ...],
    ) -> ExecutionRecord:
        cancelling = await self._recovery_commands_for(
            resumed.execution_id
        ).commit_cancel_claim(resumed)
        terminal = await self._commit_terminal(
            cancelling,
            ExecutionStatus.CANCELLED,
            None,
            ErrorCode.EXECUTION_CANCELLED.value,
            StopReason.CANCELLED,
            run_id=checkpoint.step_run_id,
        )
        for candidate in operations:
            await self._settle_cancel_operation(candidate, terminal)
        return terminal

    async def _settle_cancel_operation(
        self,
        operation: OperationLedgerRecord,
        execution: ExecutionRecord,
    ) -> None:
        current = await self._execution.operations.get(
            operation.operation_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.CANCELLED,
        }:
            return
        if current.status not in {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = OperationLedgerRecord(
            current.operation_id,
            current.tenant_id,
            current.resource_kind,
            current.resource_id,
            current.execution_id,
            current.operation_kind,
            OperationStatus.SUCCEEDED,
            current.request_digest,
            execution.execution_id,
            None,
            None,
            current.compactable,
            current.sequence,
            current.created_at,
            datetime.now(timezone.utc),
        )
        try:
            await self._execution.operations.compare_and_swap(
                current.operation_id,
                tenant_id=execution.tenant_id,
                expected_status=current.status,
                next_record=updated,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                ErrorCode.STORAGE_CONFLICT,
            }:
                raise
            latest = await self._execution.operations.get(
                current.operation_id,
                tenant_id=execution.tenant_id,
            )
            if (
                latest is not None
                and latest.status is OperationStatus.SUCCEEDED
                and latest.result_ref == execution.execution_id
                and latest.request_digest == current.request_digest
            ):
                return
            raise

    async def _tool_result_payload(
        self,
        execution: ExecutionRecord,
        operation_id: str,
        result: object,
    ) -> StoredPayload:
        encoded = encode_model_messages(
            (
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "runtime",
                            result,
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
    ) -> StoredPayload:
        encoded = json.dumps(
            {
                "kind": "error",
                "code": ErrorCode.TOOL_EXECUTION_FAILED.value,
                "safe_details": {"phase": "tool_effect_resolution"},
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
                RuntimeDomain.RECOVERY,
                execution.tenant_id,
                data,
            )
        )


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
        execution = await self._load_authorized(
            execution_id,
            request.principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
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

    @_consumed_query
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

    @_consumed_query
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
            execution = await self._load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_READ,
            )
            if self._local_stream_abort is not None:
                self._local_stream_abort(execution_id)
            while True:
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
                execution = await self._load_authorized(
                    execution_id,
                    principal,
                    AuthorizationAction.EXECUTION_READ,
                )

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
            current = await self._recovery_backend().persist_cancel_intent(
                execution,
                candidate,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            latest_execution = await self._state.executions.get(
                execution_id,
                tenant_id=request.principal.tenant_id,
            )
            if (
                latest_execution is not None
                and latest_execution.status is not ExecutionStatus.RECOVERY_REQUIRED
            ):
                return await super()._cancel(execution_id, request)
            raise
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
