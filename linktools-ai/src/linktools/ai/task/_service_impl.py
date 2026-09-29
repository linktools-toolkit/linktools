#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed TaskGraph service independent of Runtime composition."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, cast, runtime_checkable

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    JsonValue,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..observe import MetricRecorder
from ._event import TaskEvent
from ._handler import TaskEffectResolution
from ._graph import (
    CancelGraphRequest,
    RecoverGraphRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphState,
    TaskGraphView,
    TaskInputSupplyRequest,
    TaskNode,
    TaskNodeInfo,
    TaskNodeResult,
    TaskNodeView,
)
from ._metrics import _TaskMetricProjector
from ._service import (
    TaskEffectResolutionRequest,
    TaskGraphLauncher,
    TaskGraphService,
)

_logger = environ.get_logger("ai.task.service")
_TASK_EVENT_READ_LIMIT = 200
_MAX_PENDING_GRAPH_OPERATIONS = 128


@runtime_checkable
class _TaskMetricProjectorBinder(Protocol):
    def bind_metric_projector(self, projector: _TaskMetricProjector) -> None: ...


class _LocalTaskWaiter(Protocol):
    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool: ...

    def graph_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None: ...

    def graph_failure(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> AIError | None: ...

    async def wait_graph_activity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: "int | None" = None,
    ) -> None: ...


class _TaskGraphPreflight(Protocol):
    def admit_request(self, graph: TaskGraph) -> TaskGraph: ...

    async def capture_admission(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraph: ...

    async def load_admission(
        self,
        admission: TaskGraphAdmission,
    ) -> None: ...

    def validate_recovery(self, state: TaskGraphState) -> None: ...

    async def prepare_graph(
        self,
        state: TaskGraphState,
        *,
        principal: Principal,
    ) -> None: ...

    async def release_graph_dependencies(
        self,
        state: TaskGraphState,
        *,
        tenant_id: str,
    ) -> None: ...

    def validate_input(self, node: TaskNode, value: JsonValue) -> None: ...

    def validate_effect_resolution(
        self,
        node: TaskNode,
        resolution: TaskEffectResolution,
    ) -> None: ...


class _TaskRepository(Protocol):
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None: ...

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None: ...

    async def graph_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState | None: ...

    async def result_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskGraph, int] | None: ...

    async def get_node_states(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]: ...

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]: ...

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None: ...

    async def scheduler_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState: ...

    async def recover_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        cancel_requested: bool = False,
    ) -> TaskGraphView: ...

    async def requeue_recovery(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        expected_fence: int,
        execution_id: "str | None" = None,
        next_attempt_at: "datetime | None" = None,
    ) -> TaskGraphView: ...

    async def cancel_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView: ...

    async def register_cancel_request(
        self,
        operation: OperationLedgerInput,
    ) -> tuple[OperationLedgerRecord, bool]: ...

    async def cancel_node(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        return await self._settle_execution_cancellation(
            graph_id,
            node_id,
            execution_id,
            request,
        )

    async def _settle_execution_cancellation(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        header = await self._persistence.tasks.get_header(
            graph_id,
            tenant_id=tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.TASK_CANCEL,
            header,
        )
        request_digest = canonical_sha256(
            {
                "action": "task.cancel_node",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "node_id": node_id,
                "execution_id": execution_id,
                "force": request.force,
            }
        )
        operation_id = idempotency_key_digest(request.idempotency_key)
        claimed, operation = await self._claim_cancel_operation(
            operation_id,
            tenant_id,
            graph_id,
            request_digest,
            execution_id=execution_id,
        )
        if (
            operation.operation_kind is not OperationKind.TASK_CANCEL
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != graph_id
            or operation.execution_id != execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        if state is None or state.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = TaskGraphView(
            graph_state.graph_id,
            graph_state.status,
            graph_state.nodes,
        )

        if operation.status is OperationStatus.SUCCEEDED:
            return view
        if not claimed and operation.status is OperationStatus.RUNNING:
            return view

        if not _terminal(state.status):
            view = await self._persistence.tasks.cancel_node(
                graph_id,
                node_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                cancel_confirmed=False,
                expected_fence=state.fence,
            )
            graph_state = await self._persistence.tasks.graph_state(
                graph_id,
                tenant_id=tenant_id,
            )
            if graph_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            state = next(
                (item for item in graph_state.node_states if item.node_id == node_id),
                None,
            )
            if state is None or state.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if self._launcher is not None and not _terminal(state.status):
            admission = await self._persistence.admissions.get(
                graph_id,
                tenant_id=tenant_id,
            )
            if admission is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await self._launcher.cancel_node(
                    admission.launch(),
                    node_id,
                    execution_id,
                    invoke_effects=claimed,
                )
            except BaseException as error:  # noqa: BLE001
                if isinstance(error, asyncio.CancelledError):
                    raise
                await self._record_effect_unknown(operation, tenant_id)
                if isinstance(error, AIError):
                    raise
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_node_cancel",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            graph_state = await self._persistence.tasks.graph_state(
                graph_id,
                tenant_id=tenant_id,
            )
            if graph_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            state = next(
                (item for item in graph_state.node_states if item.node_id == node_id),
                None,
            )
            if state is None or state.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            view = TaskGraphView(
                graph_state.graph_id,
                graph_state.status,
                graph_state.nodes,
            )

        if state.status is TaskStatus.RECOVERY_REQUIRED:
            operation = await self._record_effect_unknown(operation, tenant_id)
        elif _terminal(state.status):
            operation = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if _terminal(view.status):
            await self._observe_metric_history(view, tenant_id=tenant_id)
        return view

    async def cancel(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        header = await self._persistence.tasks.get_header(
            graph_id,
            tenant_id=tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.TASK_CANCEL,
            header,
        )
        request_digest = canonical_sha256(
            {
                "action": "task.cancel",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "force": request.force,
            }
        )
        finalizer = asyncio.create_task(
            self._cancel_finalizer(
                graph_id,
                request,
                idempotency_key_digest(request.idempotency_key),
                request_digest,
            ),
            name=f"task-cancel-finalizer-{tenant_id}-{graph_id}",
        )
        try:
            return await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            if finalizer.done():
                return finalizer.result()
            self._detach_finalizer(cast("asyncio.Task[object]", finalizer), graph_id)
            raise

    async def _claim_cancel_operation(
        self,
        operation_id: str,
        tenant_id: str,
        graph_id: str,
        request_digest: str,
        *,
        execution_id: str | None = None,
    ) -> tuple[bool, OperationLedgerRecord]:
        operation = await self._persistence.operations.get(
            operation_id,
            tenant_id=tenant_id,
        )
        if operation is None:
            now = datetime.now(timezone.utc)
            pending = OperationLedgerInput(
                operation_id,
                tenant_id,
                ResourceKind.TASK_GRAPH,
                graph_id,
                execution_id,
                OperationKind.TASK_CANCEL,
                OperationStatus.PENDING,
                request_digest,
                None,
                None,
                None,
                True,
                now,
                now,
            )
            try:
                operation, _created = (
                    await self._persistence.tasks.register_cancel_request(pending)
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                operation = await self._persistence.operations.get(
                    operation_id,
                    tenant_id=tenant_id,
                )
                if operation is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if (
            operation.operation_kind is not OperationKind.TASK_CANCEL
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != graph_id
            or operation.tenant_id != tenant_id
            or operation.execution_id != execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if operation.status is OperationStatus.PENDING:
            running = replace(
                operation,
                status=OperationStatus.RUNNING,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                claimed = await self._persistence.operations.compare_and_swap(
                    operation_id,
                    tenant_id=tenant_id,
                    expected_status=OperationStatus.PENDING,
                    next_record=running,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                current = await self._persistence.operations.get(
                    operation_id,
                    tenant_id=tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if current.request_digest != request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if current.status is OperationStatus.PENDING:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                operation = current
            else:
                return True, claimed
        if operation.status is OperationStatus.SUCCEEDED:
            return False, operation
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            return False, operation
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _cancel_finalizer(
        self,
        graph_id: str,
        request: CancelGraphRequest,
        operation_id: str,
        request_digest: str,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        claimed, operation = await self._claim_cancel_operation(
            operation_id,
            tenant_id,
            graph_id,
            request_digest,
        )
        try:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
        except BaseException as error:  # noqa: BLE001
            if claimed or operation.status in {
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )
            raise
        if view is None:
            if claimed or operation.status in {
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if claimed or operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            try:
                view = await self._persistence.tasks.cancel_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                reloaded = await self._persistence.tasks.get_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                view = reloaded
            except BaseException as error:  # noqa: BLE001
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )
        elif operation.status is OperationStatus.SUCCEEDED:
            return view

        if not _terminal(view.status):
            try:
                await self._cleanup_graph_runtime(
                    view,
                    request.principal,
                    invoke_effects=claimed,
                )
                reloaded = await self._persistence.tasks.get_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                view = reloaded
            except BaseException as error:  # noqa: BLE001
                if isinstance(error, asyncio.CancelledError):
                    raise
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )

        if view.status is TaskStatus.RECOVERY_REQUIRED:
            if operation.status is OperationStatus.RUNNING:
                await self._record_effect_unknown(operation, tenant_id)
            _logger.info(
                "task graph cancel deferred for recovery: tenant=%s graph=%s",
                tenant_id,
                graph_id,
            )
            return view

        if view.status is TaskStatus.CANCELLED:
            try:
                await self._cleanup_graph_runtime(
                    view,
                    request.principal,
                    invoke_effects=claimed,
                )
                reloaded = await self._persistence.tasks.get_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                view = reloaded
            except BaseException as error:  # noqa: BLE001
                if isinstance(error, asyncio.CancelledError):
                    raise
                return await self._settle_cancel_error(
                    operation,
                    graph_id,
                    request,
                    error,
                )

        if not _terminal(view.status):
            return view
        if claimed or operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }:
            current = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._observe_metric_history(view, tenant_id=tenant_id)
        _logger.info(
            "task graph cancel settled: tenant=%s graph=%s status=%s",
            tenant_id,
            graph_id,
            view.status.value,
        )
        return view

    async def _settle_cancel_error(
        self,
        operation: OperationLedgerRecord,
        graph_id: str,
        request: CancelGraphRequest,
        error: BaseException,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        try:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
        except Exception as reload_error:
            await self._record_effect_unknown(operation, tenant_id)
            if isinstance(error, AIError):
                raise error
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={"phase": "task_cancel_readback"},
            ) from reload_error
        if view is None:
            await self._record_failure(
                operation,
                tenant_id,
                ErrorCode.STORAGE_INTEGRITY_ERROR.value,
            )
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if view.status is TaskStatus.RECOVERY_REQUIRED:
            try:
                await self._cleanup_graph_runtime(
                    view,
                    request.principal,
                    invoke_effects=False,
                )
            except BaseException as cleanup_error:  # noqa: BLE001
                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise
                await self._record_effect_unknown(operation, tenant_id)
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_cancel_recovery_quiesce",
                        "graph_id": graph_id,
                    },
                ) from cleanup_error
            await self._record_effect_unknown(operation, tenant_id)
            return view
        if _terminal(view.status):
            if view.status is TaskStatus.CANCELLED:
                try:
                    await self._cleanup_graph_runtime(
                        view,
                        request.principal,
                        invoke_effects=False,
                    )
                except BaseException as cleanup_error:  # noqa: BLE001
                    await self._raise_cancel_cleanup_error(
                        operation,
                        tenant_id,
                        graph_id,
                        cleanup_error,
                    )
            current = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await self._observe_metric_history(view, tenant_id=tenant_id)
            return view
        if isinstance(error, AIError) and error.code in {
            ErrorCode.STORAGE_COMMIT_UNKNOWN,
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            ErrorCode.EXECUTION_START_UNKNOWN,
            ErrorCode.TASK_EFFECT_UNKNOWN,
            ErrorCode.TOOL_EFFECT_UNKNOWN,
        }:
            await self._record_effect_unknown(operation, tenant_id)
            raise error
        if isinstance(error, AIError):
            await self._record_failure(
                operation,
                tenant_id,
                error.code.value,
            )
            raise error
        await self._record_effect_unknown(operation, tenant_id)
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            safe_details={"phase": "task_cancel"},
        ) from error

    async def _raise_cancel_cleanup_error(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        graph_id: str,
        error: BaseException,
    ) -> None:
        await self._record_effect_unknown(operation, tenant_id)
        if isinstance(error, asyncio.CancelledError):
            raise error
        if isinstance(error, AIError):
            raise error
        raise AIError(
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            safe_details={"phase": "task_cancel_cleanup", "graph_id": graph_id},
        ) from error

    async def _cleanup_graph_runtime(
        self,
        view: TaskGraphView,
        principal: Principal,
        *,
        invoke_effects: bool = True,
    ) -> None:
        if self._launcher is None:
            return
        tenant_id = principal.tenant_id
        admission = await self._persistence.admissions.get(
            view.graph_id,
            tenant_id=tenant_id,
        )
        if admission is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            launch = admission.launch()
            if launch.principal.tenant_id != tenant_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._launcher.settle_cancel(
                launch,
                invoke_effects=invoke_effects,
            )
        except asyncio.CancelledError:
            raise
        except AIError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={
                    "phase": "task_cancel_cleanup",
                    "graph_id": view.graph_id,
                },
            ) from error

    async def _record_success(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        view: TaskGraphView,
        *,
        expected_status: OperationStatus = OperationStatus.RUNNING,
    ) -> OperationLedgerRecord:
        completed = replace(
            operation,
            status=OperationStatus.SUCCEEDED,
            result_ref=view.graph_id,
            result_digest=canonical_sha256({"graph_id": view.graph_id}),
            error_code=None,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=expected_status,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.SUCCEEDED:
                if (
                    current.result_ref != completed.result_ref
                    or current.result_digest != completed.result_digest
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return current
            if current.status is OperationStatus.FAILED:
                raise _stable_operation_error(current.error_code)
            raise

    async def _record_failure(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        error_code: str,
    ) -> OperationLedgerRecord:
        failed = replace(
            operation,
            status=OperationStatus.FAILED,
            error_code=error_code,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=failed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.FAILED:
                return current
            raise

    async def _record_effect_unknown(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
    ) -> OperationLedgerRecord:
        if operation.status is OperationStatus.EFFECT_UNKNOWN:
            return operation
        unknown = replace(
            operation,
            status=OperationStatus.EFFECT_UNKNOWN,
            error_code=None,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=unknown,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(
                operation.operation_id,
                tenant_id=tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {
                OperationStatus.EFFECT_UNKNOWN,
                OperationStatus.SUCCEEDED,
            }:
                return current
            raise

    async def _observe_metric_history(
        self,
        view: TaskGraphView | TaskGraphState,
        *,
        tenant_id: str,
    ) -> None:
        if _terminal(view.status) and self._preflight is not None:
            state = (
                view
                if isinstance(view, TaskGraphState)
                else await self._persistence.tasks.scheduler_state(
                    view.graph_id,
                    tenant_id=tenant_id,
                )
            )
            await self._preflight.release_graph_dependencies(
                state,
                tenant_id=tenant_id,
            )
        if self._metric_projector is not None:
            self._metric_projector.trigger(view.graph_id, tenant_id=tenant_id)

    async def _result(
        self,
        view: TaskGraphView,
        tenant_id: str,
    ) -> TaskGraphResult:
        state = await self._persistence.tasks.graph_state(
            view.graph_id,
            tenant_id=tenant_id,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _state_result(state)

    def _detach_finalizer(
        self,
        task: "asyncio.Task[object]",
        graph_id: str,
        *,
        label: str = "task graph cancel finalizer",
    ) -> None:
        if task in self._detached_finalizers:
            return
        self._detached_finalizers.add(task)

        def consume(done: "asyncio.Task[object]") -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except BaseException as error:  # noqa: BLE001
                _logger.warning(
                    "%s failed after caller cancellation: graph=%s error=%s",
                    label,
                    graph_id,
                    type(error).__name__,
                )
                if self._detached_finalizer_failure is None:
                    self._detached_finalizer_failure = _task_service_failure(
                        error,
                        phase="task_service_finalizer",
                        graph_id=graph_id,
                    )
            finally:
                self._detached_finalizers.discard(done)

        task.add_done_callback(consume)

    async def drain_owned_finalizers(self) -> None:
        while True:
            pending = tuple(
                task for task in self._detached_finalizers if not task.done()
            )
            if not pending:
                await asyncio.sleep(0)
                return
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )

    async def drain_metric_projector(self) -> None:
        if self._metric_projector is not None:
            await self._metric_projector.close()

    async def preflight_close(self) -> None:
        pending = tuple(task for task in self._detached_finalizers if not task.done())
        if pending:
            _logger.warning(
                "task service close blocked by detached finalizers: tasks=%s",
                len(pending),
            )
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "task_service_preflight_close",
                    "pending_finalizers": len(pending),
                },
            )
        failure = self._detached_finalizer_failure
        if failure is not None:
            raise AIError(
                failure.code,
                category=failure.category,
                retryable=failure.retryable,
                operation_id=failure.operation_id,
                safe_details=dict(failure.safe_details),
                diagnostics=failure.diagnostics,
            )


def _validate_event_window(after_sequence: int, limit: int) -> None:
    if (
        isinstance(after_sequence, bool)
        or not isinstance(after_sequence, int)
        or after_sequence < 0
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _validate_event_page(
    graph_id: str,
    after_sequence: int,
    page: Page[TaskEvent],
) -> None:
    expected = after_sequence
    for event in page.items:
        if event.graph_id != graph_id or event.sequence != expected + 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = event.sequence


def _task_service_failure(
    error: BaseException,
    *,
    phase: str,
    graph_id: str,
    durable_admitted: bool = False,
) -> AIError:
    details = dict(error.safe_details) if isinstance(error, AIError) else {}
    details.setdefault("phase", phase)
    details.setdefault("graph_id", graph_id)
    if durable_admitted:
        details.setdefault("durable_admitted", True)
    if isinstance(error, AIError):
        return AIError(
            error.code,
            category=error.category,
            retryable=error.retryable,
            operation_id=error.operation_id,
            safe_details=details,
            diagnostics=error.diagnostics,
        )
    return AIError(
        ErrorCode.STORAGE_RECOVERY_REQUIRED,
        safe_details=details,
    )


def _terminal(status: TaskStatus) -> bool:
    return status in {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.BLOCKED,
    }


def _observation_boundary(status: TaskStatus) -> bool:
    return (
        _terminal(status)
        or status is TaskStatus.RECOVERY_REQUIRED
        or status is TaskStatus.WAITING
    )


def _stable_waiting_state(state: TaskGraphState) -> bool:
    unfinished = tuple(
        state
        for state in state.node_states
        if state.status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    )
    return bool(unfinished) and any(
        state.status is TaskStatus.WAITING for state in unfinished
    ) and all(
        state.status not in {TaskStatus.READY, TaskStatus.RUNNING}
        for state in unfinished
    )


def _has_wait_bound_node(state: TaskGraphState) -> bool:
    state_by_id = {node.node_id: node for node in state.node_states}
    for node in state.nodes:
        node_state = state_by_id[node.node_id]
        if node_state.status is TaskStatus.WAITING and not (
            node.task is not None
            and node.task.id == "linktools.ai.input"
            and node.task.revision == 1
        ):
            return True
    return False


def _state_result(state: TaskGraphState) -> TaskGraphResult:
    return _node_result(state.graph_id, state.status, state.node_states)


def _node_result(
    graph_id: str,
    status: TaskStatus,
    nodes: tuple[TaskNodeView, ...],
) -> TaskGraphResult:
    results = tuple(
        TaskNodeResult(
            node.node_id,
            node.status,
            node.result_digest,
            node.execution_id,
            node.error_code,
            node.error_digest,
        )
        for node in nodes
    )
    return TaskGraphResult(graph_id, status, results)


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["DefaultTaskGraphService", "TaskPersistence"]
