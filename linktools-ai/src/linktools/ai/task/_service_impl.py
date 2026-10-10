#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed TaskGraph service independent of Runtime composition."""

import asyncio
import math
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, cast, runtime_checkable

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    RunBudget,
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
from ._submission import (
    TaskGraphSubmission,
    TaskSubmissionCancellation,
    TaskSubmissionRef,
    TaskSubmissionResult,
)
from ._metrics import _TaskMetricProjector
from ._service import (
    TaskBoundExecutionRecovery,
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
    """Own resolved execution dependencies, including admitted run-budget scopes.

    Capture creates a new scope before dispatch; load validates the existing
    scope without resetting it. Implementations must honor admission.budget.
    """

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
        after_event_seq: int,
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
        *,
        tenant_id: str,
        execution_id: str,
        cancel_confirmed: bool = False,
        expected_fence: int | None = None,
    ) -> TaskGraphView: ...

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]: ...

    async def complete(
        self,
        lease: object | None,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        graph_id: str | None = None,
        node_id: str | None = None,
    ) -> object: ...


class _OperationRepository(Protocol):
    async def append(
        self,
        record: OperationLedgerInput,
    ) -> OperationLedgerRecord: ...

    async def get(
        self,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> OperationLedgerRecord | None: ...

    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord: ...

    async def list_pending(
        self,
        resource_kind: ResourceKind,
        resource_id: str,
        *,
        tenant_id: str,
        limit: int,
        states: "frozenset[OperationStatus] | None" = None,
    ) -> tuple[OperationLedgerRecord, ...]: ...


class _TaskAdmissionPersistence(Protocol):
    async def admit_prepared(self, submission: TaskGraphSubmission) -> TaskGraphView: ...

    @property
    def namespace(self) -> str: ...

    async def prepare(self, submission: TaskGraphSubmission) -> TaskGraphSubmission: ...

    async def cancel_submission(
        self, submission: TaskSubmissionRef, operation: OperationLedgerInput
    ) -> bool: ...

    async def admit(
        self, admission: TaskGraphAdmission, graph: TaskGraph
    ) -> TaskGraphView: ...

    async def get(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphAdmission | None: ...

    async def list_recoverable_page(
        self,
        *,
        cursor: "str | None",
        limit: int,
    ) -> Page[TaskGraphLaunch]: ...


class TaskPersistence(Protocol):
    tasks: _TaskRepository
    operations: _OperationRepository
    admissions: _TaskAdmissionPersistence


class DefaultTaskGraphService(TaskGraphService):
    """Own durable TaskGraph submission, observation, recovery, and cancellation."""

    def __init__(
        self,
        persistence: TaskPersistence,
        authorization: AuthorizationPolicy,
        launcher: TaskGraphLauncher | None = None,
        *,
        local_waiter: "_LocalTaskWaiter | None" = None,
        preflight: "_TaskGraphPreflight | None" = None,
        bound_execution_recovery: TaskBoundExecutionRecovery | None = None,
        metric_recorder: MetricRecorder | None = None,
        metric_source_namespace: str | None = None,
    ) -> None:
        if (metric_recorder is None) != (metric_source_namespace is None):
            raise ValueError(
                "task metric recorder and source namespace must be configured together"
            )
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._local_waiter = local_waiter
        self._preflight = preflight
        self._bound_execution_recovery = bound_execution_recovery
        self._metric_projector = (
            None
            if metric_recorder is None or metric_source_namespace is None
            else _TaskMetricProjector(
                persistence.tasks,
                metric_recorder,
                source_namespace=metric_source_namespace,
                admissions=persistence.admissions,
            )
        )
        if (
            self._metric_projector is not None
            and isinstance(launcher, _TaskMetricProjectorBinder)
        ):
            launcher.bind_metric_projector(self._metric_projector)
        self._detached_finalizers: set[asyncio.Task[object]] = set()
        self._detached_finalizer_failure: AIError | None = None

    def _require_budget_owner(self, budget: RunBudget | None) -> None:
        if budget is not None and self._preflight is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                safe_details={"reason": "run_budget_owner_missing"},
                retryable=False,
            )

    async def start(self, request: TaskGraphRequest) -> TaskGraphResult:
        return await self._start_graph(request)

    async def _start_graph(
        self,
        request: TaskGraphRequest,
    ) -> TaskGraphResult:
        submission = await self.prepare_submission(request)
        return (await self.start_prepared(submission)).result

    async def prepare_submission(
        self, request: TaskGraphRequest
    ) -> TaskGraphSubmission:
        return await self.prepare_described(await self.describe_submission(request))

    async def describe_submission(
        self, request: TaskGraphRequest
    ) -> TaskGraphSubmission:
        self._require_budget_owner(request.budget)
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.TASK_RUN,
            ResourceRef(ResourceKind.TASK_GRAPH, request.graph.graph_id, request.principal.tenant_id),
        )
        if self._preflight is not None:
            request = TaskGraphRequest(
                self._preflight.admit_request(request.graph), request.principal,
                request.idempotency_key, request.limits, request.correlation, request.budget,
            )
        return TaskGraphSubmission(self._persistence.admissions.namespace,
                                   TaskGraphAdmission.from_request(request), request.graph)

    async def prepare_described(
        self, submission: TaskGraphSubmission
    ) -> TaskGraphSubmission:
        self._require_budget_owner(submission.admission.budget)
        if submission.namespace != self._persistence.admissions.namespace:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        admission = submission.admission
        await self._authorization.authorize(admission.principal, AuthorizationAction.TASK_RUN,
            ResourceRef(ResourceKind.TASK_GRAPH, admission.graph_id, admission.principal.tenant_id))
        status = await self._persistence.admissions.submission_status(submission.ref)
        if status is None and self._preflight is not None:
            graph = await self._preflight.capture_admission(admission, submission.graph)
            submission = TaskGraphSubmission(submission.namespace, admission, graph)
        return await self._persistence.admissions.prepare(submission)

    async def start_prepared(
        self, submission: TaskGraphSubmission
    ) -> TaskSubmissionResult:
        self._require_budget_owner(submission.admission.budget)
        admission = submission.admission
        graph_id = admission.graph_id
        tenant_id = admission.principal.tenant_id
        await self._authorization.authorize(
            admission.principal,
            AuthorizationAction.TASK_RUN,
            ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id),
        )
        if submission.namespace != self._persistence.admissions.namespace:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        view = await self._persistence.admissions.admit_prepared(submission)
        durable_admission = await self._persistence.admissions.get(
            graph_id,
            tenant_id=tenant_id,
        )
        if durable_admission is None:
            if view.status is not TaskStatus.CANCELLED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskSubmissionResult(
                submission.ref, False, TaskGraphResult(graph_id, TaskStatus.CANCELLED),
            )
        if self._preflight is not None:
            await self._preflight.load_admission(durable_admission)
        state = await self._persistence.tasks.scheduler_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if self._preflight is not None:
            self._preflight.validate_recovery(state)
        if _terminal(view.status):
            await self._observe_metric_history(state, tenant_id=tenant_id)
        else:
            if self._preflight is not None:
                await self._preflight.prepare_graph(
                    state,
                    principal=admission.principal,
                )
            if view.status is not TaskStatus.RECOVERY_REQUIRED:
                await self._arm_graph(durable_admission.launch())
        return TaskSubmissionResult(
            submission.ref, True, await self._result(view, tenant_id),
        )

    async def cancel_submission(
        self,
        submission: TaskSubmissionRef,
        *,
        principal: Principal,
        idempotency_key: str,
    ) -> TaskSubmissionCancellation:
        request = CancelGraphRequest(principal, idempotency_key=idempotency_key)
        if (
            submission.namespace != self._persistence.admissions.namespace
            or submission.tenant_id != principal.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        await self._authorization.authorize(
            principal,
            AuthorizationAction.TASK_CANCEL,
            ResourceRef(
                ResourceKind.TASK_GRAPH, submission.graph_id, submission.tenant_id,
                submission.principal.principal_id,
            ),
        )
        operation_id = idempotency_key_digest(idempotency_key)
        request_digest = canonical_sha256({
            "action": "task.cancel",
            "principal": principal_identity_payload(principal),
            "graph_id": submission.graph_id,
            "force": request.force,
        })

        async def finalize() -> TaskSubmissionCancellation:
            now = datetime.now(timezone.utc)
            admitted = await self._persistence.admissions.cancel_submission(
                submission,
                OperationLedgerInput(
                    operation_id, principal.tenant_id, ResourceKind.TASK_GRAPH,
                    submission.graph_id, None, OperationKind.TASK_CANCEL,
                    OperationStatus.PENDING, request_digest, None, None, None,
                    True, now, now,
                ),
            )
            if not admitted:
                return TaskSubmissionCancellation(submission, False, TaskStatus.CANCELLED)
            view = await self._cancel_finalizer(
                submission.graph_id, request, operation_id, request_digest,
            )
            return TaskSubmissionCancellation(submission, True, view.status)

        finalizer = asyncio.create_task(finalize())
        try:
            return await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            if finalizer.done():
                return finalizer.result()
            self._detach_finalizer(
                cast("asyncio.Task[object]", finalizer), submission.graph_id,
            )
            raise


    async def _arm_graph(self, launch: TaskGraphLaunch) -> None:
        self._require_budget_owner(launch.budget)
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        task = asyncio.create_task(
            self._launcher.start(launch),
            name=f"task-scheduler-arm-{launch.principal.tenant_id}-{launch.graph_id}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                self._detach_finalizer(
                    cast("asyncio.Task[object]", task),
                    launch.graph_id,
                    label="task scheduler arm",
                )
                raise
            try:
                task.result()
            except asyncio.CancelledError as error:
                raise _task_service_failure(
                    error,
                    phase="task_scheduler_arm",
                    graph_id=launch.graph_id,
                    durable_admitted=True,
                ) from error
            except BaseException as error:  # noqa: BLE001
                raise _task_service_failure(
                    error,
                    phase="task_scheduler_arm",
                    graph_id=launch.graph_id,
                    durable_admitted=True,
                ) from error
            raise
        except BaseException as error:  # noqa: BLE001
            raise _task_service_failure(
                error,
                phase="task_scheduler_arm",
                graph_id=launch.graph_id,
                durable_admitted=True,
            ) from error

    async def _arm_committed_graph_if_runnable(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> None:
        if any(
            operation.execution_id is None
            for operation in await self._pending_cancel_operations(
                graph_id,
                tenant_id=tenant_id,
            )
        ):
            return
        state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.status not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            return
        admission = await self._validated_recovery_admission(
            graph_id,
            tenant_id,
            state,
        )
        await self._arm_graph(admission.launch())

    async def recover_pending(self) -> None:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        cursor: str | None = None
        recovered = 0
        while True:
            page = await self._persistence.admissions.list_recoverable_page(
                cursor=cursor,
                limit=128,
            )
            for launch in page.items:
                admission = await self._persistence.admissions.get(
                    launch.graph_id,
                    tenant_id=launch.principal.tenant_id,
                )
                if admission is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._require_budget_owner(admission.budget)
                state = await self._persistence.tasks.scheduler_state(
                    launch.graph_id,
                    tenant_id=launch.principal.tenant_id,
                )
                if self._preflight is not None:
                    await self._preflight.load_admission(admission)
                    self._preflight.validate_recovery(state)
                view = TaskGraphView(
                    state.graph_id,
                    state.status,
                    state.nodes,
                )
                if _terminal(view.status):
                    await self._observe_metric_history(
                        state,
                        tenant_id=launch.principal.tenant_id,
                    )
                    continue
                if self._preflight is not None:
                    await self._preflight.prepare_graph(
                        state,
                        principal=launch.principal,
                    )
                if view.status is TaskStatus.RECOVERY_REQUIRED:
                    continue
                await self._arm_graph(launch)
                recovered += 1
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        _logger.info("task graph recovery scan completed: graphs=%s", recovered)

    async def run(
        self,
        request: TaskGraphRequest,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        submitted = await self._start_graph(request)
        if _terminal(submitted.status):
            return submitted
        state = await self.wait(
            submitted.graph_id,
            principal=request.principal,
            timeout_seconds=timeout_seconds,
        )
        return _state_result(state)

    async def recover(
        self,
        graph_id: str,
        request: RecoverGraphRequest,
    ) -> TaskGraphResult:
        tenant_id = request.principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_RUN,
            principal=request.principal,
        )
        initial = await self._persistence.tasks.get_graph(
            graph_id,
            tenant_id=tenant_id,
        )
        if initial is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation_id = idempotency_key_digest(request.idempotency_key)
        existing = await self._persistence.operations.get(
            operation_id,
            tenant_id=tenant_id,
        )
        pending_cancel_operations = await self._pending_cancel_operations(
            graph_id,
            tenant_id=tenant_id,
        )
        if (
            existing is None
            and not pending_cancel_operations
            and initial.status not in {
                TaskStatus.RECOVERY_REQUIRED,
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
            }
        ):
            raise AIError(ErrorCode.TASK_NOT_READY)
        request_digest = canonical_sha256(
            {
                "action": "task.recover",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
            }
        )
        operation = await self._claim_recover_operation(
            operation_id,
            tenant_id,
            graph_id,
            request_digest,
        )
        view = await self._persistence.tasks.get_graph(
            graph_id,
            tenant_id=tenant_id,
        )
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if operation.status is OperationStatus.SUCCEEDED:
            await self._arm_committed_graph_if_runnable(
                graph_id,
                tenant_id=tenant_id,
            )
            return await self._result(view, tenant_id)
        cancel_operations = await self._pending_cancel_operations(
            graph_id,
            tenant_id=tenant_id,
        )
        graph_cancel_operations = tuple(
            selected
            for selected in cancel_operations
            if selected.execution_id is None
        )
        node_cancel_operations = tuple(
            selected
            for selected in cancel_operations
            if selected.execution_id is not None
        )
        cancel_requested = bool(graph_cancel_operations)
        if graph_cancel_operations:
            if not _terminal(view.status):
                await self._cleanup_graph_runtime(
                    view,
                    request.principal,
                    invoke_effects=True,
                )
                refreshed = await self._persistence.tasks.get_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if refreshed is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                view = refreshed
            if _terminal(view.status):
                for cancel_operation in cancel_operations:
                    settled_cancel = await self._record_success(
                        cancel_operation,
                        tenant_id,
                        view,
                        expected_status=cancel_operation.status,
                    )
                    if settled_cancel.status is not OperationStatus.SUCCEEDED:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                await self._observe_metric_history(view, tenant_id=tenant_id)
            settled = await self._record_success(
                operation,
                tenant_id,
                view,
                expected_status=operation.status,
            )
            if settled.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await self._result(view, tenant_id)

        if node_cancel_operations:
            graph_state = await self._persistence.tasks.graph_state(
                graph_id,
                tenant_id=tenant_id,
            )
            if graph_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            view = TaskGraphView(
                graph_state.graph_id,
                graph_state.status,
                graph_state.nodes,
            )
            for cancel_operation in node_cancel_operations:
                execution_id = cancel_operation.execution_id
                if execution_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                node_state = next(
                    (
                        selected
                        for selected in graph_state.node_states
                        if selected.execution_id == execution_id
                    ),
                    None,
                )
                if node_state is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if not _terminal(node_state.status) and self._launcher is not None:
                    admission = await self._persistence.admissions.get(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                    if admission is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await self._launcher.cancel_node(
                        admission.launch(),
                        node_state.node_id,
                        execution_id,
                        invoke_effects=True,
                    )
                    graph_state = await self._persistence.tasks.graph_state(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                    if graph_state is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    view = TaskGraphView(
                        graph_state.graph_id,
                        graph_state.status,
                        graph_state.nodes,
                    )
                    node_state = next(
                        (
                            selected
                            for selected in graph_state.node_states
                            if selected.execution_id == execution_id
                        ),
                        None,
                    )
                    if node_state is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if not _terminal(node_state.status):
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                settled_cancel = await self._record_success(
                    cancel_operation,
                    tenant_id,
                    TaskGraphView(
                        graph_state.graph_id,
                        graph_state.status,
                        graph_state.nodes,
                    ),
                    expected_status=cancel_operation.status,
                )
                if settled_cancel.status is not OperationStatus.SUCCEEDED:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

        admission = None
        if (
            view.status
            in {
                TaskStatus.RECOVERY_REQUIRED,
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
            }
            and not (
                view.status is TaskStatus.RECOVERY_REQUIRED
                and cancel_requested
            )
        ):
            state = await self._persistence.tasks.scheduler_state(
                graph_id,
                tenant_id=tenant_id,
            )
            admission = await self._validated_recovery_admission(
                graph_id,
                tenant_id,
                state,
            )

        if admission is not None and self._bound_execution_recovery is not None:
            await self._bound_execution_recovery.recover_bound_executions(
                graph_id,
                request,
            )

        if view.status is TaskStatus.RECOVERY_REQUIRED:
            view = await self._persistence.tasks.recover_graph(
                graph_id,
                tenant_id=tenant_id,
                cancel_requested=cancel_requested,
            )
        if view.status is TaskStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        if view.status is TaskStatus.CANCELLED:
            await self._cleanup_graph_runtime(view, request.principal)
            for cancel_operation in cancel_operations:
                settled = await self._record_success(
                    cancel_operation,
                    tenant_id,
                    view,
                    expected_status=cancel_operation.status,
                )
                if settled.status is not OperationStatus.SUCCEEDED:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif _terminal(view.status):
            await self._observe_metric_history(view, tenant_id=tenant_id)
        else:
            if admission is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            launch = admission.launch()
            if self._preflight is not None:
                state = await self._persistence.tasks.scheduler_state(
                    graph_id,
                    tenant_id=tenant_id,
                )
                await self._preflight.prepare_graph(
                    state,
                    principal=launch.principal,
                )
            await self._arm_graph(launch)

        settled = await self._record_success(
            operation,
            tenant_id,
            view,
            expected_status=operation.status,
        )
        if settled.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info(
            "task graph recovery settled: tenant=%s graph=%s status=%s",
            tenant_id,
            graph_id,
            view.status.value,
        )
        return await self._result(view, tenant_id)

    async def _validated_recovery_admission(
        self,
        graph_id: str,
        tenant_id: str,
        state: TaskGraphState,
    ) -> TaskGraphAdmission:
        admission = await self._persistence.admissions.get(
            graph_id,
            tenant_id=tenant_id,
        )
        if (
            admission is None
            or admission.graph_id != graph_id
            or admission.principal.tenant_id != tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._require_budget_owner(admission.budget)
        if self._preflight is not None:
            await self._preflight.load_admission(admission)
            self._preflight.validate_recovery(state)
        return admission

    async def _authorize_graph(
        self,
        graph_id: str,
        action: AuthorizationAction,
        *,
        principal: Principal,
    ) -> None:
        header = await self._persistence.tasks.get_header(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal,
            action,
            header,
        )

    async def recovery_nodes(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> tuple[TaskNodeInfo, ...]:
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_RUN,
            principal=principal,
        )
        state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(TaskNodeInfo.from_node(node) for node in state.nodes)

    async def resume(
        self,
        graph_id: str,
        node_id: str,
        request: TaskInputSupplyRequest,
    ) -> TaskGraphResult:
        tenant_id = request.principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_RUN,
            principal=request.principal,
        )
        graph_state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        node = next(
            (item for item in graph_state.nodes if item.node_id == node_id),
            None,
        )
        if state is None or node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        admission = await self._validated_recovery_admission(
            graph_id,
            tenant_id,
            graph_state,
        )
        if self._preflight is not None:
            self._preflight.validate_input(node, request.value)

        value_digest = canonical_sha256(request.value)
        operation_id = idempotency_key_digest(request.idempotency_key)
        request_digest = canonical_sha256(
            {
                "action": "task.resume",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "node_id": node_id,
                "wait_id": request.wait_id,
                "value": request.value,
            }
        )
        operation = await self._persistence.operations.get(
            operation_id,
            tenant_id=tenant_id,
        )
        if operation is None:
            if state.status is not TaskStatus.WAITING:
                raise AIError(ErrorCode.TASK_NOT_READY)
            if state.execution_id != request.wait_id:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            now = datetime.now(timezone.utc)
            pending = OperationLedgerInput(
                operation_id,
                tenant_id,
                ResourceKind.TASK_GRAPH,
                graph_id,
                node_id,
                OperationKind.TASK_NODE,
                OperationStatus.RUNNING,
                request_digest,
                None,
                None,
                None,
                True,
                now,
                now,
            )
            try:
                operation = await self._persistence.operations.append(pending)
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
            operation.tenant_id != tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != graph_id
            or operation.execution_id != node_id
            or operation.operation_kind is not OperationKind.TASK_NODE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if operation.status is OperationStatus.SUCCEEDED:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if view is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._arm_committed_graph_if_runnable(
                graph_id,
                tenant_id=tenant_id,
            )
            return await self._result(view, tenant_id)
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status is not OperationStatus.RUNNING:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        latest = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (item for item in latest.node_states if item.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.execution_id != request.wait_id:
            await self._record_failure(
                operation,
                tenant_id,
                ErrorCode.IDEMPOTENCY_CONFLICT.value,
            )
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)

        if state.status is TaskStatus.SUCCEEDED:
            if state.result_digest != value_digest:
                await self._record_failure(
                    operation,
                    tenant_id,
                    ErrorCode.IDEMPOTENCY_CONFLICT.value,
                )
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if view is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif state.status is TaskStatus.WAITING:
            if self._launcher is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                view = await self._launcher.supply_input(
                    admission.launch(),
                    node_id,
                    request.wait_id,
                    request.value,
                )
            except AIError as error:
                if error.code not in {ErrorCode.TASK_NOT_READY, ErrorCode.STORAGE_CONFLICT}:
                    raise
                latest = await self._persistence.tasks.graph_state(graph_id, tenant_id=tenant_id)
                accepted = None if latest is None else next(
                    (item for item in latest.node_states if item.node_id == node_id), None)
                if (accepted is None or accepted.execution_id != request.wait_id or
                        accepted.status is not TaskStatus.SUCCEEDED or accepted.result_digest != value_digest):
                    raise
                view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
                if view is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        else:
            raise AIError(ErrorCode.TASK_NOT_READY)

        completed = replace(
            operation,
            status=OperationStatus.SUCCEEDED,
            result_ref=graph_id,
            result_digest=value_digest,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            settled = await self._persistence.operations.compare_and_swap(
                operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            settled = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
            if (settled is None or settled.request_digest != operation.request_digest
                    or settled.result_digest != value_digest
                    or settled.status is not OperationStatus.SUCCEEDED):
                raise
        if settled.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        if view.status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
            TaskStatus.RECOVERY_REQUIRED,
        }:
            if self._preflight is not None:
                self._preflight.validate_recovery(
                    await self._persistence.tasks.scheduler_state(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                )
            await self._arm_graph(admission.launch())

        _logger.info(
            "task input accepted: graph=%s node=%s wait_id=%s",
            graph_id,
            node_id,
            request.wait_id,
        )
        return await self._result(view, tenant_id)

    async def resolve_effect(
        self,
        graph_id: str,
        node_id: str,
        request: TaskEffectResolutionRequest,
    ) -> TaskGraphResult:
        tenant_id = request.principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_RUN,
            principal=request.principal,
        )
        graph_state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        node = next(
            (item for item in graph_state.nodes if item.node_id == node_id),
            None,
        )
        if state is None or node is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        admission = await self._validated_recovery_admission(
            graph_id,
            tenant_id,
            graph_state,
        )
        if self._preflight is not None:
            self._preflight.validate_effect_resolution(
                node,
                request.resolution,
            )

        operation_id = idempotency_key_digest(request.idempotency_key)
        request_digest = canonical_sha256(
            {
                "action": "task.resolve_effect",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "node_id": node_id,
                "expected_fence": request.expected_fence,
                "resolution": {
                    "kind": request.resolution.kind,
                    "value": request.resolution.value,
                },
            }
        )
        operation = await self._persistence.operations.get(
            operation_id,
            tenant_id=tenant_id,
        )
        if operation is None:
            if (
                state.status is not TaskStatus.RECOVERY_REQUIRED
                or state.fence != request.expected_fence
                or state.execution_id is None
            ):
                raise AIError(ErrorCode.TASK_NOT_READY)
            now = datetime.now(timezone.utc)
            pending = OperationLedgerInput(
                operation_id,
                tenant_id,
                ResourceKind.TASK_GRAPH,
                graph_id,
                node_id,
                OperationKind.TASK_NODE,
                OperationStatus.RUNNING,
                request_digest,
                None,
                None,
                None,
                True,
                now,
                now,
            )
            try:
                operation = await self._persistence.operations.append(pending)
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
            operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != graph_id
            or operation.execution_id != node_id
            or operation.operation_kind is not OperationKind.TASK_NODE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if operation.status is OperationStatus.SUCCEEDED:
            view = await self._persistence.tasks.get_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if view is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._arm_committed_graph_if_runnable(
                graph_id,
                tenant_id=tenant_id,
            )
            return await self._result(view, tenant_id)
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status is not OperationStatus.RUNNING:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        latest = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (item for item in latest.node_states if item.node_id == node_id),
            None,
        )
        if (
            state is None
            or state.execution_id is None
            or state.fence != request.expected_fence
        ):
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        view = await self._launcher.resolve_effect(
            admission.launch(),
            node_id,
            state.execution_id,
            request.expected_fence,
            request.resolution,
        )
        settled = await self._record_success(operation, tenant_id, view)
        if settled.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        if view.status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
            TaskStatus.RECOVERY_REQUIRED,
        }:
            if self._preflight is not None:
                self._preflight.validate_recovery(
                    await self._persistence.tasks.scheduler_state(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                )
            await self._arm_graph(admission.launch())

        _logger.info(
            "task effect resolved: graph=%s node=%s kind=%s fence=%s",
            graph_id,
            node_id,
            request.resolution.kind,
            request.expected_fence,
        )
        return await self._result(view, tenant_id)

    async def _claim_recover_operation(
        self,
        operation_id: str,
        tenant_id: str,
        graph_id: str,
        request_digest: str,
    ) -> OperationLedgerRecord:
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
                None,
                OperationKind.TASK_RECOVER,
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
                operation = await self._persistence.operations.append(pending)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                operation = await self._persistence.operations.get(
                    operation_id,
                    tenant_id=tenant_id,
                )
                if operation is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        if (
            operation.tenant_id != tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != graph_id
            or operation.operation_kind is not OperationKind.TASK_RECOVER
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if operation.status is OperationStatus.PENDING:
            running = replace(
                operation,
                status=OperationStatus.RUNNING,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                return await self._persistence.operations.compare_and_swap(
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
                operation = current
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status not in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
            OperationStatus.SUCCEEDED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return operation

    async def _pending_cancel_operations(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[OperationLedgerRecord, ...]:
        pending = await self._persistence.operations.list_pending(
            ResourceKind.TASK_GRAPH,
            graph_id,
            tenant_id=tenant_id,
            limit=_MAX_PENDING_GRAPH_OPERATIONS + 1,
            states=frozenset(
                {
                    OperationStatus.PENDING,
                    OperationStatus.RUNNING,
                    OperationStatus.EFFECT_UNKNOWN,
                }
            ),
        )
        if len(pending) > _MAX_PENDING_GRAPH_OPERATIONS:
            raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
        result = tuple(
            operation
            for operation in pending
            if operation.operation_kind is OperationKind.TASK_CANCEL
        )
        for operation in result:
            if (
                operation.resource_kind is not ResourceKind.TASK_GRAPH
                or operation.resource_id != graph_id
                or operation.tenant_id != tenant_id
                or operation.status
                not in {
                    OperationStatus.PENDING,
                    OperationStatus.RUNNING,
                    OperationStatus.EFFECT_UNKNOWN,
                }
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result

    async def inspect(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphView:
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        view = await self._persistence.tasks.get_graph(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return view

    async def state(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphState:
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        state = await self._persistence.tasks.graph_state(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return state

    async def result_header(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> tuple[TaskGraph, int]:
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        result_header = await self._persistence.tasks.result_header(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if result_header is None or result_header[0].graph_id != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result_header

    async def result_node_states(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        principal: Principal,
    ) -> tuple[TaskNodeView, ...]:
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        states = await self._persistence.tasks.get_node_states(
            graph_id,
            node_ids,
            tenant_id=principal.tenant_id,
        )
        if tuple(state.node_id for state in states) != node_ids:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return states

    async def list_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]:
        _validate_event_window(after_event_seq, limit)
        tenant_id = principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        page = await self._persistence.tasks.list_events(
            graph_id,
            tenant_id=tenant_id,
            after_event_seq=after_event_seq,
            limit=limit,
        )
        _validate_event_page(graph_id, after_event_seq, page)
        if after_event_seq == 0 and (
            not page.items or page.items[0].event_type.value != "GRAPH_ADMITTED"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if after_event_seq > 0 and not page.items:
            latest = await self._persistence.tasks.latest_event(
                graph_id,
                tenant_id=tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return page

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
    ) -> AsyncIterator[TaskEvent]:
        return self._observe_graph_events(
            graph_id,
            principal=principal,
            after_event_seq=after_event_seq,
        )

    async def _observe_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int,
    ) -> AsyncIterator[TaskEvent]:
        _validate_event_window(after_event_seq, _TASK_EVENT_READ_LIMIT)
        tenant_id = principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_READ,
            principal=principal,
        )
        async for event in self._observe_graph_events_authorized(
            graph_id,
            tenant_id=tenant_id,
            after_event_seq=after_event_seq,
        ):
            yield event

    async def _observe_graph_events_authorized(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_event_seq: int,
    ) -> AsyncIterator[TaskEvent]:
        cursor = after_event_seq
        pending_wait_error: AIError | None = None
        fallback_backoff = 1.0
        while True:
            generation = self._local_activity_generation(
                graph_id,
                tenant_id=tenant_id,
            )
            page = await self._persistence.tasks.list_events(
                graph_id,
                tenant_id=tenant_id,
                after_event_seq=cursor,
                limit=_TASK_EVENT_READ_LIMIT,
            )
            _validate_event_page(graph_id, cursor, page)
            if cursor == 0 and (
                not page.items or page.items[0].event_type.value != "GRAPH_ADMITTED"
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if page.items:
                pending_wait_error = None
                fallback_backoff = 1.0
                for event in page.items:
                    cursor = event.event_seq
                    yield event
                    if event.node_id is None and _observation_boundary(event.status):
                        return
                continue
            latest = await self._persistence.tasks.latest_event(
                graph_id,
                tenant_id=tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                latest.node_id is None
                and _observation_boundary(latest.status)
                and latest.event_seq <= cursor
            ):
                return
            if pending_wait_error is not None:
                raise pending_wait_error
            try:
                fallback_backoff = await self._wait_graph_activity_opportunity(
                    graph_id,
                    tenant_id=tenant_id,
                    after_generation=generation,
                    fallback_backoff=fallback_backoff,
                )
            except AIError as error:
                pending_wait_error = error

    def _local_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None:
        waiter = self._local_waiter
        if waiter is None or not waiter.owns_graph(graph_id, tenant_id=tenant_id):
            return None
        return waiter.graph_activity_generation(graph_id, tenant_id=tenant_id)

    async def _wait_graph_activity_opportunity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: int | None,
        fallback_backoff: float,
    ) -> float:
        waiter = self._local_waiter
        if (
            waiter is None
            or not waiter.owns_graph(graph_id, tenant_id=tenant_id)
            or after_generation is None
        ):
            await asyncio.sleep(fallback_backoff)
            return min(30.0, fallback_backoff * 2)
        try:
            await asyncio.wait_for(
                waiter.wait_graph_activity(
                    graph_id,
                    tenant_id=tenant_id,
                    after_generation=after_generation,
                ),
                timeout=fallback_backoff,
            )
        except asyncio.TimeoutError:
            return min(30.0, fallback_backoff * 2)
        return 1.0

    async def wait(
        self,
        graph_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphState:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def consume() -> TaskGraphState:
            tenant_id = principal.tenant_id
            await self._authorize_graph(
                graph_id,
                AuthorizationAction.TASK_READ,
                principal=principal,
            )
            fallback_backoff = 1.0
            while True:
                generation = self._local_activity_generation(
                    graph_id,
                    tenant_id=tenant_id,
                )
                state = await self._persistence.tasks.graph_state(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if state is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if _terminal(state.status):
                    await self._observe_metric_history(state, tenant_id=tenant_id)
                    return state
                if state.status is TaskStatus.RECOVERY_REQUIRED:
                    return state
                waiter = self._local_waiter
                if state.wait_status is TaskStatus.WAITING and (
                    not _has_wait_bound_node(state)
                    or waiter is None
                    or not waiter.owns_graph(graph_id, tenant_id=tenant_id)
                ):
                    return state
                if waiter is not None:
                    failure = waiter.graph_failure(graph_id, tenant_id=tenant_id)
                    if failure is not None:
                        raise failure
                if generation is not None and self._local_activity_generation(
                    graph_id, tenant_id=tenant_id,
                ) is None:
                    # A retired local owner may have committed after our read.
                    fallback_backoff = 1.0
                    continue
                fallback_backoff = await self._wait_graph_activity_opportunity(
                    graph_id,
                    tenant_id=tenant_id,
                    after_generation=generation,
                    fallback_backoff=fallback_backoff,
                )

        try:
            if timeout_seconds is None:
                return await consume()
            return await asyncio.wait_for(consume(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(
                ErrorCode.TASK_WAIT_TIMEOUT,
                safe_details={"graph_id": graph_id},
            ) from error

    async def cancel_node(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        return await self.settle_execution_cancellation(
            graph_id,
            node_id,
            execution_id,
            request,
            cancel_confirmed=None,
        )

    async def settle_execution_cancellation(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
        *,
        cancel_confirmed: bool | None,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_CANCEL,
            principal=request.principal,
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

        if cancel_confirmed is False:
            view = TaskGraphView(
                graph_state.graph_id,
                graph_state.status,
                graph_state.nodes,
            )
        elif not _terminal(state.status):
            view = await self._persistence.tasks.cancel_node(
                graph_id,
                node_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                cancel_confirmed=cancel_confirmed is True,
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

        if self._launcher is not None and (
            state.status
            in {
                TaskStatus.CANCELLED,
                TaskStatus.RECOVERY_REQUIRED,
            }
            or cancel_confirmed is False
        ):
            admission = await self._persistence.admissions.get(
                graph_id,
                tenant_id=tenant_id,
            )
            if admission is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._launcher.cancel_node(
                admission.launch(),
                node_id,
                execution_id,
                invoke_effects=cancel_confirmed is True,
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
            view = TaskGraphView(
                graph_state.graph_id,
                graph_state.status,
                graph_state.nodes,
            )

        if state.status is TaskStatus.RECOVERY_REQUIRED:
            if claimed or operation.status is OperationStatus.EFFECT_UNKNOWN:
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
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        await self._authorize_graph(
            graph_id,
            AuthorizationAction.TASK_CANCEL,
            principal=request.principal,
        )
        request_digest = canonical_sha256(
            {
                "action": "task.cancel",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "force": request.force,
            }
        )
        if admission_guard is not None:
            admission_guard()
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
        if operation.status is OperationStatus.SUCCEEDED:
            return view

        drive_cancel = operation.status in {
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
        }
        late_terminal = claimed and _terminal(view.status)

        if drive_cancel and not late_terminal:
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

        if drive_cancel and not late_terminal and (
            not _terminal(view.status)
            or view.status is TaskStatus.CANCELLED
        ):
            try:
                await self._cleanup_graph_runtime(
                    view,
                    request.principal,
                    invoke_effects=True,
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

        if drive_cancel and not late_terminal and _terminal(view.status):
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

        if not _terminal(view.status):
            return view
        if drive_cancel:
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


def _validate_event_window(after_event_seq: int, limit: int) -> None:
    if (
        isinstance(after_event_seq, bool)
        or not isinstance(after_event_seq, int)
        or after_event_seq < 0
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _validate_event_page(
    graph_id: str,
    after_event_seq: int,
    page: Page[TaskEvent],
) -> None:
    expected = after_event_seq
    for event in page.items:
        if event.graph_id != graph_id or event.event_seq != expected + 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = event.event_seq


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
