#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import (
    OperationKind,
    OperationStatus,
    Principal,
    ResourceKind,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeStorage
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskGraphService,
    RecoverGraphRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)


class _Launcher:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.launches: list[TaskGraphLaunch] = []
        self.cancelled: list[str] = []

    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        self.started.append(launch.graph_id)
        self.launches.append(launch)
        return TaskGraphHandle(launch.graph_id)

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        self.cancelled.append(launch.graph_id)
        return TaskGraphView(
            launch.graph_id,
            TaskStatus.RECOVERY_REQUIRED,
            (),
        )


class _RecoveryPreflight:
    def __init__(self) -> None:
        self.prepared_principals: list[Principal] = []
        self.validated = 0

    async def load_admission(self, admission: TaskGraphAdmission) -> None:
        assert admission.graph_id

    def validate_recovery(self, state: object) -> None:
        del state
        self.validated += 1

    async def prepare_graph(self, state: object, *, principal: Principal) -> None:
        del state
        self.prepared_principals.append(principal)


def _request(graph_id: str) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(graph_id, (TaskNode("node"),)),
        Principal("tester", "tenant"),
        f"submit:{graph_id}",
    )


async def _recovery_graph(state: RuntimeStorage, graph_id: str) -> TaskGraphRequest:
    request = _request(graph_id)
    await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        request.graph,
    )
    lease = await state.task.tasks.claim(
        graph_id,
        "node",
        tenant_id="tenant",
        owner="worker",
        lease_seconds=30,
    )
    await state.task.tasks.mark_recovery_required(
        lease,
        tenant_id="tenant",
        error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        error_digest=canonical_sha256(
            {"graph_id": graph_id, "node_id": "node", "code": ErrorCode.TOOL_EFFECT_UNKNOWN.value}
        ),
        execution_id="execution",
    )
    return request


@pytest.mark.asyncio
async def test_wait_and_stream_end_at_durable_recovery_boundary() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "boundary")
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            _Launcher(),
        )

        result = await service.wait(
            "boundary",
            principal=request.principal,
            timeout_seconds=1,
        )
        events = [
            event
            async for event in service.stream_events(
                "boundary",
                principal=request.principal,
            )
        ]

        assert result.status is TaskStatus.RECOVERY_REQUIRED
        assert result.node_results[0].status is TaskStatus.RECOVERY_REQUIRED
        assert events[-1].node_id is None
        assert events[-1].status is TaskStatus.RECOVERY_REQUIRED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_explicit_recovery_rearms_original_graph() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "resume")
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )

        result = await service.recover(
            "resume",
            RecoverGraphRequest(request.principal, "recover:resume"),
        )

        assert result.status is TaskStatus.PENDING
        assert launcher.started == ["resume"]
        assert result.node_results[0].status is TaskStatus.READY
        assert result.node_results[0].execution_id == "execution"
        operation = await state.task.operations.get(
            idempotency_key_digest("recover:resume"),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.operation_kind is OperationKind.TASK_RECOVER
        assert operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recovery_actor_does_not_replace_admitted_execution_principal() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "principal-recovery")
        actor = Principal("operator", "tenant")
        launcher = _Launcher()
        preflight = _RecoveryPreflight()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
            preflight=preflight,  # type: ignore[arg-type]
        )

        result = await service.recover(
            "principal-recovery",
            RecoverGraphRequest(actor, "recover:principal-recovery"),
        )

        assert result.status is TaskStatus.PENDING
        assert preflight.prepared_principals == [request.principal]
        assert launcher.launches[0].principal == request.principal
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recovery_validation_fails_before_graph_state_transition() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "validate-before-recover")
        launcher = _Launcher()

        class RejectingPreflight(_RecoveryPreflight):
            def validate_recovery(self, state: object) -> None:
                super().validate_recovery(state)
                raise AIError(ErrorCode.BINDING_NOT_REGISTERED)

        preflight = RejectingPreflight()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
            preflight=preflight,  # type: ignore[arg-type]
        )

        with pytest.raises(AIError) as raised:
            await service.recover(
                "validate-before-recover",
                RecoverGraphRequest(request.principal, "recover:validate-before-recover"),
            )

        assert raised.value.code is ErrorCode.BINDING_NOT_REGISTERED
        persisted = await state.task.tasks.get_graph(
            "validate-before-recover",
            tenant_id="tenant",
        )
        assert persisted is not None
        assert persisted.status is TaskStatus.RECOVERY_REQUIRED
        assert preflight.validated == 1
        assert preflight.prepared_principals == []
        assert launcher.started == []
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_intent_stays_unknown_until_execution_fact_is_available() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "cancel")
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )

        deferred = await service.cancel(
            "cancel",
            CancelGraphRequest(request.principal, "cancel:intent"),
        )
        cancel_operation = await state.task.operations.get(
            idempotency_key_digest("cancel:intent"),
            tenant_id="tenant",
        )
        pending = await state.task.operations.list_pending(
            ResourceKind.TASK_GRAPH,
            "cancel",
            tenant_id="tenant",
            limit=10,
            states=frozenset(
                {
                    OperationStatus.PENDING,
                    OperationStatus.RUNNING,
                    OperationStatus.EFFECT_UNKNOWN,
                }
            ),
        )

        assert deferred.status is TaskStatus.RECOVERY_REQUIRED
        assert cancel_operation is not None
        assert cancel_operation.operation_kind is OperationKind.TASK_CANCEL
        assert cancel_operation.status is OperationStatus.EFFECT_UNKNOWN
        assert any(item.operation_id == cancel_operation.operation_id for item in pending)

        result = await service.recover(
            "cancel",
            RecoverGraphRequest(request.principal, "recover:cancel"),
        )
        settled_cancel = await state.task.operations.get(
            cancel_operation.operation_id,
            tenant_id="tenant",
        )

        assert result.status is TaskStatus.RECOVERY_REQUIRED
        assert settled_cancel is not None
        assert settled_cancel.status is OperationStatus.EFFECT_UNKNOWN
        assert launcher.started == []
        assert launcher.cancelled == ["cancel"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_node_cancel_requires_execution_confirmation_before_terminal_projection() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "node-cancel")
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        cancel_request = CancelGraphRequest(request.principal, "cancel:node")

        unresolved = await service._settle_execution_cancellation(
            "node-cancel",
            "node",
            "execution",
            cancel_request,
            cancel_confirmed=None,
        )
        unresolved_operation = await state.task.operations.get(
            idempotency_key_digest("cancel:node"),
            tenant_id="tenant",
        )
        graph_state = await state.task.tasks.graph_state(
            "node-cancel",
            tenant_id="tenant",
        )
        assert unresolved.status is TaskStatus.RECOVERY_REQUIRED
        assert unresolved_operation is not None
        assert unresolved_operation.status is OperationStatus.EFFECT_UNKNOWN
        assert graph_state is not None
        assert graph_state.node_states[0].status is TaskStatus.RECOVERY_REQUIRED

        confirmed = await service._settle_execution_cancellation(
            "node-cancel",
            "node",
            "execution",
            cancel_request,
            cancel_confirmed=True,
        )
        confirmed_operation = await state.task.operations.get(
            idempotency_key_digest("cancel:node"),
            tenant_id="tenant",
        )
        graph_state = await state.task.tasks.graph_state(
            "node-cancel",
            tenant_id="tenant",
        )
        assert confirmed.status is TaskStatus.CANCELLED
        assert confirmed_operation is not None
        assert confirmed_operation.status is OperationStatus.SUCCEEDED
        assert graph_state is not None
        assert graph_state.node_states[0].status is TaskStatus.CANCELLED
    finally:
        await state.close()
