#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import (
    JsonValue,
    OperationKind,
    OperationStatus,
    Principal,
    ResourceKind,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeStorage
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskGraphService,
    RecoverGraphRequest,
    TaskEffectResolution,
    TaskEffectResolutionRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskInputSupplyRequest,
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

    async def settle_cancel(
        self,
        launch: TaskGraphLaunch,
        *,
        invoke_effects: bool,
    ) -> TaskGraphView:
        if invoke_effects:
            return await self.cancel(launch)
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
async def test_successful_recover_key_does_not_clear_a_later_recovery_boundary() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recover-replay", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "recover-replay")
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        recovery_request = RecoverGraphRequest(request.principal, "recover:stable-key")

        first = await service.recover("recover-replay", recovery_request)
        assert first.status is TaskStatus.PENDING
        assert launcher.started == ["recover-replay"]

        lease = await state.task.tasks.claim(
            "recover-replay",
            "node",
            tenant_id="tenant",
            owner="later-worker",
            lease_seconds=30,
        )
        await state.task.tasks.mark_recovery_required(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            error_digest=canonical_sha256(
                {"graph_id": "recover-replay", "later": "unknown"}
            ),
            execution_id="execution",
        )

        replayed = await service.recover("recover-replay", recovery_request)
        operation = await state.task.operations.get(
            idempotency_key_digest("recover:stable-key"),
            tenant_id="tenant",
        )

        assert replayed.status is TaskStatus.RECOVERY_REQUIRED
        assert replayed.node_results[0].status is TaskStatus.RECOVERY_REQUIRED
        assert launcher.started == ["recover-replay"]
        assert operation is not None
        assert operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_concurrent_recover_replay_keeps_stable_receipt_as_graph_advances() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-recover-concurrent", tenant_id="tenant")
    release_first_starts = asyncio.Event()
    both_first_starts = asyncio.Event()
    first: asyncio.Task | None = None
    second: asyncio.Task | None = None
    try:
        request = _request("recover-concurrent")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )

        class _ProgressLauncher(_Launcher):
            async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
                await super().start(launch)
                if len(self.started) <= 2:
                    if len(self.started) == 2:
                        both_first_starts.set()
                    await release_first_starts.wait()
                return TaskGraphHandle(launch.graph_id)

        launcher = _ProgressLauncher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        recovery_request = RecoverGraphRequest(request.principal, "recover:concurrent")
        first = asyncio.create_task(
            service.recover("recover-concurrent", recovery_request)
        )
        second = asyncio.create_task(
            service.recover("recover-concurrent", recovery_request)
        )
        await asyncio.wait_for(both_first_starts.wait(), 1)

        lease = await state.task.tasks.claim(
            "recover-concurrent",
            "node",
            tenant_id="tenant",
            owner="concurrent-worker",
            lease_seconds=30,
        )
        third = await service.recover("recover-concurrent", recovery_request)
        assert third.status is TaskStatus.RUNNING

        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id="execution-concurrent",
            result_digest="b" * 64,
        )
        after_completion = await service.recover(
            "recover-concurrent",
            recovery_request,
        )
        assert after_completion.status is TaskStatus.SUCCEEDED

        other = _request("recover-concurrent-other")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(other),
            other.graph,
        )
        with pytest.raises(AIError) as conflicting_request:
            await service.recover(
                other.graph.graph_id,
                RecoverGraphRequest(other.principal, "recover:concurrent"),
            )
        assert conflicting_request.value.code is ErrorCode.IDEMPOTENCY_CONFLICT

        release_first_starts.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result.status is TaskStatus.SUCCEEDED
        assert second_result.status is TaskStatus.SUCCEEDED
        assert launcher.started == ["recover-concurrent"] * 3

        operation = await state.task.operations.get(
            idempotency_key_digest("recover:concurrent"),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.SUCCEEDED
        assert operation.result_ref == "recover-concurrent"
        assert operation.result_digest == canonical_sha256(
            {"graph_id": "recover-concurrent"}
        )
    finally:
        release_first_starts.set()
        blocked = tuple(task for task in (first, second) if task is not None)
        if blocked:
            await asyncio.gather(*blocked, return_exceptions=True)
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
        assert set(launcher.cancelled) == {"cancel"}
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_effect_unknown_cancel_is_discovered_after_filesystem_reopen(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "cancel-effect-unknown"
    state = RuntimeStorage.filesystem(storage_root)
    await state.initialize(namespace="task-service-cancel-reopen", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "cancel-reopen")
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            _Launcher(),
        )
        cancelled = await service.cancel(
            "cancel-reopen",
            CancelGraphRequest(request.principal, "cancel:reopen"),
        )
        operation_id = idempotency_key_digest("cancel:reopen")
        operation = await state.task.operations.get(
            operation_id,
            tenant_id="tenant",
        )

        assert cancelled.status is TaskStatus.RECOVERY_REQUIRED
        assert operation is not None
        assert operation.status is OperationStatus.EFFECT_UNKNOWN
    finally:
        await state.close()

    reopened = RuntimeStorage.filesystem(storage_root)
    await reopened.initialize(namespace="task-service-cancel-reopen", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            reopened.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        pending = await reopened.task.operations.list_pending(
            ResourceKind.TASK_GRAPH,
            "cancel-reopen",
            tenant_id="tenant",
            limit=10,
            states=frozenset({OperationStatus.EFFECT_UNKNOWN}),
        )
        assert [item.operation_id for item in pending] == [operation_id]

        recovered = await service.recover(
            "cancel-reopen",
            RecoverGraphRequest(request.principal, "recover:cancel-reopen"),
        )
        cancel_operation = await reopened.task.operations.get(
            operation_id,
            tenant_id="tenant",
        )

        assert recovered.status is TaskStatus.RECOVERY_REQUIRED
        assert cancel_operation is not None
        assert cancel_operation.status is OperationStatus.EFFECT_UNKNOWN
        assert launcher.started == []
        assert launcher.cancelled == ["cancel-reopen"]
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_recover_settles_cancel_receipt_after_graph_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-terminal-cancel", tenant_id="tenant")
    try:
        request = _request("terminal-cancel-receipt")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        original_record_success = service._record_success
        interrupted = False

        async def interrupt_cancel_receipt(
            operation,
            tenant_id,
            view,
            *,
            expected_status=OperationStatus.RUNNING,
        ):
            nonlocal interrupted
            if operation.operation_kind is OperationKind.TASK_CANCEL and not interrupted:
                interrupted = True
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await original_record_success(
                operation,
                tenant_id,
                view,
                expected_status=expected_status,
            )

        monkeypatch.setattr(service, "_record_success", interrupt_cancel_receipt)
        with pytest.raises(AIError) as raised:
            await service.cancel(
                request.graph.graph_id,
                CancelGraphRequest(request.principal, "cancel:terminal-receipt"),
            )
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        operation_id = idempotency_key_digest("cancel:terminal-receipt")
        interrupted_operation = await state.task.operations.get(
            operation_id,
            tenant_id="tenant",
        )
        terminal = await state.task.tasks.get_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert interrupted_operation is not None
        assert interrupted_operation.status is OperationStatus.RUNNING
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED

        monkeypatch.setattr(service, "_record_success", original_record_success)
        recovered = await service.recover(
            request.graph.graph_id,
            RecoverGraphRequest(request.principal, "recover:terminal-receipt"),
        )
        settled_operation = await state.task.operations.get(
            operation_id,
            tenant_id="tenant",
        )
        assert recovered.status is TaskStatus.CANCELLED
        assert settled_operation is not None
        assert settled_operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recover_settles_cancel_receipt_when_business_success_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-cancel-success-race", tenant_id="tenant")
    release_cancel = asyncio.Event()
    cancel_entered = asyncio.Event()
    try:
        request = _request("cancel-success-race")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        original_cancel_graph = state.task.tasks.cancel_graph
        original_record_success = service._record_success
        interrupted = False

        async def pause_cancel_projection(
            graph_id: str,
            *,
            tenant_id: str,
        ) -> TaskGraphView:
            cancel_entered.set()
            await release_cancel.wait()
            return await original_cancel_graph(graph_id, tenant_id=tenant_id)

        async def interrupt_cancel_receipt(
            operation,
            tenant_id,
            view,
            *,
            expected_status=OperationStatus.RUNNING,
        ):
            nonlocal interrupted
            if operation.operation_kind is OperationKind.TASK_CANCEL and not interrupted:
                interrupted = True
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await original_record_success(
                operation,
                tenant_id,
                view,
                expected_status=expected_status,
            )

        monkeypatch.setattr(state.task.tasks, "cancel_graph", pause_cancel_projection)
        monkeypatch.setattr(service, "_record_success", interrupt_cancel_receipt)
        cancellation = asyncio.create_task(
            service.cancel(
                request.graph.graph_id,
                CancelGraphRequest(request.principal, "cancel:success-race"),
            )
        )
        await asyncio.wait_for(cancel_entered.wait(), 1)

        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id="execution-success-race",
            result_digest="a" * 64,
        )
        release_cancel.set()
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(cancellation, 2)
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        operation_id = idempotency_key_digest("cancel:success-race")
        pending = await state.task.operations.get(operation_id, tenant_id="tenant")
        terminal = await state.task.tasks.get_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert pending is not None
        assert pending.status is OperationStatus.RUNNING
        assert terminal is not None
        assert terminal.status is TaskStatus.SUCCEEDED

        monkeypatch.setattr(service, "_record_success", original_record_success)
        recovered = await service.recover(
            request.graph.graph_id,
            RecoverGraphRequest(request.principal, "recover:success-race"),
        )
        settled = await state.task.operations.get(operation_id, tenant_id="tenant")
        assert recovered.status is TaskStatus.SUCCEEDED
        assert settled is not None
        assert settled.status is OperationStatus.SUCCEEDED
    finally:
        release_cancel.set()
        await state.close()


@pytest.mark.asyncio
async def test_running_cancel_operation_replays_cancel_effect_after_owner_loss() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-cancel-owner-loss", tenant_id="tenant")
    try:
        request = _request("cancel-owner-loss")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        cancel_request = CancelGraphRequest(
            request.principal,
            "cancel:owner-loss",
        )
        request_digest = canonical_sha256(
            {
                "action": "task.cancel",
                "principal": principal_identity_payload(request.principal),
                "graph_id": request.graph.graph_id,
                "force": cancel_request.force,
            }
        )
        claimed, operation = await service._claim_cancel_operation(
            idempotency_key_digest(cancel_request.idempotency_key),
            "tenant",
            request.graph.graph_id,
            request_digest,
        )
        assert claimed
        assert operation.status is OperationStatus.RUNNING

        result = await service.cancel(
            request.graph.graph_id,
            cancel_request,
        )
        settled = await state.task.operations.get(
            operation.operation_id,
            tenant_id="tenant",
        )

        assert result.status is TaskStatus.CANCELLED
        assert launcher.cancelled == [request.graph.graph_id]
        assert settled is not None
        assert settled.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_new_cancel_after_business_terminal_preserves_terminal_status() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-late-cancel", tenant_id="tenant")
    try:
        request = _request("late-cancel")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="a" * 64,
        )
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )

        result = await service.cancel(
            request.graph.graph_id,
            CancelGraphRequest(request.principal, "cancel:late-terminal"),
        )

        assert result.status is TaskStatus.FAILED
        assert launcher.cancelled == []
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_running_cancel_replay_reapplies_durable_graph_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-cancel-replay", tenant_id="tenant")
    try:
        request = _request("cancel-replay")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        original_cancel_graph = state.task.tasks.cancel_graph
        calls = 0

        async def interrupt_before_projection(
            graph_id: str,
            *,
            tenant_id: str,
        ) -> TaskGraphView:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await original_cancel_graph(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(
            state.task.tasks,
            "cancel_graph",
            interrupt_before_projection,
        )
        cancel_request = CancelGraphRequest(request.principal, "cancel:replay")
        with pytest.raises(AIError) as interrupted:
            await service.cancel("cancel-replay", cancel_request)
        assert interrupted.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        result = await service.cancel("cancel-replay", cancel_request)
        operation = await state.task.operations.get(
            idempotency_key_digest("cancel:replay"),
            tenant_id="tenant",
        )

        assert calls == 3
        assert result.status is TaskStatus.CANCELLED
        assert operation is not None
        assert operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_registration_wins_claim_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-cancel-claim-race", tenant_id="tenant")
    release_registration = asyncio.Event()
    registered = asyncio.Event()
    try:
        request = _request("cancel-claim-race")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        original_register = state.task.tasks.register_cancel_request

        async def register_then_pause(operation: object):
            result = await original_register(operation)  # type: ignore[arg-type]
            registered.set()
            await release_registration.wait()
            return result

        monkeypatch.setattr(
            state.task.tasks,
            "register_cancel_request",
            register_then_pause,
        )
        cancel_request = CancelGraphRequest(request.principal, "cancel:claim-race")
        cancellation = asyncio.create_task(
            service.cancel("cancel-claim-race", cancel_request)
        )
        await asyncio.wait_for(registered.wait(), 1)

        with pytest.raises(AIError) as rejected_claim:
            await state.task.tasks.claim(
                "cancel-claim-race",
                "node",
                tenant_id="tenant",
                owner="racing-worker",
                lease_seconds=30,
            )
        assert rejected_claim.value.code is ErrorCode.TASK_NOT_READY

        release_registration.set()
        result = await asyncio.wait_for(cancellation, 2)
        assert result.status is TaskStatus.CANCELLED
    finally:
        release_registration.set()
        await state.close()


@pytest.mark.asyncio
async def test_cancel_intent_blocks_claim_after_not_applied_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-cancel-resolution-race", tenant_id="tenant")
    release_cancel = asyncio.Event()
    cancel_entered = asyncio.Event()
    try:
        request = await _recovery_graph(state, "cancel-resolution-race")

        class _ResolutionLauncher(_Launcher):
            async def resolve_effect(
                self,
                launch: TaskGraphLaunch,
                node_id: str,
                execution_id: str,
                expected_fence: int,
                resolution: TaskEffectResolution,
            ) -> TaskGraphView:
                assert resolution.kind == "not_applied"
                await state.task.tasks.requeue_recovery(
                    launch.graph_id,
                    node_id,
                    tenant_id="tenant",
                    expected_fence=expected_fence,
                    execution_id=execution_id,
                )
                view = await state.task.tasks.get_graph(
                    launch.graph_id,
                    tenant_id="tenant",
                )
                assert view is not None
                return view

        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            _ResolutionLauncher(),
        )
        original_cancel_graph = state.task.tasks.cancel_graph

        async def pause_cancel_graph(
            graph_id: str,
            *,
            tenant_id: str,
        ) -> TaskGraphView:
            cancel_entered.set()
            await release_cancel.wait()
            return await original_cancel_graph(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(
            state.task.tasks,
            "cancel_graph",
            pause_cancel_graph,
        )
        cancel_request = CancelGraphRequest(request.principal, "cancel:resolution-race")
        cancellation = asyncio.create_task(
            service.cancel("cancel-resolution-race", cancel_request)
        )
        await asyncio.wait_for(cancel_entered.wait(), 1)

        result = await service.resolve_effect(
            "cancel-resolution-race",
            "node",
            TaskEffectResolutionRequest(
                request.principal,
                1,
                TaskEffectResolution("not_applied"),
                "resolve:cancel-resolution-race",
            ),
        )
        assert result.status is TaskStatus.PENDING
        with pytest.raises(AIError) as rejected_claim:
            await state.task.tasks.claim(
                "cancel-resolution-race",
                "node",
                tenant_id="tenant",
                owner="racing-worker",
                lease_seconds=30,
            )
        assert rejected_claim.value.code is ErrorCode.TASK_NOT_READY

        release_cancel.set()
        cancellation_result = await asyncio.wait_for(cancellation, 2)
        assert cancellation_result.status is TaskStatus.PENDING
        operation = await state.task.operations.get(
            idempotency_key_digest("cancel:resolution-race"),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.RUNNING
    finally:
        release_cancel.set()
        await state.close()


@pytest.mark.asyncio
async def test_successful_effect_resolution_replay_completes_missing_arm() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-effect-arm", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "effect-arm")

        class _ResolutionLauncher(_Launcher):
            def __init__(self) -> None:
                super().__init__()
                self.resolve_calls = 0

            async def resolve_effect(
                self,
                launch: TaskGraphLaunch,
                node_id: str,
                execution_id: str,
                expected_fence: int,
                resolution: TaskEffectResolution,
            ) -> TaskGraphView:
                assert resolution.kind == "not_applied"
                self.resolve_calls += 1
                await state.task.tasks.requeue_recovery(
                    launch.graph_id,
                    node_id,
                    tenant_id="tenant",
                    expected_fence=expected_fence,
                    execution_id=execution_id,
                )
                view = await state.task.tasks.get_graph(
                    launch.graph_id,
                    tenant_id="tenant",
                )
                assert view is not None
                return view

        launcher = _ResolutionLauncher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        original_arm = service._arm_graph
        arm_calls = 0

        async def fail_first_arm(launch: TaskGraphLaunch) -> None:
            nonlocal arm_calls
            arm_calls += 1
            if arm_calls == 1:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await original_arm(launch)

        service._arm_graph = fail_first_arm  # type: ignore[method-assign]
        resolution_request = TaskEffectResolutionRequest(
            request.principal,
            1,
            TaskEffectResolution("not_applied"),
            "resolve:effect-arm",
        )
        with pytest.raises(AIError) as interrupted:
            await service.resolve_effect("effect-arm", "node", resolution_request)
        assert interrupted.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        result = await service.resolve_effect("effect-arm", "node", resolution_request)

        assert result.status is TaskStatus.PENDING
        assert launcher.resolve_calls == 1
        assert launcher.started == ["effect-arm"]
        assert arm_calls == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_successful_input_resume_replay_completes_missing_arm() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-input-arm", tenant_id="tenant")
    try:
        graph = TaskGraph(
            "input-arm",
            (TaskNode("input"), TaskNode("dependent", ("input",))),
        )
        request = TaskGraphRequest(
            graph,
            Principal("tester", "tenant"),
            "submit:input-arm",
        )
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            graph,
        )
        lease = await state.task.tasks.claim(
            graph.graph_id,
            "input",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="input-arm-wait-id",
            occupies_concurrency=False,
        )

        class _InputLauncher(_Launcher):
            def __init__(self) -> None:
                super().__init__()
                self.supply_calls = 0

            async def supply_input(
                self,
                launch: TaskGraphLaunch,
                node_id: str,
                execution_id: str,
                value: JsonValue,
            ) -> TaskGraphView:
                assert node_id == "input"
                assert execution_id == "input-arm-wait-id"
                self.supply_calls += 1
                await state.task.tasks.complete(
                    None,
                    tenant_id="tenant",
                    graph_id=launch.graph_id,
                    node_id=node_id,
                    execution_id=execution_id,
                    result_digest=canonical_sha256(value),
                )
                view = await state.task.tasks.get_graph(
                    launch.graph_id,
                    tenant_id="tenant",
                )
                assert view is not None
                return view

        launcher = _InputLauncher()
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )
        original_arm = service._arm_graph
        arm_calls = 0

        async def fail_first_arm(launch: TaskGraphLaunch) -> None:
            nonlocal arm_calls
            arm_calls += 1
            if arm_calls == 1:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await original_arm(launch)

        service._arm_graph = fail_first_arm  # type: ignore[method-assign]
        resume_request = TaskInputSupplyRequest(
            request.principal,
            "input-arm-wait-id",
            {"value": "accepted"},
            "resume:input-arm",
        )
        with pytest.raises(AIError) as interrupted:
            await service.resume("input-arm", "input", resume_request)
        assert interrupted.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

        first_projection = await state.task.tasks.graph_state(
            "input-arm",
            tenant_id="tenant",
        )
        first_operation = await state.task.operations.get(
            idempotency_key_digest("resume:input-arm"),
            tenant_id="tenant",
        )
        assert first_projection is not None
        assert first_operation is not None
        first_input = next(
            item for item in first_projection.node_states if item.node_id == "input"
        )
        assert first_input.status is TaskStatus.SUCCEEDED
        assert first_input.execution_id == "input-arm-wait-id"
        assert first_operation.operation_kind is OperationKind.TASK_NODE
        assert first_operation.status is OperationStatus.SUCCEEDED
        assert first_operation.result_digest == canonical_sha256({"value": "accepted"})

        replay = await service.resume("input-arm", "input", resume_request)

        assert replay.status is TaskStatus.PENDING
        assert launcher.supply_calls == 1
        assert launcher.started == ["input-arm"]
        assert arm_calls == 2
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

        unresolved = await service.cancel_node(
            "node-cancel",
            "node",
            "execution",
            cancel_request,
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
        node_state = graph_state.node_states[0]
        assert node_state.status is TaskStatus.RECOVERY_REQUIRED

        await state.task.tasks.cancel_node(
            "node-cancel",
            "node",
            tenant_id="tenant",
            execution_id="execution",
            cancel_confirmed=True,
            expected_fence=node_state.fence,
        )
        confirmed = await service.cancel_node(
            "node-cancel",
            "node",
            "execution",
            cancel_request,
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


@pytest.mark.asyncio
async def test_node_cancel_scope_and_execution_identity_are_preserved() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-service-node-cancel-scope", tenant_id="tenant")
    try:
        request = TaskGraphRequest(
            TaskGraph(
                "node-cancel-scope",
                (TaskNode("target"), TaskNode("sibling")),
            ),
            Principal("tester", "tenant"),
            "node-cancel-scope-submit",
        )
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "target",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.mark_recovery_required(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            error_digest=canonical_sha256({"code": ErrorCode.TOOL_EFFECT_UNKNOWN.value}),
            execution_id="target-execution",
        )
        service = DefaultTaskGraphService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
        )
        cancel_request = CancelGraphRequest(request.principal, "cancel:node-scope")
        unresolved = await service.cancel_node(
            request.graph.graph_id,
            "target",
            "target-execution",
            cancel_request,
        )
        assert unresolved.status is TaskStatus.RECOVERY_REQUIRED
        unresolved_state = await state.task.tasks.graph_state(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert unresolved_state is not None
        target_state = next(
            node for node in unresolved_state.node_states if node.node_id == "target"
        )
        await state.task.tasks.cancel_node(
            request.graph.graph_id,
            "target",
            tenant_id="tenant",
            execution_id="target-execution",
            cancel_confirmed=True,
            expected_fence=target_state.fence,
        )
        result = await service.cancel_node(
            request.graph.graph_id,
            "target",
            "target-execution",
            cancel_request,
        )
        snapshot = await state.task.tasks.graph_state(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert snapshot is not None
        statuses = {node.node_id: node.status for node in snapshot.node_states}
        assert statuses == {
            "target": TaskStatus.CANCELLED,
            "sibling": TaskStatus.READY,
        }
        assert result.status is TaskStatus.PENDING

        with pytest.raises(AIError) as conflicting_replay:
            await service.cancel_node(
                request.graph.graph_id,
                "target",
                "different-execution",
                cancel_request,
            )
        assert conflicting_replay.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    finally:
        await state.close()
