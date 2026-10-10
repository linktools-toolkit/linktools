#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph recovery authorizes control separately from bound execution identity."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingContract
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    AuthorizationAction,
    ExecutionStatus,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceRef,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.task import (
    DefaultTaskGraphService,
    RecoverGraphRequest,
    TaskGraph,
    TaskNode,
)

from .test_execution_recovery_commands import _execution
from .test_task_recovery_service import _Launcher


_ORIGIN = Principal("executor", "default")
_ACTOR = Principal("operator", "default")


class _GraphAuthorization:
    def __init__(self) -> None:
        self.setup = True
        self.calls = []
        self.default = TenantAuthorizationPolicy("default")

    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        self.calls.append((principal, action, resource))
        if self.setup:
            return await self.default.authorize(principal, action, resource)
        if principal == _ACTOR and action is AuthorizationAction.TASK_RUN:
            return
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _RecoveryBackend:
    def __init__(self, storage: RuntimeStorage) -> None:
        self.storage = storage
        self.operations = []
        self.claims = {}
        self.fail_execution = None

    async def recover_execution(
        self, execution_id: str, *, tenant_id: str, expected_revision: int,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord:
        assert recovery_operation is not None
        operation = recovery_operation
        self.operations.append(operation)
        current = await self.storage.execution.executions.get(execution_id, tenant_id=tenant_id)
        assert current is not None
        if operation.operation_id in self.claims:
            return current
        if execution_id == self.fail_execution:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        assert current.revision == expected_revision
        self.claims[operation.operation_id] = operation
        return await self.storage.execution.executions.compare_and_swap(
            execution_id, tenant_id=tenant_id, expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1, status=ExecutionStatus.STARTED,
                                error_code=None, safe_error_details={}),
        )


@asynccontextmanager
async def _graph_recovery(
    node_ids: tuple[str, ...] = ("agent",),
    *,
    live_status: TaskStatus | None = None,
) -> AsyncIterator[tuple[Runtime, RuntimeStorage, DefaultTaskGraphService, _RecoveryBackend, _Launcher, _GraphAuthorization]]:
    authorization = _GraphAuthorization()
    storage = RuntimeStorage.in_memory()
    capabilities = CapabilityGroup("bound-recovery")
    capabilities.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "bound-recovery", models=ModelRegistry.openai(model="gpt-test"),
        storage=storage, capabilities=(capabilities,), authorization=authorization,
    ) as runtime:
        task = runtime.tasks.from_agent("test.bound-agent", runtime.agents.get("default"))
        graph = TaskGraph("bound-graph", tuple(
            TaskNode(node_id, task=task, input={"prompt": "recover"})
            for node_id in node_ids
        ))
        submission = await runtime.tasks.bind(task).prepare_submission(
            graph, idempotency_key="prepare-bound-graph", principal=_ORIGIN,
        )
        await storage.task.admissions.admit_prepared(submission)
        runner = runtime._task_node_runtime
        assert runner is not None
        await runner.activate_graph(submission.graph, (task,), ())
        binding = AgentBindingContract.from_payload(task.contract["config"]["binding_contract"])
        leases = []
        for node_id in node_ids:
            record = replace(
                _execution(datetime.now(timezone.utc)),
                execution_id=f"execution-{node_id}", root_execution_id=f"execution-{node_id}",
                binding=binding, principal_id=_ORIGIN.principal_id,
                principal_kind=_ORIGIN.kind, requires_task_invocation_capture=True,
                status=(ExecutionStatus.STARTED if node_id == "live" and live_status is not None
                        else ExecutionStatus.RECOVERY_REQUIRED),
                error_code=(None if node_id == "live" and live_status is not None
                            else ErrorCode.STORAGE_RECOVERY_REQUIRED.value),
            )
            await storage.execution.executions.create_with_history_head(record)
            leases.append(await storage.task.tasks.claim(
                graph.graph_id, node_id, tenant_id="default",
                owner="stopped-worker", lease_seconds=30,
            ))
        for lease in leases:
            if lease.node_id == "live" and live_status is not None:
                await storage.task.tasks.bind_execution(
                    lease, tenant_id="default", execution_id="execution-live",
                )
                if live_status is TaskStatus.WAITING:
                    await storage.task.tasks.handoff_execution(
                        lease, tenant_id="default", execution_id="execution-live",
                    )
                continue
            await storage.task.tasks.mark_recovery_required(
                lease, tenant_id="default", execution_id=f"execution-{lease.node_id}",
                error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
                error_digest=canonical_sha256({"node": lease.node_id}),
            )
        backend = _RecoveryBackend(storage)
        runner._recovery_backend = backend
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            storage.task, authorization, launcher, preflight=runner,
            bound_execution_recovery=runner,
        )
        authorization.setup = False
        authorization.calls.clear()
        yield runtime, storage, service, backend, launcher, authorization


@pytest.mark.asyncio
async def test_graph_run_actor_recovers_bound_agent_without_execution_control_permission() -> None:
    async with _graph_recovery() as (runtime, storage, service, backend, launcher, authorization):
        with pytest.raises(AIError) as denied:
            await runtime.executions.recover(
                "execution-agent", principal=_ACTOR, idempotency_key="standalone-recovery",
            )
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert backend.operations == []
        authorization.calls.clear()

        request = RecoverGraphRequest(_ACTOR, "recover-bound-graph")
        result = await service.recover("bound-graph", request)
        assert result.status in {TaskStatus.RUNNING, TaskStatus.PENDING}
        assert launcher.launches[0].principal == _ORIGIN
        assert authorization.calls
        assert all(principal == _ACTOR and action is AuthorizationAction.TASK_RUN
                   for principal, action, _resource in authorization.calls)
        operation = backend.operations[0]
        assert operation.operation_kind is OperationKind.EXECUTION_RECOVER
        assert operation.status is OperationStatus.RUNNING
        assert operation.result_ref and operation.result_digest is None
        assert not operation.compactable
        current = await storage.execution.executions.get("execution-agent", tenant_id="default")
        assert current.principal_id == _ORIGIN.principal_id
        assert current.principal_kind == _ORIGIN.kind
        assert current.status is ExecutionStatus.STARTED
        await service.recover("bound-graph", request)
        assert len(backend.claims) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["principal", "binding", "unbound"])
async def test_graph_recovery_rejects_untrusted_execution_binding_before_takeover(mismatch: str) -> None:
    async with _graph_recovery() as (_runtime, storage, service, backend, launcher, _authorization):
        current = await storage.execution.executions.get("execution-agent", tenant_id="default")
        assert current is not None
        changes = {
            "principal": {"principal_kind": "service"},
            "binding": {"binding": replace(current.binding, agent_spec=replace(
                current.binding.agent_spec, revision=current.binding.agent_spec.revision + 1,
            ))},
            "unbound": {"requires_task_invocation_capture": False},
        }[mismatch]
        await storage.execution.executions.compare_and_swap(
            current.execution_id, tenant_id="default", expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1, **changes),
        )
        with pytest.raises(AIError) as raised:
            await service.recover("bound-graph", RecoverGraphRequest(_ACTOR, "reject-binding"))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert backend.operations == []
        assert launcher.started == []
        state = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
        assert state.status is TaskStatus.RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_partial_graph_recovery_replays_child_receipt_without_new_producer() -> None:
    async with _graph_recovery(("first", "second")) as (_runtime, storage, service, backend, launcher, _authorization):
        backend.fail_execution = "execution-second"
        request = RecoverGraphRequest(_ACTOR, "recover-partial-graph")
        with pytest.raises(AIError) as raised:
            await service.recover("bound-graph", request)
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert len(backend.claims) == 1
        assert launcher.started == []
        first_claim = next(iter(backend.claims.values()))
        backend.fail_execution = None
        await service.recover("bound-graph", request)
        assert len(backend.claims) == 2
        assert backend.claims[first_claim.operation_id] == first_claim
        assert len({item.operation_id for item in backend.operations if item.execution_id == "execution-first"}) == 1


@pytest.mark.asyncio
async def test_old_child_receipt_cannot_clear_newer_execution_recovery_required() -> None:
    async with _graph_recovery(("first", "second")) as (_runtime, storage, service, backend, launcher, _authorization):
        backend.fail_execution = "execution-second"
        request = RecoverGraphRequest(_ACTOR, "recover-newer-generation")
        with pytest.raises(AIError):
            await service.recover("bound-graph", request)
        current = await storage.execution.executions.get("execution-first", tenant_id="default")
        assert current is not None
        newer = await storage.execution.executions.compare_and_swap(
            current.execution_id, tenant_id="default", expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1, status=ExecutionStatus.RECOVERY_REQUIRED,
                                error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value),
        )
        backend.fail_execution = None
        with pytest.raises(AIError) as raised:
            await service.recover("bound-graph", request)
        assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        unchanged = await storage.execution.executions.get(current.execution_id, tenant_id="default")
        assert unchanged == newer
        assert len(backend.claims) == 1
        assert launcher.started == []
        state = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
        assert state.status is TaskStatus.RECOVERY_REQUIRED


@pytest.mark.asyncio
@pytest.mark.parametrize("live_status", [TaskStatus.RUNNING, TaskStatus.WAITING])
async def test_explicit_graph_takeover_is_limited_to_durable_bindings(live_status: TaskStatus) -> None:
    async with _graph_recovery(("live", "agent"), live_status=live_status) as (
        _runtime, storage, service, backend, _launcher, _authorization,
    ):
        before = await storage.execution.executions.get("execution-live", tenant_id="default")
        assert before is not None
        unrelated = replace(
            before, execution_id="unrelated-execution", root_execution_id="unrelated-execution",
            requires_task_invocation_capture=False,
        )
        await storage.execution.executions.create_with_history_head(unrelated)
        await service.recover("bound-graph", RecoverGraphRequest(_ACTOR, "take-over-bound-graph"))
        after = await storage.execution.executions.get("execution-live", tenant_id="default")
        assert after.revision > before.revision
        assert {operation.execution_id for operation in backend.operations} == {
            "execution-live", "execution-agent",
        }
        assert await storage.execution.executions.get(
            unrelated.execution_id, tenant_id="default",
        ) == unrelated
        graph = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
        bound_node = next(node for node in graph.node_states if node.node_id == "live")
        assert bound_node.status is live_status
        assert bound_node.execution_id == before.execution_id


@pytest.mark.asyncio
async def test_expired_running_binding_is_reconciled_before_graph_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _graph_recovery() as (_runtime, storage, service, backend, _launcher, _authorization):
        await storage.task.tasks.recover_graph("bound-graph", tenant_id="default")
        current = await storage.execution.executions.get("execution-agent", tenant_id="default")
        assert current is not None
        await storage.execution.executions.compare_and_swap(
            current.execution_id, tenant_id="default", expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1,
                                status=ExecutionStatus.STARTED, error_code=None),
        )
        lease = await storage.task.tasks.claim(
            "bound-graph", "agent", tenant_id="default", owner="stopped-remote", lease_seconds=1,
        )
        before = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
        assert before.node_states[0].status is TaskStatus.RUNNING
        assert before.node_states[0].execution_id == current.execution_id
        recover = backend.recover_execution
        observed_states = []

        async def recover_after_reconciliation(
            execution_id: str, *, tenant_id: str, expected_revision: int,
            recovery_operation: OperationLedgerInput | None = None,
        ) -> ExecutionRecord:
            graph = await storage.task.tasks.graph_state("bound-graph", tenant_id=tenant_id)
            node = graph.node_states[0]
            observed_states.append(node.status)
            assert node.status is TaskStatus.READY
            assert node.owner is None and node.lease_expires_at is None
            assert node.execution_id == execution_id and node.fence == lease.fence
            return await recover(
                execution_id, tenant_id=tenant_id, expected_revision=expected_revision,
                recovery_operation=recovery_operation,
            )

        monkeypatch.setattr(backend, "recover_execution", recover_after_reconciliation)
        await asyncio.sleep(1.05)
        await service.recover("bound-graph", RecoverGraphRequest(_ACTOR, "recover-expired-binding"))
        assert observed_states and set(observed_states) == {TaskStatus.READY}
        assert len(backend.claims) == 1
