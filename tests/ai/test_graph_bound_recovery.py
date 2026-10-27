#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task.engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Graph recovery authorizes control separately from bound execution identity."""
from ._runtime_test_helpers import model_features
from linktools.ai.model import create_model


import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingContract
from linktools.ai.runtime import RuntimeFeatures
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

from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionRecord
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
    def __init__(self, storage: Persistence) -> None:
        self.storage = storage
        self.operations = []
        self.claims = {}
        self.fail_execution = None
        self.owned = set()

    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
        return tenant_id == "default" and execution_id in self.owned

    async def recover_execution(
        self, execution_id: str, *, tenant_id: str, expected_revision: int,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord:
        assert recovery_operation is not None
        if self.owns_execution(execution_id, tenant_id=tenant_id):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
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
    capture_phase: str | None = None,
    capture_status: ExecutionStatus = ExecutionStatus.STARTED,
    error_origin: str = "node",
) -> AsyncIterator[tuple[Runtime, PersistenceStorage, DefaultTaskGraphService, _RecoveryBackend, _Launcher, _GraphAuthorization]]:
    authorization = _GraphAuthorization()
    storage = PersistenceStorage.in_memory()
    capabilities = RuntimeFeatures("bound-recovery")
    capabilities.agent("default", model="default", allow_tools=())
    async with Persistence.open('bound-recovery', storage=storage) as persistence_115, Runtime.attach(persistence=persistence_115, authorization=authorization, features=model_features(create_model(api='openai', model='gpt-test'), capabilities)) as runtime, TaskEngine.open(persistence=persistence_115, authorization=authorization, context=runtime.context) as runtime_engine:
        task = Task.from_agent("test.bound-agent", runtime.agents.get("default"))
        graph = TaskGraph("bound-graph", tuple(
            TaskNode(node_id, task=task, input={"prompt": "recover"})
            for node_id in node_ids
        ))
        submission = await runtime_engine.with_definitions(task).prepare_submission(
            graph, idempotency_key="prepare-bound-graph", principal=_ORIGIN,
        )
        await runtime._state.task.admissions.admit_prepared(submission)
        runner = runtime_engine._task_node_runtime
        assert runner is not None
        await runner.activate_graph(submission.graph, (runtime_engine._bind_agent_task(task),), ())
        binding = AgentBindingContract.from_payload(task.contract["config"]["binding_contract"])
        leases = []
        for node_id in node_ids:
            status = (capture_status if capture_phase is not None else
                      ExecutionStatus.STARTED if node_id == "live" and live_status is not None
                      else ExecutionStatus.RECOVERY_REQUIRED)
            record = replace(
                _execution(datetime.now(timezone.utc)),
                execution_id=f"execution-{node_id}", root_execution_id=f"execution-{node_id}",
                binding=binding, principal_id=_ORIGIN.principal_id,
                principal_kind=_ORIGIN.kind, requires_task_invocation_capture=True,
                status=status,
                error_code=None if status is ExecutionStatus.STARTED else ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
            )
            await runtime._state.execution.executions.create_with_history_head(record)
            leases.append(await runtime._state.task.tasks.claim(
                graph.graph_id, node_id, tenant_id="default",
                owner="stopped-worker", lease_seconds=30,
            ))
        for lease in leases:
            if lease.node_id == "live" and live_status is not None:
                await runtime._state.task.tasks.bind_execution(
                    lease, tenant_id="default", execution_id="execution-live",
                )
                if live_status is TaskStatus.WAITING:
                    await runtime._state.task.tasks.handoff_execution(
                        lease, tenant_id="default", execution_id="execution-live",
                    )
                continue
            await runtime._state.task.tasks.mark_recovery_required(
                lease, tenant_id="default", execution_id=f"execution-{lease.node_id}",
                error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
                error_digest=canonical_sha256({"node": lease.node_id}),
                error_origin=error_origin,
                safe_error_details={} if capture_phase is None else {"phase": capture_phase},
            )
        backend = _RecoveryBackend(runtime._state)
        if capture_phase is not None:
            backend.owned.update(f"execution-{node_id}" for node_id in node_ids)
        execution_service = runtime._execution_service
        original_backend = execution_service.runtime_backend
        execution_service.runtime_backend = lambda: backend
        launcher = _Launcher()
        service = DefaultTaskGraphService(
            runtime._state.task, authorization, launcher, preflight=runner,
            bound_execution_recovery=runner,
        )
        authorization.setup = False
        authorization.calls.clear()
        try:
            yield runtime, runtime._state, service, backend, launcher, authorization
        finally:
            execution_service.runtime_backend = original_backend


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,status,origin,reattach", (
    ("task_invocation_capture_cancel", ExecutionStatus.STARTED, "execution", True),
    ("unrelated_storage_failure", ExecutionStatus.STARTED, "execution", False),
    ("task_invocation_capture_cancel", ExecutionStatus.RECOVERY_REQUIRED, "execution", False),
    ("task_invocation_capture_cancel", ExecutionStatus.STARTED, "node", False),
))
async def test_live_owner_reattachment_requires_retained_capture_handoff(
    phase: str, status: ExecutionStatus, origin: str, reattach: bool,
) -> None:
    async with _graph_recovery(capture_phase=phase, capture_status=status, error_origin=origin) as (
        _runtime, storage, service, backend, launcher, _authorization,
    ):
        before = await storage.execution.executions.get("execution-agent", tenant_id="default")
        request = RecoverGraphRequest(_ACTOR, "recover-capture-handoff")
        if reattach:
            await service.recover("bound-graph", request)
            assert len(launcher.launches) == 1
        else:
            with pytest.raises(AIError) as raised:
                await service.recover("bound-graph", request)
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert launcher.launches == []
        assert await storage.execution.executions.get("execution-agent", tenant_id="default") == before
        assert backend.claims == {}
        assert backend.operations == []


@pytest.mark.asyncio
async def test_capture_handoff_reattachment_rechecks_recovery_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _graph_recovery(capture_phase="task_invocation_capture_cancel", error_origin="execution") as (
        _runtime, storage, service, backend, launcher, _authorization,
    ):
        get_execution = storage.execution.executions.get
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused_get(execution_id, *, tenant_id):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return await get_execution(execution_id, tenant_id=tenant_id)

        monkeypatch.setattr(storage.execution.executions, "get", paused_get)
        pending = asyncio.create_task(service.recover(
            "bound-graph", RecoverGraphRequest(_ACTOR, "recover-cleared-handoff"),
        ))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await storage.task.tasks.recover_graph("bound-graph", tenant_id="default")
            release.set()
            with pytest.raises(AIError) as raised:
                await asyncio.wait_for(pending, 5)
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert backend.claims == {}
            assert launcher.launches == []
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_binding", (False, True))
async def test_bound_recovery_retains_validated_contracts_across_terminal_cleanup(
    monkeypatch: pytest.MonkeyPatch, changed_binding: bool,
) -> None:
    async with _graph_recovery(("first", "second")) as (runtime, storage, service, backend, launcher, _authorization):
        runner = service._preflight
        get_execution = storage.execution.executions.get
        entered, release = asyncio.Event(), asyncio.Event()

        async def gated_get(execution_id, *, tenant_id):
            if not entered.is_set():
                entered.set()
                await asyncio.wait_for(release.wait(), 10)
            return await get_execution(execution_id, tenant_id=tenant_id)

        monkeypatch.setattr(storage.execution.executions, "get", gated_get)
        pending = asyncio.create_task(service.recover(
            "bound-graph", RecoverGraphRequest(_ACTOR, "finish-during-bound-recovery"),
        ))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            state = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
            for node in state.node_states:
                execution = await get_execution(node.execution_id, tenant_id="default")
                binding = execution.binding
                if changed_binding and node.node_id == "second":
                    binding = replace(binding, agent_spec=replace(
                        binding.agent_spec, revision=binding.agent_spec.revision + 1,
                    ))
                await storage.execution.executions.compare_and_swap(
                    node.execution_id, tenant_id="default", expected_revision=execution.revision,
                    next_record=replace(execution, revision=execution.revision + 1,
                                        status=ExecutionStatus.SUCCEEDED, error_code=None, binding=binding),
                )
                await storage.task.tasks.complete(
                    None, tenant_id="default", graph_id="bound-graph", node_id=node.node_id,
                    expected_fence=node.fence, execution_id=node.execution_id,
                    result_digest=canonical_sha256({"node": node.node_id}),
                )
            completed = await storage.task.tasks.graph_state("bound-graph", tenant_id="default")
            assert completed.status is TaskStatus.SUCCEEDED
            await runner.release_graph_dependencies(completed, tenant_id="default")
            release.set()
            if changed_binding:
                with pytest.raises(AIError) as raised:
                    await asyncio.wait_for(pending, 10)
                assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                result = await asyncio.wait_for(pending, 10)
                assert result.status is TaskStatus.SUCCEEDED
                assert all(node.status is TaskStatus.SUCCEEDED for node in result.node_results)
            assert backend.operations == []
            assert launcher.started == []
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
