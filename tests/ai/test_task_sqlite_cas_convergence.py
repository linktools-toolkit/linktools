#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph optimistic CAS convergence."""

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    JsonValue,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.runtime.state._repositories import TaskRepositoryImpl
from linktools.ai.runtime.state._task_recovery_repository import DurableTaskRepositoryImpl
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.task import (
    CancelGraphRequest,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeRunResult,
    TaskTerminalRecord,
)
from linktools.ai.task._service_impl import DefaultTaskService
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import create_async_engine


class _TaskTestModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:task"
    fingerprint = "a" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "task",
    }

    def materialize(self) -> TestModel:
        return TestModel()


class _TaskTestModels:
    def snapshot(self) -> "_TaskTestModels":
        return self

    def resolve(self, route_id: str) -> _TaskTestModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _TaskTestModelBinding()

    def restore(
        self,
        payload: dict[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TaskTestModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _TaskTestModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TaskTestModelBinding()


def _workspace(root: Path) -> Workspace:
    agent_path = root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    return Workspace.load(root)


async def _provision_sqlite(path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()


async def _digest_run(
    self: RuntimeTaskNodeRunner,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    dependency_results: Mapping[str, TaskDependencyResult],
) -> TaskNodeRunResult:
    del self, principal, dependency_results
    await asyncio.sleep(0)
    return TaskNodeRunResult(
        canonical_sha256({"graph_id": graph_id, "node_id": node.node_id})
    )


async def _noop_cancel(
    self: RuntimeTaskNodeRunner,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    dependency_results: Mapping[str, TaskDependencyResult],
) -> None:
    del self, node, graph_id, principal, dependency_results


@pytest.mark.asyncio
async def test_sqlite_public_runtime_task_graph_repeated_concurrency_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", _digest_run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "state.sqlite"
    await _provision_sqlite(database)
    state = RuntimeState.sqlite(database)
    workspace = _workspace(tmp_path / "workspace")

    async with Runtime.open(
        workspace,
        models=_TaskTestModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        agent = runtime.agent("default")
        for index in range(20):
            graph = TaskGraph(
                f"serial-{index}",
                (
                    agent.task("a", "run a"),
                    agent.task("b", "run b", dependencies=("a",)),
                ),
            )
            result = await runtime.run_graph_and_wait(
                graph,
                idempotency_key=f"submit:serial:{index}",
                limits=TaskGraphLimits(max_concurrency=1),
                timeout_seconds=10,
            )
            assert result.status is TaskStatus.SUCCEEDED
            assert all(
                node.status is TaskStatus.SUCCEEDED
                for node in result.node_results
            )

        for index in range(20):
            graph = TaskGraph(
                f"parallel-{index}",
                (
                    agent.task("a", "run a"),
                    agent.task("b", "run b"),
                    agent.task("c", "run c"),
                    agent.task(
                        "join",
                        "join",
                        dependencies=("a", "b", "c"),
                    ),
                ),
            )
            result = await runtime.run_graph_and_wait(
                graph,
                idempotency_key=f"submit:parallel:{index}",
                limits=TaskGraphLimits(max_concurrency=3),
                timeout_seconds=10,
            )
            assert result.status is TaskStatus.SUCCEEDED
            assert all(
                node.status is TaskStatus.SUCCEEDED
                for node in result.node_results
            )


@pytest.mark.asyncio
async def test_sqlite_public_runtime_task_failure_blocks_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run(
        self: RuntimeTaskNodeRunner,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> TaskNodeRunResult:
        del self, principal, dependency_results
        if node.node_id == "fail":
            raise AIError(ErrorCode.TASK_NODE_FAILED)
        return TaskNodeRunResult(
            canonical_sha256({"graph_id": graph_id, "node_id": node.node_id})
        )

    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "failure.sqlite"
    await _provision_sqlite(database)
    state = RuntimeState.sqlite(database)
    workspace = _workspace(tmp_path / "workspace")

    async with Runtime.open(
        workspace,
        models=_TaskTestModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        agent = runtime.agent("default")
        graph = TaskGraph(
            "failure",
            (
                agent.task("fail", "fail"),
                agent.task("dependent", "dependent", dependencies=("fail",)),
            ),
        )
        result = await runtime.run_graph_and_wait(
            graph,
            idempotency_key="submit:failure",
            limits=TaskGraphLimits(max_concurrency=1),
            timeout_seconds=10,
        )

    statuses = {node.node_id: node.status for node in result.node_results}
    errors = {node.node_id: node.error_code for node in result.node_results}
    assert result.status is TaskStatus.FAILED
    assert statuses == {
        "fail": TaskStatus.FAILED,
        "dependent": TaskStatus.BLOCKED,
    }
    assert errors["fail"] == ErrorCode.TASK_NODE_FAILED.value
    assert errors["fail"] != ErrorCode.STORAGE_CONFLICT.value


@pytest.mark.asyncio
async def test_sqlite_public_runtime_task_wait_timeout_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def run(
        self: RuntimeTaskNodeRunner,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> TaskNodeRunResult:
        del self, node, graph_id, principal, dependency_results
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked task unexpectedly completed")

    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "cancel.sqlite"
    await _provision_sqlite(database)
    state = RuntimeState.sqlite(database)
    workspace = _workspace(tmp_path / "workspace")

    async with Runtime.open(
        workspace,
        models=_TaskTestModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        agent = runtime.agent("default")
        graph = TaskGraph("timeout", (agent.task("blocked", "blocked"),))
        with pytest.raises(AIError) as raised:
            await runtime.run_graph_and_wait(
                graph,
                idempotency_key="submit:timeout",
                limits=TaskGraphLimits(max_concurrency=1),
                timeout_seconds=0.05,
            )
        assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
        await asyncio.wait_for(started.wait(), timeout=1)
        view = await runtime.task.cancel_graph(
            graph.graph_id,
            CancelGraphRequest(runtime.default_principal, "cancel:timeout"),
        )
        assert view.status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_sqlite_terminal_nodes_leave_recovery_index_after_reconcile(
    tmp_path: Path,
) -> None:
    database = tmp_path / "recovery.sqlite"
    await _provision_sqlite(database)
    request = TaskGraphRequest(
        TaskGraph("recovery-projection", (TaskNode("root"),)),
        Principal("tester", "tenant"),
        "submit:recovery-projection",
        TaskGraphLimits(max_concurrency=1),
    )
    admission = TaskGraphAdmission.from_request(request)
    state = RuntimeState.sqlite(database)
    await state.initialize(namespace="task-cas-recovery", tenant_id="tenant")
    try:
        await state.task.admissions.admit(admission, request.graph)
        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "root",
            tenant_id="tenant",
            owner="recovery-runner",
            lease_seconds=60,
        )
        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest=canonical_sha256({"result": "done"}),
        )
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert page.items == (admission.bind(request.graph),)
    finally:
        await state.close()

    reopened = RuntimeState.sqlite(database)
    await reopened.initialize(namespace="task-cas-recovery", tenant_id="tenant")
    try:
        view = await reopened.task.tasks.reconcile_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert view.status is TaskStatus.SUCCEEDED
        page = await reopened.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert page.items == ()
    finally:
        await reopened.close()


class _ReadOnlyTaskRepository:
    def __init__(self, view: TaskGraphView) -> None:
        self.view = view
        self.reconcile_calls = 0

    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView:
        del graph_id, tenant_id
        return self.view

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView:
        del graph_id, tenant_id
        self.reconcile_calls += 1
        raise AssertionError("observer must not reconcile Task state")

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        del graph_id, tenant_id
        return ()


@pytest.mark.asyncio
async def test_task_inspect_and_wait_are_read_only() -> None:
    tasks = _ReadOnlyTaskRepository(
        TaskGraphView("observed", TaskStatus.SUCCEEDED, ())
    )
    service = DefaultTaskService(
        SimpleNamespace(tasks=tasks),
        TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal("tester", "tenant")

    inspected = await service.inspect_graph("observed", principal=principal)
    waited = await service.wait_graph("observed", principal=principal)

    assert inspected.status is TaskStatus.SUCCEEDED
    assert waited.status is TaskStatus.SUCCEEDED
    assert tasks.reconcile_calls == 0


async def _admitted_state(
    graph: TaskGraph,
) -> tuple[RuntimeState, TaskGraphRequest]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cas", tenant_id="tenant")
    request = TaskGraphRequest(
        graph,
        Principal("tester", "tenant"),
        f"submit:{graph.graph_id}",
        TaskGraphLimits(max_concurrency=1),
    )
    await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        graph,
    )
    return state, request


@pytest.mark.asyncio
async def test_task_claim_conflict_reloads_and_reclassifies_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("claim-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    original = TaskRepositoryImpl.claim
    attempts = 0

    async def claim(
        self: TaskRepositoryImpl,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await original(
                self,
                graph_id,
                node_id,
                tenant_id=tenant_id,
                owner="winner",
                lease_seconds=lease_seconds,
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original(
            self,
            graph_id,
            node_id,
            tenant_id=tenant_id,
            owner=owner,
            lease_seconds=lease_seconds,
        )

    monkeypatch.setattr(TaskRepositoryImpl, "claim", claim)
    try:
        with pytest.raises(AIError) as raised:
            await repository.claim(
                request.graph.graph_id,
                "root",
                tenant_id="tenant",
                owner="loser",
                lease_seconds=60,
            )
        assert raised.value.code is ErrorCode.TASK_OWNER_CONFLICT
        assert attempts == 2
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].owner == "winner"
        assert nodes[0].status is TaskStatus.RUNNING
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_complete_retries_after_same_fence_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("complete-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    original_complete = TaskRepositoryImpl.complete
    original_renew = TaskRepositoryImpl.renew
    attempts = 0

    async def complete(
        self: TaskRepositoryImpl,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
    ) -> TaskTerminalRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await original_renew(
                self,
                lease,
                tenant_id=tenant_id,
                lease_seconds=60,
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original_complete(
            self,
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
            result_digest=result_digest,
        )

    monkeypatch.setattr(TaskRepositoryImpl, "complete", complete)
    try:
        terminal = await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest=canonical_sha256({"result": True}),
        )
        assert terminal.status is TaskStatus.SUCCEEDED
        assert attempts == 2
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].status is TaskStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_fail_retries_after_same_fence_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("fail-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    original_fail = TaskRepositoryImpl.fail
    original_renew = TaskRepositoryImpl.renew
    attempts = 0

    async def fail(
        self: TaskRepositoryImpl,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
    ) -> TaskTerminalRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await original_renew(
                self,
                lease,
                tenant_id=tenant_id,
                lease_seconds=60,
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original_fail(
            self,
            lease,
            tenant_id=tenant_id,
            error_code=error_code,
            error_digest=error_digest,
        )

    monkeypatch.setattr(TaskRepositoryImpl, "fail", fail)
    try:
        terminal = await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest=canonical_sha256({"failure": True}),
        )
        assert terminal.status is TaskStatus.FAILED
        assert attempts == 2
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].status is TaskStatus.FAILED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_terminal_retry_preserves_cancelled_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("cancel-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    original_complete = TaskRepositoryImpl.complete
    attempts = 0

    async def complete(
        self: TaskRepositoryImpl,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
    ) -> TaskTerminalRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await self.cancel_graph(lease.graph_id, tenant_id=tenant_id)
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original_complete(
            self,
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
            result_digest=result_digest,
        )

    monkeypatch.setattr(TaskRepositoryImpl, "complete", complete)
    try:
        with pytest.raises(AIError) as raised:
            await repository.complete(
                lease,
                tenant_id="tenant",
                execution_id=None,
                result_digest=canonical_sha256({"result": True}),
            )
        assert raised.value.code is ErrorCode.TASK_FENCE_STALE
        assert attempts == 2
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_reconcile_retries_whole_projection_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TaskGraph(
        "reconcile-race",
        (TaskNode("a"), TaskNode("b", ("a",))),
    )
    state, request = await _admitted_state(graph)
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "a",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    await repository.complete(
        lease,
        tenant_id="tenant",
        execution_id=None,
        result_digest=canonical_sha256({"result": "a"}),
    )
    original = repository._sync_recovery_projection
    attempts = 0

    async def sync(transaction, view: TaskGraphView) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await original(transaction, view)

    monkeypatch.setattr(repository, "_sync_recovery_projection", sync)
    try:
        view = await repository.reconcile_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert attempts == 2
        assert view.status is TaskStatus.PENDING
        nodes = {
            node.node_id: node
            for node in await repository.list_nodes(
                request.graph.graph_id,
                tenant_id="tenant",
            )
        }
        assert nodes["a"].status is TaskStatus.SUCCEEDED
        assert nodes["b"].status is TaskStatus.READY
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_cancel_retries_whole_projection_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(
        TaskGraph("cancel-projection-race", (TaskNode("a"), TaskNode("b")))
    )
    repository = state.task.tasks
    assert isinstance(repository, DurableTaskRepositoryImpl)
    original = repository._sync_recovery_projection
    attempts = 0

    async def sync(transaction, view: TaskGraphView) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await original(transaction, view)

    monkeypatch.setattr(repository, "_sync_recovery_projection", sync)
    try:
        view = await repository.cancel_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert attempts == 2
        assert view.status is TaskStatus.CANCELLED
        nodes = await repository.list_nodes(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert {node.status for node in nodes} == {TaskStatus.CANCELLED}
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert page.items == ()
    finally:
        await state.close()
