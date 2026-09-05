#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph review closure invariants."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import (
    OperationStatus,
    Page,
    Principal,
    ResourceRef,
    TaskStatus,
    idempotency_key_digest,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._planner import _AgentTaskNodeHandler
from linktools.ai.runtime.state._store import StateTransaction
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskService,
    LocalTaskGraphLauncher,
    TaskEvent,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskNode,
    TaskNodeContext,
    TaskNodeRunResult,
    TaskNodeView,
)
from linktools.ai.workspace import trusted_workspace_principal


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _CountingAuthorization:
    def __init__(self) -> None:
        self.calls = 0

    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.calls += 1


class _NoopRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, *args: object, **kwargs: object) -> TaskNodeRunResult:
        del args, kwargs
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _FailingCancelLauncher:
    async def start(self, launch: object) -> object:
        del launch
        return SimpleNamespace()

    async def cancel(self, launch: object) -> object:
        del launch
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)


@pytest.mark.asyncio
async def test_natural_failure_is_visible_before_explicit_cancel_override() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-natural-aggregate", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("natural-aggregate", (TaskNode("failed"), TaskNode("active")))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="a" * 64,
        )

        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert before is not None
        assert before.status is TaskStatus.FAILED

        view = await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert after is not None
        states = {item.node_id: item for item in after.node_states}
        assert view.status is TaskStatus.CANCELLED
        assert after.status is TaskStatus.CANCELLED
        assert states["failed"].status is TaskStatus.FAILED
        assert states["active"].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_service_cancel_overrides_failed_graph_with_active_node() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-service-cancel-override", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "service-cancel-override", (TaskNode("failed"), TaskNode("active"))
        )
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="c" * 64,
        )
        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert before is not None
        assert before.status is TaskStatus.FAILED
        assert {item.node_id: item.status for item in before.node_states}[
            "active"
        ] is TaskStatus.READY

        service = DefaultTaskService(state.task, _AllowAuthorization())
        view = await service.cancel_graph(
            graph.graph_id,
            CancelGraphRequest(
                trusted_workspace_principal("tenant"),
                "service-cancel-override-0001",
            ),
        )

        assert view.status is TaskStatus.CANCELLED
        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert after is not None
        states = {item.node_id: item.status for item in after.node_states}
        assert after.status is TaskStatus.CANCELLED
        assert states["failed"] is TaskStatus.FAILED
        assert states["active"] is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_explicit_cancel_preserves_blocked_terminal_node() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-blocked", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "cancel-blocked",
            (
                TaskNode("failed"),
                TaskNode("blocked", ("failed",)),
                TaskNode("active"),
            ),
        )
        await admit_graph(state, graph)
        await repository.claim(
            graph.graph_id,
            "active",
            tenant_id="tenant",
            owner="active-worker",
            lease_seconds=30,
        )
        failed = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="failed-worker",
            lease_seconds=30,
        )
        await repository.fail(
            failed,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="b" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")

        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert before is not None
        before_states = {item.node_id: item for item in before.node_states}
        assert before_states["blocked"].status is TaskStatus.BLOCKED
        assert before_states["active"].status is TaskStatus.RUNNING

        await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert after is not None
        states = {item.node_id: item for item in after.node_states}
        assert after.status is TaskStatus.CANCELLED
        assert states["failed"].status is TaskStatus.FAILED
        assert states["blocked"].status is TaskStatus.BLOCKED
        assert states["active"].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_launcher_counts_remote_live_lease_in_max_concurrency() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-global-capacity", tenant_id="tenant")
    launcher = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("global-capacity", (TaskNode("remote"), TaskNode("local")))
        await admit_graph(state, graph)
        await repository.claim(
            graph.graph_id,
            "remote",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        runner = _NoopRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(max_concurrency=1),
            )
        )

        await asyncio.sleep(0.05)
        assert not runner.started.is_set()
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        states = {item.node_id: item for item in snapshot.node_states}
        assert states["local"].status is TaskStatus.READY
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


def test_task_node_context_input_is_deeply_detached() -> None:
    context = TaskNodeContext(
        None,
        Principal("tester", "tenant"),
        "graph",
        "node",
        {"nested": {"value": 1}},
        {},
        "context-idempotency",
    )
    nested = context.input["nested"]
    assert isinstance(nested, dict)
    nested["value"] = 2
    assert context.input["nested"] == {"value": 1}


@pytest.mark.asyncio
async def test_cancel_cleanup_failure_marks_operation_effect_unknown() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-ledger", tenant_id="tenant")
    try:
        graph = TaskGraph("cancel-ledger", (TaskNode("node"),))
        await admit_graph(state, graph)
        service = DefaultTaskService(
            state.task,
            _AllowAuthorization(),
            _FailingCancelLauncher(),
        )
        key = "cancel-ledger-request-0001"
        with pytest.raises(AIError) as error:
            await service.cancel_graph(
                graph.graph_id,
                CancelGraphRequest(
                    trusted_workspace_principal("tenant"),
                    key,
                ),
            )
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        operation = await state.task.operations.get(
            idempotency_key_digest(key),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.EFFECT_UNKNOWN
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_detached_bind_unknown_surfaces_background_failure() -> None:
    handler = _AgentTaskNodeHandler(object(), object(), object())
    started = asyncio.Event()
    release = asyncio.Event()

    class Control:
        async def bind_execution(self, execution_id: str) -> None:
            del execution_id
            started.set()
            await release.wait()
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    task = asyncio.create_task(
        handler._bind_execution(
            Control(),
            "execution",
            key=("tenant", "graph", "node"),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    pending = handler.pending_background_tasks
    assert pending
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    failure = handler.background_failure
    assert failure is not None
    assert failure.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_bind_after_launch_ignores_expected_ownership_loss() -> None:
    handler = _AgentTaskNodeHandler(object(), object(), object())

    async def launched() -> object:
        return SimpleNamespace(execution_id="execution")

    class Control:
        async def bind_execution(self, execution_id: str) -> None:
            del execution_id
            raise AIError(ErrorCode.TASK_FENCE_STALE)

    launch_task = asyncio.create_task(launched())
    await handler._bind_after_launch(
        launch_task,
        ("tenant", "graph", "node"),
        Control(),
    )
    assert handler.background_failure is None


@pytest.mark.asyncio
async def test_task_get_graph_rejects_missing_canonical_node() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-get-graph-missing-node", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("get-graph-missing-node", (TaskNode("node"),))
        await admit_graph(state, graph)
        node_key = repository._node_key(graph.graph_id, "node")

        async def delete_node(transaction: StateTransaction) -> None:
            record = await transaction.get_record(node_key)
            assert record is not None
            assert await transaction.delete_record(
                node_key,
                expected_storage_version=record.storage_version,
            )

        await repository.state_store.mutate(delete_node)

        with pytest.raises(AIError) as repository_error:
            await repository.get_graph(graph.graph_id, tenant_id="tenant")
        assert repository_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        service = DefaultTaskService(state.task, _AllowAuthorization())
        with pytest.raises(AIError) as service_error:
            await service.inspect_graph(
                graph.graph_id,
                principal=trusted_workspace_principal("tenant"),
            )
        assert service_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_get_graph_rejects_tampered_node_dependencies() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-get-graph-dependencies", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "get-graph-dependencies",
            (TaskNode("root"), TaskNode("child", ("root",))),
        )
        await admit_graph(state, graph)
        node_key = repository._node_key(graph.graph_id, "child")

        async def corrupt_dependencies(transaction: StateTransaction) -> None:
            record = await transaction.get_record(node_key)
            assert record is not None
            node = await repository._decode(record, TaskNodeView)
            candidate = repository._task_node_record(
                record,
                replace(node, dependencies=()),
            )
            assert await transaction.replace_record(
                candidate,
                expected_storage_version=record.storage_version,
            )

        await repository.state_store.mutate(corrupt_dependencies)

        with pytest.raises(AIError) as error:
            await repository.get_graph(graph.graph_id, tenant_id="tenant")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_readback_does_not_accept_corrupt_graph_as_terminal() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-corrupt-readback", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("cancel-corrupt-readback", (TaskNode("node"),))
        await admit_graph(state, graph)
        node_key = repository._node_key(graph.graph_id, "node")

        async def delete_node(transaction: StateTransaction) -> None:
            record = await transaction.get_record(node_key)
            assert record is not None
            assert await transaction.delete_record(
                node_key,
                expected_storage_version=record.storage_version,
            )

        await repository.state_store.mutate(delete_node)
        service = DefaultTaskService(state.task, _AllowAuthorization())
        key = "cancel-corrupt-readback-0001"

        with pytest.raises(AIError) as error:
            await service.cancel_graph(
                graph.graph_id,
                CancelGraphRequest(
                    trusted_workspace_principal("tenant"),
                    key,
                ),
            )
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        operation = await state.task.operations.get(
            idempotency_key_digest(key),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.EFFECT_UNKNOWN
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_wait_graph_reuses_existing_read_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-wait-auth-read", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("wait-auth-read", (TaskNode("node"),))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")

        original_get_header = repository.get_header
        original_list_events = repository.list_events
        header_calls = 0
        completed = False

        async def counted_get_header(
            graph_id: str,
            *,
            tenant_id: str,
        ) -> ResourceRef | None:
            nonlocal header_calls
            header_calls += 1
            return await original_get_header(graph_id, tenant_id=tenant_id)

        async def complete_before_read(
            graph_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[TaskEvent]:
            nonlocal completed
            if not completed:
                completed = True
                await repository.complete(
                    lease,
                    tenant_id="tenant",
                    execution_id=None,
                    result_digest="d" * 64,
                )
                await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")
            return await original_list_events(
                graph_id,
                tenant_id=tenant_id,
                after_sequence=after_sequence,
                limit=limit,
            )

        monkeypatch.setattr(repository, "get_header", counted_get_header)
        monkeypatch.setattr(repository, "list_events", complete_before_read)
        authorization = _CountingAuthorization()
        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            authorization,
        )

        result = await service.wait_graph(
            graph.graph_id,
            principal=trusted_workspace_principal("tenant"),
            timeout_seconds=1,
        )

        assert result.status is TaskStatus.SUCCEEDED
        assert header_calls == 1
        assert authorization.calls == 1
    finally:
        await state.close()
