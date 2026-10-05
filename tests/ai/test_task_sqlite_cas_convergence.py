#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph optimistic CAS convergence."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
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
from linktools.ai.runtime import (
    AgentTaskInput,
    Runtime,
    RuntimeStorage,
    TaskGraphRunEvent,
)
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.runtime.service_api import ExecutionStreamEvent, ExecutionTreeEvent
from linktools.ai.runtime.state._sql import _SqlTransaction
from linktools.ai.runtime.state._task_repository import TaskRepositoryImpl
from linktools.ai.storage import FilesystemObjectStore, ObjectRef, StoredPayload
from linktools.ai.task import (
    DefaultTaskGraphService,
    Task,
    TaskExpansionContext,
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphState,
    TaskGraphView,
    TaskLease,
    TaskExpanderRef,
    TaskNode,
    TaskNodeContext,
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    TaskExpander,
    TaskNodeView,
    TaskTerminalRecord,
)
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import create_async_engine


class _TaskTestModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:task"
    model_digest = "a" * 64
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "task",
    }

    def materialize(self) -> TestModel:
        return TestModel()


class _TaskTestModels:
    def capture(self) -> "_TaskTestModels":
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
        if dict(payload) != _TaskTestModelBinding.contract:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TaskTestModelBinding()


def _agent_group() -> CapabilityGroup[object]:
    group = CapabilityGroup[object]("application")
    group.agent("default", model="default", allow_tools=())
    return group


async def _provision_sqlite(path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()


async def _digest_run(
    self: RuntimeTaskNodeRunner,
    invocation: TaskNodeInvocation,
    *,
    control: TaskNodeRunControl,
) -> TaskNodeRunResult:
    del self, control
    await asyncio.sleep(0)
    payload = StoredPayload.inline_json(
        {"graph_id": invocation.graph_id, "node_id": invocation.node.node_id}
    )
    return TaskNodeRunResult(
        payload.digest,
        execution_id=f"execution-{invocation.graph_id}-{invocation.node.node_id}",
    )


async def _noop_cancel(
    self: RuntimeTaskNodeRunner,
    invocation: TaskNodeInvocation,
) -> None:
    del self, invocation


def _agent_task(runtime: Runtime[object]) -> Task[object]:
    return runtime.tasks.from_agent(
        "sqlite.agent.default",
        runtime.agents.get("default"),
    )


def _agent_node(
    task: Task[object],
    node_id: str,
    prompt: str,
    *,
    dependencies: tuple[str, ...] = (),
) -> TaskNode:
    return TaskNode(
        node_id,
        dependencies,
        task=task,
        input=AgentTaskInput(prompt),
    )


@pytest.mark.asyncio
async def test_sqlite_state_group_serializes_mutation_callbacks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "serialized.sqlite"
    await _provision_sqlite(database)
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    await state.initialize(namespace="sqlite-serialize", tenant_id="default")
    store = state.task.tasks.state_store
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first(transaction):
        await transaction.next_sequence(b"a" * 32)
        first_entered.set()
        await release_first.wait()

    async def second(transaction):
        second_entered.set()
        await transaction.next_sequence(b"b" * 32)

    try:
        first_task = asyncio.create_task(store.mutate(first))
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second_task = asyncio.create_task(store.mutate(second))
        await asyncio.sleep(0.05)
        assert not second_entered.is_set()
        release_first.set()
        await asyncio.wait_for(
            asyncio.gather(first_task, second_task),
            timeout=2,
        )
        assert second_entered.is_set()
    finally:
        release_first.set()
        await state.close()


@pytest.mark.asyncio
async def test_sqlite_graph_state_keeps_one_snapshot_during_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "dynamic-state.sqlite"
    await _provision_sqlite(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    finally:
        connection.close()

    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    graph = TaskGraph(
        "dynamic-state",
        (TaskNode("expand", expander=TaskExpanderRef("application.expand", 1)),),
    )
    request = TaskGraphRequest(
        graph,
        Principal("tester", "default"),
        "submit:dynamic-state",
        TaskGraphLimits(max_concurrency=1, max_depth=20),
    )
    definition_read = asyncio.Event()
    continue_read = asyncio.Event()
    original_list_records = _SqlTransaction.list_records

    async def gated_list_records(self, query):
        records = await original_list_records(self, query)
        if query.kind == "task_node_definition" and not definition_read.is_set():
            definition_read.set()
            await continue_read.wait()
        return records

    try:
        async with Runtime.open(
            "sqlite-dynamic-state",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=state,
        ) as runtime:
            repository = state.task.tasks
            await state.task.admissions.admit(
                TaskGraphAdmission.from_request(request),
                graph,
            )
            run = await runtime.tasks.bind().get(
                graph.graph_id,
                principal=request.principal,
            )
            lease = await repository.claim(
                graph.graph_id,
                "expand",
                tenant_id="default",
                owner="expander",
                lease_seconds=60,
            )
            monkeypatch.setattr(_SqlTransaction, "list_records", gated_list_records)
            state_read = asyncio.create_task(run.state(include_content=True))
            await asyncio.wait_for(definition_read.wait(), timeout=1)
            await asyncio.wait_for(
                repository.complete(
                    lease,
                    tenant_id="default",
                    execution_id="execution-expand",
                    result_digest="b" * 64,
                    expanded_nodes=(
                        TaskNode(
                            "child",
                            dependencies=("expand",),
                            expander=TaskExpanderRef("application.expand", 1),
                        ),
                    ),
                ),
                timeout=5,
            )
            continue_read.set()
            before_expansion = await asyncio.wait_for(state_read, timeout=5)

            assert {node.node_id for node in before_expansion.nodes} == {"expand"}
            assert {node.node_id for node in before_expansion.node_states} == {"expand"}
            assert before_expansion.node_states[0].status is TaskStatus.RUNNING

            after_expansion = await run.state(include_content=True)
            definitions = {node.node_id: node for node in after_expansion.nodes}
            statuses = {
                node.node_id: node for node in after_expansion.node_states
            }
            assert set(definitions) == set(statuses) == {"child", "expand"}
            assert statuses["child"].dependencies == definitions["child"].dependencies
            assert statuses["child"].dependencies == ("expand",)
            assert statuses["expand"].status is TaskStatus.SUCCEEDED

            def assert_coherent(snapshot: TaskGraphState) -> int:
                nodes = {node.node_id: node for node in snapshot.nodes}
                node_states = {
                    node.node_id: node for node in snapshot.node_states
                }
                assert set(nodes) == set(node_states)
                assert all(
                    node_states[node_id].dependencies == node.dependencies
                    for node_id, node in nodes.items()
                )
                assert all(isinstance(node.status, TaskStatus) for node in node_states.values())
                return len(nodes)

            stop_reader = asyncio.Event()
            read_count = 0
            observed_sizes: set[int] = set()

            async def read_states() -> None:
                nonlocal read_count
                while not stop_reader.is_set():
                    observed_sizes.add(
                        assert_coherent(await run.state(include_content=True))
                    )
                    read_count += 1
                    await asyncio.sleep(0)

            async def expand_repeatedly() -> None:
                parent_id = "child"
                for index in range(12):
                    lease = await repository.claim(
                        graph.graph_id,
                        parent_id,
                        tenant_id="default",
                        owner="expander",
                        lease_seconds=60,
                    )
                    next_id = f"child-{index}"
                    await repository.complete(
                        lease,
                        tenant_id="default",
                        execution_id=f"execution-{next_id}",
                        result_digest="c" * 64,
                        expanded_nodes=(
                            TaskNode(
                                next_id,
                                dependencies=(parent_id,),
                                expander=TaskExpanderRef(
                                    "application.expand",
                                    1,
                                ),
                            ),
                        ),
                    )
                    parent_id = next_id
                    await asyncio.sleep(0)

            reader = asyncio.create_task(read_states())
            try:
                await expand_repeatedly()
                stop_reader.set()
                await reader
            finally:
                stop_reader.set()
                if not reader.done():
                    await reader

            assert read_count > 0
            assert max(observed_sizes) > min(observed_sizes)
            final_state = await run.state(include_content=True)
            assert assert_coherent(final_state) == 14

            state_store = state.task.tasks.state_store
            child_state_key = repository._state_key(graph.graph_id, "child")

            async def remove_child_state(transaction) -> None:
                assert await transaction.delete_record(child_state_key)

            await state_store.mutate(remove_child_state)
            with pytest.raises(AIError) as raised:
                await run.state(include_content=True)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        continue_read.set()
        await state.close()


@pytest.mark.asyncio
async def test_sqlite_dynamic_watch_resumes_after_runtime_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "dynamic-watch.sqlite"
    await _provision_sqlite(database)
    storage = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    expansion_started = asyncio.Event()
    stream_requests: list[tuple[str, dict[str, int]]] = []
    principal = Principal("watcher", "default")
    original_run = RuntimeTaskNodeRunner.run

    async def wait_for_expansion(context: TaskNodeContext[object]) -> JsonValue:
        del context
        root_started.set()
        await release_root.wait()
        return {"expanded": True}

    root_task = Task(
        "sqlite.watch.root",
        wait_for_expansion,
        effect_policy="none",
    )

    async with Runtime.open(
        "sqlite-dynamic-watch",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=storage,
        capabilities=(_agent_group(),),
    ) as runtime:
        agent_task = runtime.tasks.from_agent(
            "sqlite.watch.agent",
            runtime.agents.get("default"),
        )

        def expand_children(
            context: TaskExpansionContext,
        ) -> tuple[TaskNode, ...]:
            assert context.source_node.node_id == "root"
            expansion_started.set()
            return (
                TaskNode(
                    "child-succeeded",
                    dependencies=("root",),
                    task=agent_task,
                    input=AgentTaskInput("succeed"),
                ),
                TaskNode(
                    "child-failed",
                    dependencies=("child-succeeded",),
                    task=agent_task,
                    input=AgentTaskInput("fail"),
                ),
            )

        expander = TaskExpander("sqlite.watch.expand", expand_children)
        graph = TaskGraph(
            "dynamic-watch",
            (TaskNode("root", task=root_task, expander=expander),),
        )

        async def run_child(
            self: RuntimeTaskNodeRunner,
            invocation: TaskNodeInvocation,
            *,
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            if invocation.node.node_id == "root":
                return await original_run(self, invocation, control=control)
            execution_id = f"execution-{invocation.node.node_id}"
            await control.bind_execution(execution_id)
            if invocation.node.node_id == "child-failed":
                raise TaskNodeRunError(ErrorCode.TASK_NODE_FAILED, execution_id)
            result = StoredPayload.inline_json(
                {"graph_id": invocation.graph_id, "node_id": invocation.node.node_id}
            )
            return TaskNodeRunResult(result.digest, execution_id=execution_id)

        async def inspect_execution(
            execution_id: str,
            *,
            principal: Principal,
        ) -> SimpleNamespace:
            assert principal == principal_arg
            return SimpleNamespace(
                binding_kind=(
                    "agent"
                    if execution_id in {
                        "execution-child-succeeded",
                        "execution-child-failed",
                    }
                    else "task"
                )
            )

        def watch_tree(
            execution_id: str,
            *,
            principal: Principal,
            after_sequences: dict[str, int] | None = None,
            include_content: bool = False,
        ) -> AsyncIterator[ExecutionTreeEvent]:
            del include_content
            assert principal == principal_arg
            after = dict(after_sequences or {})
            stream_requests.append((execution_id, after))
            event_type = (
                ExecutionEventType.EXECUTION_FAILED.value
                if execution_id == "execution-child-failed"
                else ExecutionEventType.EXECUTION_SUCCEEDED.value
            )

            async def events():
                if after.get(execution_id, 0) >= 1:
                    return
                yield ExecutionTreeEvent(
                    execution_id,
                    "default",
                    ExecutionLineageKind.RUN,
                    None,
                    execution_id,
                    None,
                    0,
                    ExecutionStreamEvent(execution_id, 1, event_type, {}),
                )

            return events()

        principal_arg = principal
        monkeypatch.setattr(RuntimeTaskNodeRunner, "run", run_child)
        monkeypatch.setattr(runtime.executions, "inspect", inspect_execution)
        monkeypatch.setattr(runtime, "_watch_execution_tree", watch_tree)
        engine = runtime.tasks.bind(root_task, agent_task, expander)
        run = await engine.start(
            graph,
            idempotency_key="submit:dynamic-watch",
            principal=principal,
            limits=TaskGraphLimits(max_concurrency=1),
        )
        await asyncio.wait_for(root_started.wait(), timeout=5)
        watch = run.watch()
        first = await asyncio.wait_for(anext(watch), timeout=5)
        assert isinstance(first.event, TaskEvent)
        assert first.event.event_type is TaskEventType.GRAPH_ADMITTED
        assert not expansion_started.is_set()

        execution_events: dict[str, TaskGraphRunEvent] = {}
        release_root.set()
        try:
            while set(execution_events) != {
                "child-succeeded",
                "child-failed",
            }:
                event = await asyncio.wait_for(anext(watch), timeout=10)
                if isinstance(event.event, ExecutionTreeEvent):
                    execution_events[event.node_id] = event
        finally:
            await watch.aclose()

        assert expansion_started.is_set()
        result = await run.wait(timeout_seconds=10)
        assert result.status is TaskStatus.FAILED
        final_state = await run.state(include_content=True)
        statuses = {node.node_id: node.status for node in final_state.node_states}
        assert statuses["child-succeeded"] is TaskStatus.SUCCEEDED
        assert statuses["child-failed"] is TaskStatus.FAILED

        successful = execution_events["child-succeeded"]
        assert isinstance(successful.event, ExecutionTreeEvent)
        success_cursor = successful.cursor
        assert success_cursor is not None

    reopened_storage = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(
        "sqlite-dynamic-watch",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=reopened_storage,
        capabilities=(_agent_group(),),
    ) as runtime:
        agent_task = runtime.tasks.from_agent(
            "sqlite.watch.agent",
            runtime.agents.get("default"),
        )
        monkeypatch.setattr(runtime.executions, "inspect", inspect_execution)
        monkeypatch.setattr(runtime, "_watch_execution_tree", watch_tree)
        reopened = await runtime.tasks.bind(root_task, agent_task, expander).get(
            graph.graph_id,
            principal=principal,
        )
        replayed = [item async for item in reopened.watch(cursor=success_cursor)]

    replayed_execution_events = [
        item
        for item in replayed
        if isinstance(item.event, ExecutionTreeEvent)
    ]
    assert [
        (item.node_id, item.event.event.event_type)
        for item in replayed_execution_events
    ] == [
        ("child-failed", ExecutionEventType.EXECUTION_FAILED.value),
    ]
    assert any(
        execution_id == "execution-child-succeeded"
        and after.get(execution_id) == 1
        for execution_id, after in stream_requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("repetitions", (
    pytest.param(1, id="representative"),
    pytest.param(20, marks=pytest.mark.manual, id="stress-20"),
))
async def test_sqlite_public_runtime_task_graph_repeated_concurrency_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repetitions: int,
) -> None:
    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", _digest_run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "state.sqlite"
    await _provision_sqlite(database)
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(_agent_group(),),
    ) as runtime:
        task = _agent_task(runtime)
        engine = runtime.tasks.bind(task)
        for index in range(repetitions):
            graph = TaskGraph(
                f"serial-{index}",
                (
                    _agent_node(task, "a", "run a"),
                    _agent_node(task, "b", "run b", dependencies=("a",)),
                ),
            )
            run = await engine.start(
                graph,
                idempotency_key=f"submit:serial:{index}",
                limits=TaskGraphLimits(max_concurrency=1),
            )
            result = await run.wait(timeout_seconds=10)
            assert result.status is TaskStatus.SUCCEEDED
            assert all(
                node.status is TaskStatus.SUCCEEDED for node in result.node_results
            )

        for index in range(repetitions):
            graph = TaskGraph(
                f"parallel-{index}",
                (
                    _agent_node(task, "a", "run a"),
                    _agent_node(task, "b", "run b"),
                    _agent_node(task, "c", "run c"),
                    _agent_node(
                        task,
                        "join",
                        "join",
                        dependencies=("a", "b", "c"),
                    ),
                ),
            )
            run = await engine.start(
                graph,
                idempotency_key=f"submit:parallel:{index}",
                limits=TaskGraphLimits(max_concurrency=3),
            )
            result = await run.wait(timeout_seconds=10)
            assert result.status is TaskStatus.SUCCEEDED
            assert all(
                node.status is TaskStatus.SUCCEEDED for node in result.node_results
            )


@pytest.mark.asyncio
async def test_sqlite_public_runtime_task_failure_blocks_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run(
        self: RuntimeTaskNodeRunner,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        del self, principal, correlation, dependency_results, control
        if node.node_id == "fail":
            raise AIError(ErrorCode.TASK_NODE_FAILED)
        payload = StoredPayload.inline_json(
            {"graph_id": graph_id, "node_id": node.node_id}
        )
        return TaskNodeRunResult(payload.digest, result_payload=payload)

    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "failure.sqlite"
    await _provision_sqlite(database)
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(_agent_group(),),
    ) as runtime:
        task = _agent_task(runtime)
        engine = runtime.tasks.bind(task)
        graph = TaskGraph(
            "failure",
            (
                _agent_node(task, "fail", "fail"),
                _agent_node(task, "dependent", "dependent", dependencies=("fail",)),
            ),
        )
        run = await engine.start(
            graph,
            idempotency_key="submit:failure",
            limits=TaskGraphLimits(max_concurrency=1),
        )
        result = await run.wait(timeout_seconds=10)

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
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        del self, node, graph_id, principal, correlation, dependency_results, control
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked task unexpectedly completed")

    monkeypatch.setattr(RuntimeTaskNodeRunner, "run", run)
    monkeypatch.setattr(RuntimeTaskNodeRunner, "cancel", _noop_cancel)
    database = tmp_path / "cancel.sqlite"
    await _provision_sqlite(database)
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(_agent_group(),),
    ) as runtime:
        task = _agent_task(runtime)
        engine = runtime.tasks.bind(task)
        graph = TaskGraph("timeout", (_agent_node(task, "blocked", "blocked"),))
        run = await engine.start(
            graph,
            idempotency_key="submit:timeout",
            limits=TaskGraphLimits(max_concurrency=1),
        )
        with pytest.raises(AIError) as raised:
            await run.wait(timeout_seconds=0.05)
        assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
        await asyncio.wait_for(started.wait(), timeout=1)
        view = await run.cancel(idempotency_key="cancel:timeout")
        assert view.status is TaskStatus.RECOVERY_REQUIRED


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
    state = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
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
            execution_id="execution-root",
            result_digest=canonical_sha256({"result": "done"}),
        )
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert page.items == ()
    finally:
        await state.close()

    reopened = RuntimeStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    )
    await reopened.initialize(namespace="task-cas-recovery", tenant_id="tenant")
    try:
        view = await reopened.task.tasks.scheduler_state(
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

    async def graph_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState:
        del graph_id, tenant_id
        return TaskGraphState(
            self.view.graph_id,
            self.view.status,
            self.view.nodes,
            (),
        )

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent:
        del graph_id, tenant_id
        return TaskEvent(
            1,
            self.view.graph_id,
            1,
            TaskEventType.GRAPH_ADMITTED,
            datetime.now(timezone.utc),
            self.view.status,
        )

    async def scheduler_state(
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
    tasks = _ReadOnlyTaskRepository(TaskGraphView("observed", TaskStatus.SUCCEEDED, ()))
    service = DefaultTaskGraphService(
        SimpleNamespace(tasks=tasks),
        TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal("tester", "tenant")

    inspected = await service.inspect("observed", principal=principal)
    waited = await service.wait("observed", principal=principal)

    assert inspected.status is TaskStatus.SUCCEEDED
    assert waited.status is TaskStatus.SUCCEEDED
    assert tasks.reconcile_calls == 0


async def _admitted_state(
    graph: TaskGraph,
) -> tuple[RuntimeStorage, TaskGraphRequest]:
    state = RuntimeStorage.in_memory()
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
async def test_task_node_dependency_projection_fails_closed() -> None:
    state, request = await _admitted_state(
        TaskGraph(
            "dependency-integrity",
            (TaskNode("a"), TaskNode("b", ("a",))),
        )
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)

    async def corrupt(transaction) -> None:
        key = repository._definition_key(request.graph.graph_id, "b")
        record = await transaction.get_record(key)
        assert record is not None
        node = await repository._decode(record, TaskNode)
        tampered = TaskNode(
            node.node_id,
            ("missing",),
            input=node.input,
            budget_cost=node.budget_cost,
            expander=node.expander,
        )
        candidate = repository._stored(
            "task_node_definition",
            [request.graph.graph_id, node.node_id],
            tampered,
            parent=repository._definition_parent(request.graph.graph_id),
        )
        candidate = replace(
            candidate,
            storage_version=record.storage_version + 1,
        )
        assert await transaction.replace_record(
            candidate,
            expected_storage_version=record.storage_version,
        )

    await repository.state_store.mutate(corrupt)
    try:
        with pytest.raises(AIError) as reconcile_error:
            await repository.scheduler_state(
                request.graph.graph_id,
                tenant_id="tenant",
            )
        assert reconcile_error.value.code is ErrorCode.TASK_DAG_INVALID

        with pytest.raises(AIError) as claim_error:
            await repository.claim(
                request.graph.graph_id,
                "b",
                tenant_id="tenant",
                owner="runner",
                lease_seconds=60,
            )
        assert claim_error.value.code is ErrorCode.TASK_DAG_INVALID
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_admission_readback_rejects_corrupt_graph_identity() -> None:
    state, request = await _admitted_state(
        TaskGraph("readback-integrity", (TaskNode("root"),))
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)

    async def corrupt(transaction) -> None:
        key = repository._admission_key(request.graph.graph_id)
        record = await transaction.get_record(key)
        assert record is not None
        admission = await repository._decode(record, TaskGraphAdmission)
        tampered = replace(admission, graph_id="other-graph")
        encoded = repository._stored(
            "task_admission",
            request.graph.graph_id,
            tampered,
            scope=repository._recovery_scope(),
            state=record.state,
        )
        candidate = replace(
            record,
            data=encoded.data,
            storage_version=record.storage_version + 1,
        )
        assert await transaction.replace_record(
            candidate,
            expected_storage_version=record.storage_version,
        )

    await repository.state_store.mutate(corrupt)
    try:
        with pytest.raises(AIError) as raised:
            await state.task.admissions.get(
                request.graph.graph_id,
                tenant_id="tenant",
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_claim_conflict_reloads_and_reclassifies_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("claim-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    initial = (await repository.list_nodes(request.graph.graph_id, tenant_id="tenant"))[
        0
    ]
    winner = replace(
        initial,
        status=TaskStatus.RUNNING,
        owner="winner",
        fence=initial.fence + 1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    attempts = 0
    readbacks = 0

    async def mutate_with_event_retry(operation):
        del operation
        nonlocal attempts
        attempts += 1
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def node(graph_id: str, node_id: str, tenant_id: str):
        del graph_id, node_id, tenant_id
        nonlocal readbacks
        readbacks += 1
        return winner

    monkeypatch.setattr(
        repository,
        "_mutate_with_event_retry",
        mutate_with_event_retry,
    )
    monkeypatch.setattr(repository, "_node", node)
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
        assert attempts == 1
        assert readbacks == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_complete_conflict_reads_back_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(
        TaskGraph("complete-race", (TaskNode("root"),))
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    original_renew = TaskRepositoryImpl.renew
    attempts = 0

    async def complete(
        self: TaskRepositoryImpl,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> TaskTerminalRecord:
        del execution_id, result_digest, result_payload
        nonlocal attempts
        attempts += 1
        if attempts != 1:
            raise AssertionError("durable complete must not retry internally")
        await original_renew(
            self,
            lease,
            tenant_id=tenant_id,
            lease_seconds=60,
        )
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    monkeypatch.setattr(TaskRepositoryImpl, "complete", complete)
    try:
        with pytest.raises(AIError) as raised:
            await repository.complete(
                lease,
                tenant_id="tenant",
                execution_id="execution-root",
                result_digest=canonical_sha256({"result": True}),
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert attempts == 1
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].status is TaskStatus.RUNNING
        assert nodes[0].owner == "runner"
        assert nodes[0].fence == lease.fence
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_complete_readback_rejects_same_digest_different_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(
        TaskGraph("complete-execution-race", (TaskNode("root"),))
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    result_digest = canonical_sha256({"result": "same"})
    await repository.complete(
        lease,
        tenant_id="tenant",
        execution_id="execution-a",
        result_digest=result_digest,
    )

    async def conflict(operation):
        del operation
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    monkeypatch.setattr(repository, "_mutate_with_event_retry", conflict)
    try:
        with pytest.raises(AIError) as raised:
            await repository.complete(
                lease,
                tenant_id="tenant",
                execution_id="execution-b",
                result_digest=result_digest,
            )
        assert raised.value.code is ErrorCode.TASK_RESULT_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_fail_conflict_reads_back_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(TaskGraph("fail-race", (TaskNode("root"),)))
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    original_renew = TaskRepositoryImpl.renew
    attempts = 0

    async def fail(
        self: TaskRepositoryImpl,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> TaskTerminalRecord:
        del error_code, error_digest, execution_id
        nonlocal attempts
        attempts += 1
        if attempts != 1:
            raise AssertionError("durable fail must not retry internally")
        await original_renew(
            self,
            lease,
            tenant_id=tenant_id,
            lease_seconds=60,
        )
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    monkeypatch.setattr(TaskRepositoryImpl, "fail", fail)
    try:
        with pytest.raises(AIError) as raised:
            await repository.fail(
                lease,
                tenant_id="tenant",
                error_code=ErrorCode.TASK_NODE_FAILED.value,
                error_digest=canonical_sha256({"failure": True}),
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert attempts == 1
        nodes = await repository.list_nodes(request.graph.graph_id, tenant_id="tenant")
        assert nodes[0].status is TaskStatus.RUNNING
        assert nodes[0].owner == "runner"
        assert nodes[0].fence == lease.fence
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_terminal_conflict_preserves_cancelled_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(
        TaskGraph("cancel-race", (TaskNode("root"),))
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    lease = await repository.claim(
        request.graph.graph_id,
        "root",
        tenant_id="tenant",
        owner="runner",
        lease_seconds=60,
    )
    running = (await repository.list_nodes(request.graph.graph_id, tenant_id="tenant"))[
        0
    ]
    cancelled = replace(
        running,
        status=TaskStatus.CANCELLED,
        owner=None,
        lease_expires_at=None,
    )
    attempts = 0

    async def mutate_with_event_retry(operation):
        del operation
        nonlocal attempts
        attempts += 1
        if attempts != 1:
            raise AssertionError("durable complete must not retry internally")
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def node(graph_id: str, node_id: str, tenant_id: str):
        del graph_id, node_id, tenant_id
        return cancelled

    monkeypatch.setattr(
        repository,
        "_mutate_with_event_retry",
        mutate_with_event_retry,
    )
    monkeypatch.setattr(repository, "_node", node)
    try:
        with pytest.raises(AIError) as raised:
            await repository.complete(
                lease,
                tenant_id="tenant",
                execution_id="execution-root",
                result_digest=canonical_sha256({"result": True}),
            )
        assert raised.value.code is ErrorCode.TASK_TERMINAL_CONFLICT
        assert attempts == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_reconcile_conflict_uses_readback_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = TaskGraph(
        "reconcile-race",
        (TaskNode("a"), TaskNode("b", ("a",))),
    )
    state, request = await _admitted_state(graph)
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
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
        execution_id="execution-a",
        result_digest=canonical_sha256({"result": "a"}),
    )
    original = repository._mutate_with_event_retry
    attempts = 0

    async def mutate_with_event_retry(operation: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original(operation)  # type: ignore[arg-type]

    monkeypatch.setattr(
        repository,
        "_mutate_with_event_retry",
        mutate_with_event_retry,
    )
    try:
        view = await repository.scheduler_state(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert attempts == 1
        assert view.status is TaskStatus.PENDING
        nodes = {
            node.node_id: node
            for node in await repository.list_nodes(
                request.graph.graph_id,
                tenant_id="tenant",
            )
        }
        assert nodes["a"].status is TaskStatus.SUCCEEDED
        assert nodes["b"].status is TaskStatus.PENDING

        view = await repository.scheduler_state(
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
        assert nodes["b"].status is TaskStatus.READY
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_cancel_conflict_requires_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, request = await _admitted_state(
        TaskGraph("cancel-projection-race", (TaskNode("a"), TaskNode("b")))
    )
    repository = state.task.tasks
    assert isinstance(repository, TaskRepositoryImpl)
    original = repository._mutate_with_event_retry
    attempts = 0

    async def mutate_with_event_retry(operation: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await original(operation)  # type: ignore[arg-type]

    monkeypatch.setattr(
        repository,
        "_mutate_with_event_retry",
        mutate_with_event_retry,
    )
    try:
        with pytest.raises(AIError) as raised:
            await repository.cancel_graph(
                request.graph.graph_id,
                tenant_id="tenant",
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert attempts == 1
        nodes = await repository.list_nodes(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        assert {node.status for node in nodes} == {TaskStatus.READY}

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
