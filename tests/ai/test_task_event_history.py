#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph event history contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import (
    TaskEventType,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
)
from linktools.ai.task._service_impl import DefaultTaskService
from sqlalchemy.ext.asyncio import create_async_engine


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _request(graph: TaskGraph) -> TaskGraphRequest:
    return TaskGraphRequest(
        graph,
        Principal("tester", "tenant"),
        f"submit:{graph.graph_id}",
        TaskGraphLimits(),
    )


@pytest.mark.asyncio
async def test_task_admission_starts_contiguous_durable_event_history() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-admission", tenant_id="tenant")
    try:
        graph = TaskGraph(
            "event-admission",
            (
                TaskNode("root"),
                TaskNode("child", dependencies=("root",)),
            ),
        )
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(_request(graph)),
            graph,
        )

        page = await state.task.tasks.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )

        assert [event.sequence for event in page.items] == [1, 2, 3]
        assert [event.event_type for event in page.items] == [
            TaskEventType.GRAPH_ADMITTED,
            TaskEventType.NODE_CHANGED,
            TaskEventType.NODE_CHANGED,
        ]
        assert page.items[0].status is TaskStatus.PENDING
        assert page.items[1].node_id == "root"
        assert page.items[1].status is TaskStatus.READY
        assert page.items[2].node_id == "child"
        assert page.items[2].status is TaskStatus.PENDING
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_event_page_accepts_maximum_limit() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-max-limit", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-max-limit", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")

        page = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=1000,
        )

        assert [event.sequence for event in page.items] == [1, 2]
        assert page.next_cursor is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_empty_graph_create_is_terminal_from_first_event() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-empty", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-empty", ())

        view = await repository.create_graph(graph, tenant_id="tenant")
        page = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )

        assert view.status is TaskStatus.SUCCEEDED
        assert len(page.items) == 1
        assert page.items[0].event_type is TaskEventType.GRAPH_ADMITTED
        assert page.items[0].status is TaskStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_node_event_mutations_do_not_read_full_graph_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-local-mutations", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-local-mutations", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")

        async def forbidden_snapshot(*args: object, **kwargs: object):
            del args, kwargs
            raise AssertionError("node event mutation must not scan the full graph")

        monkeypatch.setattr(repository, "_snapshot_graph_in_transaction", forbidden_snapshot)

        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.bind_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution-local",
        )
        await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id="execution-local",
            result_digest="e" * 64,
        )
        page = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=2,
            limit=100,
        )

        assert [event.sequence for event in page.items] == [3, 4, 5]
        assert all(event.event_type is TaskEventType.NODE_CHANGED for event in page.items)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_event_history_records_semantic_changes_but_not_heartbeat() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-transitions", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-transitions", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        initial = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )
        assert [event.sequence for event in initial.items] == [1, 2]

        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        claimed = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=initial.items[-1].sequence,
            limit=100,
        )
        assert [event.sequence for event in claimed.items] == [3]
        assert claimed.items[0].event_type is TaskEventType.NODE_CHANGED
        assert claimed.items[0].previous_status is TaskStatus.READY
        assert claimed.items[0].status is TaskStatus.RUNNING
        assert claimed.items[0].owner == "worker"
        assert claimed.items[0].fence == 1

        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")
        running = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=claimed.items[-1].sequence,
            limit=100,
        )
        assert [event.sequence for event in running.items] == [4]
        assert running.items[0].event_type is TaskEventType.GRAPH_CHANGED
        assert running.items[0].previous_status is TaskStatus.PENDING
        assert running.items[0].status is TaskStatus.RUNNING

        renewed = await repository.renew(
            lease,
            tenant_id="tenant",
            lease_seconds=30,
        )
        after_renew = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=running.items[-1].sequence,
            limit=100,
        )
        assert after_renew.items == ()

        await repository.bind_execution(
            renewed,
            tenant_id="tenant",
            execution_id="execution-1",
        )
        await repository.complete(
            renewed,
            tenant_id="tenant",
            execution_id="execution-1",
            result_digest="a" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")
        terminal = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=running.items[-1].sequence,
            limit=100,
        )
        assert [event.sequence for event in terminal.items] == [5, 6, 7]
        assert terminal.items[0].execution_id == "execution-1"
        assert terminal.items[0].status is TaskStatus.RUNNING
        assert terminal.items[1].previous_status is TaskStatus.RUNNING
        assert terminal.items[1].status is TaskStatus.SUCCEEDED
        assert terminal.items[1].owner is None
        assert terminal.items[1].result_digest == "a" * 64
        assert terminal.items[2].event_type is TaskEventType.GRAPH_CHANGED
        assert terminal.items[2].previous_status is TaskStatus.RUNNING
        assert terminal.items[2].status is TaskStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_event_stream_replays_from_durable_sequence() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-terminal-stream", tenant_id="tenant")
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-terminal-stream", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="b" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")
        durable = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )
        assert durable.items[-1].status is TaskStatus.SUCCEEDED
        assert durable.items[-1].event_type is TaskEventType.GRAPH_CHANGED

        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
        )
        stream = service.stream_graph_events(
            graph.graph_id,
            principal=Principal("tester", "tenant"),
            after_sequence=2,
        )
        replayed = [event async for event in stream]

        assert replayed == list(durable.items[2:])
        assert [event.sequence for event in replayed] == list(
            range(3, durable.items[-1].sequence + 1)
        )
    finally:
        if stream is not None:
            await stream.aclose()
        await state.close()


@pytest.mark.asyncio
async def test_terminal_event_early_close_releases_graph_handoff() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-event-terminal-release", tenant_id="tenant")
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("event-terminal-release", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="d" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")

        released: list[tuple[str, str]] = []

        async def release_terminal(graph_id: str, *, tenant_id: str) -> None:
            released.append((tenant_id, graph_id))

        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            release_terminal=release_terminal,
        )
        stream = service.stream_graph_events(
            graph.graph_id,
            principal=Principal("tester", "tenant"),
        )
        terminal = None
        async for event in stream:
            if event.node_id is None and event.status is TaskStatus.SUCCEEDED:
                terminal = event
                break
        assert terminal is not None
        assert released == []

        await stream.aclose()
        stream = None

        assert released == [("tenant", graph.graph_id)]
    finally:
        if stream is not None:
            await stream.aclose()
        await state.close()


@pytest.mark.asyncio
async def test_sqlite_task_event_history_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "task-events.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    try:
        await provision_runtime_database(engine)
    finally:
        await engine.dispose()

    graph = TaskGraph("sqlite-task-events", (TaskNode("node"),))
    state = RuntimeState.sqlite(database)
    await state.initialize(namespace="task-event-sqlite", tenant_id="tenant")
    try:
        repository = state.task.tasks
        await repository.create_graph(graph, tenant_id="tenant")
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code="TASK_NODE_FAILED",
            error_digest="c" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")
        before = await repository.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )
    finally:
        await state.close()

    reopened = RuntimeState.sqlite(database)
    await reopened.initialize(namespace="task-event-sqlite", tenant_id="tenant")
    try:
        after = await reopened.task.tasks.list_events(
            graph.graph_id,
            tenant_id="tenant",
            after_sequence=0,
            limit=100,
        )
        assert after.items == before.items
        assert [event.sequence for event in after.items] == list(
            range(1, len(after.items) + 1)
        )
        assert after.items[-1].status is TaskStatus.FAILED
        assert after.items[-1].event_type is TaskEventType.GRAPH_CHANGED
    finally:
        await reopened.close()
