#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph observers reconcile safe metadata from the real Runtime owners."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionDeltaType, JsonValue, TaskStatus
from linktools.ai.runtime import (
    AgentTaskInput, ExecutionTreeEvent, Runtime, RuntimeStorage,
    TaskGraphProjection, TaskGraphRunEvent, TaskModelProjection,
)
from linktools.ai.runtime._watch_cursor import decode_graph_watch_cursor
from linktools.ai.storage import FilesystemObjectStore
from linktools.ai.task import (
    Task, TaskEvent, TaskExpander, TaskExpansionContext, TaskGraph,
    TaskInputSupplyRequest, TaskNode, TaskNodeContext,
)

from .test_evaluation_consumers import FixtureModels
from .test_task_sqlite_cas_convergence import _provision_sqlite


@asynccontextmanager
async def _runtime(tmp_path: Path, *, backend: str = "filesystem", models=None) -> AsyncIterator[Runtime]:
    if backend == "sqlite":
        database = tmp_path / "state.sqlite"
        await _provision_sqlite(database)
        storage = RuntimeStorage.sqlite(database, object_store=FilesystemObjectStore(tmp_path / "objects"))
    else:
        storage = RuntimeStorage.filesystem(tmp_path)
    group = CapabilityGroup("metadata")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open(
        "metadata-projection", models=FixtureModels() if models is None else models,
        storage=storage, capabilities=(group,),
    ) as runtime:
        yield runtime


async def _collect(stream: AsyncIterator[TaskGraphRunEvent]) -> list[TaskGraphRunEvent]:
    try:
        return [event async for event in stream]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend, include_content", [("filesystem", False), ("filesystem", True), ("sqlite", False)])
async def test_terminal_graph_reconnect_repeats_safe_metadata_without_replaying_acknowledged_events(
    tmp_path: Path, backend: str, include_content: bool,
) -> None:
    async with _runtime(tmp_path, backend=backend) as runtime:
        task = runtime.tasks.from_agent("metadata.respond", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("terminal-metadata", (
            TaskNode("agent", task=task, input=AgentTaskInput("private-input-marker")),
        )), idempotency_key="terminal")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        execution = await run.execution("agent")
        stored = await execution.model_interactions(include_content=True)
        assert "private-input-marker" in str(stored.items[0].request)
        execution_events: list[ExecutionTreeEvent] = []

        async def on_execution(event: ExecutionTreeEvent) -> None:
            execution_events.append(event)

        execution_outcome = await execution.wait(on_event=on_execution, timeout_seconds=5)
        assert execution_outcome.result.status.value == "SUCCEEDED"
        assert execution_outcome.observation_error is None
        assert execution_events

        events = await asyncio.wait_for(_collect(run.watch(include_content=include_content)), 10)
        assert isinstance(events[0].event, TaskGraphProjection)
        assert events[0].event.phase == "initial"
        assert events[0].event.graph.event_seq > 0
        assert decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, events[0].cursor,
            include_content=include_content,
        ) == (0, {})
        projections = [event.event for event in events if isinstance(event.event, TaskModelProjection)]
        assert projections
        assert {item.item.status for item in projections} == {"SUCCEEDED"}
        assert {item.item.execution_id for item in projections} == {execution.execution_id}
        assert all(item.item.request == {} and item.item.response is None and not item.item.content_included for item in projections)
        assert all(item.item.usage is not None for item in projections)
        assert "private-input-marker" not in str([
            event.event for event in events if isinstance(event.event, (TaskGraphProjection, TaskModelProjection))
        ])
        checkpoint = events[-1].event
        assert isinstance(checkpoint, TaskGraphProjection)
        assert checkpoint.phase == "final"
        assert checkpoint.coverage is not None
        assert checkpoint.coverage.durable_events_complete
        assert checkpoint.coverage.state_complete

        cursor = events[-1].cursor
        reconnected = await asyncio.wait_for(_collect(run.watch(cursor=cursor, include_content=include_content)), 10)
        assert isinstance(reconnected[0].event, TaskGraphProjection)
        assert any(isinstance(event.event, TaskModelProjection) for event in reconnected)
        assert not any(isinstance(event.event, (TaskEvent, ExecutionTreeEvent)) for event in reconnected)
        position = decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, cursor, include_content=include_content,
        )
        assert all(decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, event.cursor, include_content=include_content,
        ) == position for event in reconnected)


@pytest.mark.asyncio
async def test_dynamic_graph_projection_defines_unbound_nodes_before_execution_updates(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    release_child = asyncio.Event()

    class ControlledModels(FixtureModels):
        def materialize(self) -> FunctionModel:
            async def respond(messages, info):
                await release_child.wait()
                yield "dynamic answer"
            return FunctionModel(stream_function=respond)

    async def expand_source(context: TaskNodeContext[None]) -> JsonValue:
        entered.set()
        await release.wait()
        return "expand"

    root_task = Task("metadata.expand-source", expand_source, effect_policy="none")
    async with _runtime(tmp_path, models=ControlledModels()) as runtime:
        agent_task = runtime.tasks.from_agent("metadata.dynamic-agent", runtime.agents.get())

        def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            return (
                TaskNode("child", ("root",), task=agent_task, input=AgentTaskInput("private-dynamic-input")),
                TaskNode("later", ("child",), task=agent_task, input=AgentTaskInput("private-later-input")),
            )

        expander = TaskExpander("metadata.expand", expand)
        run = await runtime.tasks.bind(root_task, agent_task, expander).start(TaskGraph("dynamic-metadata", (
            TaskNode("root", task=root_task, expander=expander),
        )), idempotency_key="dynamic")
        await asyncio.wait_for(entered.wait(), 10)
        stream = run.watch()
        first = await asyncio.wait_for(anext(stream), 10)
        assert isinstance(first.event, TaskGraphProjection)
        assert [node.node_id for node in first.event.graph.nodes] == ["root"]
        release.set()
        events = [first]

        async def collect_expansion() -> None:
            try:
                async for event in stream:
                    events.append(event)
                    if isinstance(event.event, TaskGraphProjection) and any(
                        state.node_id == "later" and state.execution_id is None
                        for state in event.event.graph.node_states
                    ):
                        release_child.set()
            finally:
                await stream.aclose()

        try:
            await asyncio.wait_for(collect_expansion(), 15)
        finally:
            release_child.set()
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED

        known_nodes: set[str] = set()
        bound: dict[str, str] = {}
        unbound_dynamic = False
        child_updates = 0
        for event in events:
            if isinstance(event.event, TaskGraphProjection):
                known_nodes.update(node.node_id for node in event.event.graph.nodes)
                for state in event.event.graph.node_states:
                    if state.execution_id is not None:
                        bound[state.node_id] = state.execution_id
                    elif state.node_id == "later":
                        unbound_dynamic = True
            elif isinstance(event.event, ExecutionTreeEvent):
                assert event.node_id in known_nodes
                assert bound[event.node_id] == event.event.root_execution_id
                child_updates += 1
        assert known_nodes == {"root", "child", "later"}
        assert unbound_dynamic
        assert child_updates > 0


@pytest.mark.asyncio
async def test_running_graph_projects_request_finish_without_a_new_model_sequence(tmp_path: Path) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    graph_holding = asyncio.Event()
    release_graph = asyncio.Event()
    running_seen = asyncio.Event()
    terminal_seen = asyncio.Event()

    class ControlledModels(FixtureModels):
        def materialize(self) -> FunctionModel:
            async def respond(messages, info):
                model_started.set()
                await release_model.wait()
                yield "private-response-marker"
            return FunctionModel(stream_function=respond)

    async def hold_graph(context: TaskNodeContext[None]) -> JsonValue:
        graph_holding.set()
        await release_graph.wait()
        return "held"

    hold_task = Task("metadata.hold", hold_graph, effect_policy="none")
    async with _runtime(tmp_path, models=ControlledModels()) as runtime:
        agent_task = runtime.tasks.from_agent("metadata.running-agent", runtime.agents.get())
        run = await runtime.tasks.bind(agent_task, hold_task).start(TaskGraph("running-metadata", (
            TaskNode("agent", task=agent_task, input=AgentTaskInput("private-request-marker")),
            TaskNode("hold", ("agent",), task=hold_task),
        )), idempotency_key="running")
        await asyncio.wait_for(model_started.wait(), 10)
        events: list[TaskGraphRunEvent] = []

        async def observe() -> None:
            stream = run.watch()
            try:
                async for event in stream:
                    events.append(event)
                    if isinstance(event.event, TaskModelProjection):
                        if event.event.item.status == "RUNNING":
                            running_seen.set()
                        elif event.event.item.status == "SUCCEEDED":
                            terminal_seen.set()
            finally:
                await stream.aclose()

        watcher = asyncio.create_task(observe())
        try:
            await asyncio.wait_for(running_seen.wait(), 10)
            release_model.set()
            await asyncio.wait_for(graph_holding.wait(), 10)
            await asyncio.wait_for(terminal_seen.wait(), 10)
            assert not watcher.done()
            assert (await run.state()).status is TaskStatus.RUNNING
            release_graph.set()
            await asyncio.wait_for(watcher, 10)
        finally:
            release_model.set()
            release_graph.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        rows = [event.event.item for event in events if isinstance(event.event, TaskModelProjection)]
        identities = {(item.execution_id, item.agent_run_seq, item.model_request_seq) for item in rows}
        assert len(identities) == 1
        assert {item.model_request_seq for item in rows} == {1}
        assert rows[0].status == "RUNNING"
        assert rows[0].usage is None and rows[0].duration_ns is None
        assert rows[-1].status == "SUCCEEDED"
        assert rows[-1].usage is not None and rows[-1].duration_ns is not None
        assert rows[-1].started_at is not None and rows[-1].finished_at is not None
        assert all(item.request == {} and item.response is None for item in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", [TaskStatus.WAITING, TaskStatus.RECOVERY_REQUIRED])
async def test_graph_wait_final_projection_closes_at_nonterminal_authoritative_boundary(
    tmp_path: Path, boundary: TaskStatus,
) -> None:
    class ExpectedOutput(BaseModel):
        value: str

    async def uncertain_output(context: TaskNodeContext[None]) -> JsonValue:
        return {"wrong": True}

    uncertain = Task(
        "metadata.uncertain", uncertain_output,
        effect_policy="non_replay_safe", output_type=ExpectedOutput,
    )
    async with _runtime(tmp_path) as runtime:
        agent_task = runtime.tasks.from_agent("metadata.boundary-agent", runtime.agents.get())
        stop = (
            TaskNode.wait("stop", dependencies=("agent",))
            if boundary is TaskStatus.WAITING
            else TaskNode("stop", ("agent",), task=uncertain)
        )
        run = await runtime.tasks.bind(agent_task, uncertain).start(TaskGraph("boundary-metadata", (
            TaskNode("agent", task=agent_task, input=AgentTaskInput("finish model before boundary")), stop,
        )), idempotency_key="boundary")
        assert (await run.wait(timeout_seconds=10)).result.wait_status is boundary
        events: list[TaskGraphRunEvent] = []

        async def on_event(event: TaskGraphRunEvent) -> None:
            events.append(event)

        outcome = await run.wait(on_event=on_event, timeout_seconds=3)
        assert outcome.result.wait_status is boundary
        assert outcome.observation_error is None
        assert any(isinstance(event.event, TaskModelProjection) for event in events)
        checkpoint = events[-1].event
        assert isinstance(checkpoint, TaskGraphProjection)
        assert checkpoint.phase == "final"
        assert checkpoint.graph.wait_status is boundary
        assert checkpoint.coverage is not None and checkpoint.coverage.durable_events_complete
        assert checkpoint.coverage.state_complete


@pytest.mark.asyncio
async def test_final_graph_projection_keeps_newer_ack_without_replacing_waiter_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(tmp_path) as runtime:
        agent_task = runtime.tasks.from_agent("metadata.resumed-agent", runtime.agents.get())
        run = await runtime.tasks.bind(agent_task).start(TaskGraph("resumed-metadata", (
            TaskNode.wait("input"),
            TaskNode("agent", ("input",), task=agent_task, input=AgentTaskInput("resume after input")),
        )), idempotency_key="resume")
        boundary = (await run.wait(include_content=True, timeout_seconds=10)).result
        assert boundary.wait_status is TaskStatus.WAITING
        wait_id = next(state.execution_id for state in boundary.node_states if state.node_id == "input")
        await run.resume("input", TaskInputSupplyRequest(runtime.default_principal, wait_id, "accepted", "resume-input"))
        completed = (await run.wait(include_content=True, timeout_seconds=10)).result
        assert completed.wait_status is TaskStatus.SUCCEEDED
        assert completed.event_seq > boundary.event_seq
        delivered_newer = asyncio.Event()
        events: list[TaskGraphRunEvent] = []

        async def stopped_waiter(*args, **kwargs):
            await delivered_newer.wait()
            return boundary

        async def on_event(event: TaskGraphRunEvent) -> None:
            events.append(event)
            if isinstance(event.event, TaskEvent) and event.event.event_seq == completed.event_seq:
                delivered_newer.set()

        monkeypatch.setattr(runtime._graph_service, "wait", stopped_waiter)
        outcome = await run.wait(on_event=on_event, include_content=True, timeout_seconds=5)
        assert outcome.result is boundary
        assert outcome.result.wait_status is TaskStatus.WAITING
        assert outcome.observation_error is None
        sequence, _ = decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, outcome.cursor, include_content=False,
        )
        assert sequence == completed.event_seq
        checkpoint = events[-1].event
        assert isinstance(checkpoint, TaskGraphProjection)
        assert checkpoint.phase == "final" and checkpoint.graph.event_seq == completed.event_seq
        assert checkpoint.graph.wait_status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_graph_observers_keep_independent_ack_and_bound_metadata_reads_for_text_bursts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(tmp_path) as runtime:
        task = runtime.tasks.from_agent("metadata.burst-agent", runtime.agents.get())
        engine = runtime.tasks.bind(task)
        run = await engine.start(TaskGraph("burst-metadata", (
            TaskNode("agent", task=task, input=AgentTaskInput("one model request")),
        )), idempotency_key="burst")
        assert (await run.wait(timeout_seconds=10)).result.wait_status is TaskStatus.SUCCEEDED
        counters = {"graph_state": 0, "model_cutoffs": 0, "model_rows": 0}
        original_state = runtime._graph_service.state
        original_cutoffs = runtime.history.capture_model_interaction_cutoffs
        original_rows = runtime.history.read_model_interaction_metadata
        original_tree = runtime._watch_execution_tree

        async def state(*args, **kwargs):
            counters["graph_state"] += 1
            return await original_state(*args, **kwargs)

        async def cutoffs(*args, **kwargs):
            counters["model_cutoffs"] += 1
            return await original_cutoffs(*args, **kwargs)

        async def rows(*args, **kwargs):
            counters["model_rows"] += 1
            return await original_rows(*args, **kwargs)

        async def tree(*args, **kwargs):
            stream = original_tree(*args, **kwargs)
            injected = False
            try:
                async for event in stream:
                    yield event
                    if not injected:
                        injected = True
                        for index in range(200):
                            yield replace(event, event=replace(
                                event.event, durable_seq=None,
                                event_type=ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value,
                                payload={"text": str(index)},
                            ))
            finally:
                await stream.aclose()

        monkeypatch.setattr(runtime._graph_service, "state", state)
        monkeypatch.setattr(runtime.history, "capture_model_interaction_cutoffs", cutoffs)
        monkeypatch.setattr(runtime.history, "read_model_interaction_metadata", rows)
        monkeypatch.setattr(runtime, "_watch_execution_tree", tree)
        run = await engine.get(run.graph_id)
        before = asyncio.all_tasks()

        async def consume(limit: int | None) -> list[TaskGraphRunEvent]:
            stream = run.watch()
            collected: list[TaskGraphRunEvent] = []
            chunks = 0
            try:
                async for event in stream:
                    collected.append(event)
                    if isinstance(event.event, ExecutionTreeEvent) and event.event.event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
                        chunks += 1
                        if chunks == limit:
                            break
            finally:
                await stream.aclose()
            return collected

        early, complete = await asyncio.wait_for(asyncio.gather(consume(10), consume(None)), 10)
        assert sum(isinstance(event.event, ExecutionTreeEvent) and event.event.event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value for event in early) == 10
        assert sum(isinstance(event.event, ExecutionTreeEvent) and event.event.event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value for event in complete) == 200
        _, early_positions = decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, early[-1].cursor, include_content=False,
        )
        _, complete_positions = decode_graph_watch_cursor(
            runtime.namespace, runtime.tenant_id, run.graph_id, complete[-1].cursor, include_content=False,
        )
        execution_id = next(iter(complete_positions["agent"]))
        assert early_positions["agent"][execution_id] < complete_positions["agent"][execution_id]
        # Two initial projections and one completed observer stay independent of 210 text callbacks.
        assert counters["graph_state"] <= 12
        assert counters["model_cutoffs"] <= 12
        assert counters["model_rows"] <= 12
        assert not [item for item in asyncio.all_tasks() - before if not item.done()]


@pytest.mark.asyncio
async def test_terminal_graph_wait_acknowledges_delayed_initial_projection_before_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(tmp_path) as runtime:
        task = runtime.tasks.from_agent("metadata.delayed-initial", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("delayed-initial", (
            TaskNode("agent", task=task, input=AgentTaskInput("already terminal")),
        )), idempotency_key="delayed-initial")
        assert (await run.wait(timeout_seconds=10)).result.wait_status is TaskStatus.SUCCEEDED
        original = runtime._graph_service.state
        reads = 0

        async def state(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 2:
                await asyncio.sleep(0.01)
            return await original(*args, **kwargs)

        monkeypatch.setattr(runtime._graph_service, "state", state)
        projections: list[TaskGraphProjection] = []
        acknowledged_initial = False

        async def on_event(event: TaskGraphRunEvent) -> None:
            nonlocal acknowledged_initial
            if isinstance(event.event, TaskGraphProjection):
                if event.event.phase == "initial" and event.event.coverage is not None:
                    acknowledged_initial = True
                if event.event.phase == "final":
                    assert acknowledged_initial
                projections.append(event.event)

        outcome = await run.wait(on_event=on_event, timeout_seconds=5)
        assert outcome.result.wait_status is TaskStatus.SUCCEEDED
        assert outcome.observation_error is None
        assert acknowledged_initial
        assert projections[0].phase == "initial"
        assert projections[-1].phase == "final" and projections[-1].coverage is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", [False, True])
async def test_standalone_final_watch_close_releases_subscriptions(
    tmp_path: Path, checkpoint: bool,
) -> None:
    async with _runtime(tmp_path) as runtime:
        task = runtime.tasks.from_agent("metadata.close", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("close-final-watch", (
            TaskNode("agent", task=task, input=AgentTaskInput("done")),
        )), idempotency_key="close-final")
        await run.wait(timeout_seconds=5)
        before = asyncio.all_tasks()
        stream = run.watch()
        try:
            async for event in stream:
                if (
                    isinstance(event.event, TaskGraphProjection)
                    and event.event.phase == "final"
                    and (not checkpoint or event.event.coverage is not None)
                ):
                    break
            else:
                pytest.fail("Final projection was not delivered")
        finally:
            await stream.aclose()
        assert not [task for task in asyncio.all_tasks() - before if not task.done()]
