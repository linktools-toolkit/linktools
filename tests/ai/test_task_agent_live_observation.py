#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-backed graph observations follow each producer's live execution port."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import TaskStatus
from linktools.ai.execution import ExecutionTreeEvent
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_count", [1, 2])
async def test_graph_observes_tools_before_models_finish_and_replays_after_reopen(tmp_path, producer_count):
    release = asyncio.Event()
    entered = [asyncio.Event() for _ in range(producer_count)]
    observed = [asyncio.Event() for _ in range(producer_count)]
    requests = []

    def features(index):
        async def evidence(context: AgentContext[None]) -> str:
            return "synthetic evidence"

        async def model(messages, info):
            requests.append(index)
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart("evidence", {}, tool_call_id=f"tool-{index}")])
            entered[index].set()
            await release.wait()
            return ModelResponse(parts=[TextPart("done")])

        group = RuntimeFeatures(f"producer-{index}")
        group.tool(evidence, effect_policy="none")
        group.model("default", _Models(_UsageFunctionModel(model)))
        group.agent(f"agent-{index}", allow_tools=("evidence",), allow_subagents=())
        return group

    storage_path = tmp_path / "state"
    async with AsyncExitStack() as stack:
        persistence = await stack.enter_async_context(Persistence.open(
            "graph-live", storage=PersistenceStorage.filesystem(storage_path),
        ))
        runtimes = [await stack.enter_async_context(Runtime.attach(persistence=persistence, features=features(i)))
                    for i in range(producer_count)]
        tasks = [Task.from_agent(f"worker-{i}", runtime.agents.get(f"agent-{i}"))
                 for i, runtime in enumerate(runtimes)]
        engine = await stack.enter_async_context(TaskEngine.open(*tasks, persistence=persistence))
        graph = TaskGraph("live-tools", tuple(TaskNode(f"node-{i}", task=task, input=AgentTaskInput("inspect"))
                                              for i, task in enumerate(tasks)))
        run = await engine.start(graph, idempotency_key="live-tools")
        events = []

        async def observe(event):
            if isinstance(event.event, ExecutionTreeEvent):
                events.append(event)
                if event.event.event.event_type == "TOOL_CALL_FINISHED":
                    observed[int(event.node_id.removeprefix("node-"))].set()

        waiting = asyncio.create_task(run.wait(on_event=observe, include_event_content=True, timeout_seconds=10))
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 3)
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in observed)), 3)
            assert not release.is_set() and not waiting.done()
        finally:
            release.set()
            outcome = await waiting
        assert outcome.result.status is TaskStatus.SUCCEEDED
        assert outcome.observation_error is None
        for index in range(producer_count):
            finished = [event for event in events if event.node_id == f"node-{index}"
                        and event.event.event.event_type == "TOOL_CALL_FINISHED"]
            assert len(finished) == 1

    request_count = len(requests)
    async with AsyncExitStack() as stack:
        persistence = await stack.enter_async_context(Persistence.open(
            "graph-live", storage=PersistenceStorage.filesystem(storage_path),
        ))
        runtimes = [await stack.enter_async_context(Runtime.attach(persistence=persistence, features=features(i)))
                    for i in range(producer_count)]
        tasks = [Task.from_agent(f"worker-{i}", runtime.agents.get(f"agent-{i}"))
                 for i, runtime in enumerate(runtimes)]
        engine = await stack.enter_async_context(TaskEngine.open(*tasks, persistence=persistence))
        run = await engine.get("live-tools")
        replay = [event async for event in await run.replay_events(include_content=True)]
        finished = [event for event in replay if isinstance(event.event, ExecutionTreeEvent)
                    and event.event.event.event_type == "TOOL_CALL_FINISHED"]
        assert len(finished) == producer_count
        assert len(requests) == request_count


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["task", "agent", "child", "durable"])
async def test_watch_routes_preserve_authorization_cursor_readiness_and_close(route):
    from types import SimpleNamespace
    from linktools.ai.core import Principal
    from linktools.ai.errors import AIError, ErrorCode

    principal = Principal("reader", "default")
    ready = asyncio.Event()
    closed = []
    calls = []
    cursor = {"execution": 7}
    permitted = False

    async def inspect(execution_id, *, principal):
        calls.append(("authorize", execution_id, principal))
        if not permitted:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return SimpleNamespace(binding_kind="task" if route == "task" else "agent", root_execution_id="root")

    def watch(owner):
        async def stream(execution_id, **options):
            calls.append((owner, execution_id, options))
            options["ready"].set()
            try:
                yield owner
                await asyncio.Event().wait()
            finally:
                closed.append(owner)
        return stream

    async with Persistence.open("routing", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(persistence=persistence) as engine:
            engine._execution_reads = SimpleNamespace(inspect=inspect)
            engine._tree_streamer = SimpleNamespace(stream=watch("task-owner"))
            port = SimpleNamespace(watch_tree=watch("agent-owner"))
            if route in {"agent", "task"}:
                engine._agent_execution_ports["execution"] = port
            elif route == "child":
                engine._agent_execution_ports["root"] = port
            with pytest.raises(AIError) as denied:
                await anext(engine._watch_execution_tree("execution", principal=principal, ready=ready))
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert len(calls) == 1 and not ready.is_set()
            permitted = True
            stream = engine._watch_execution_tree("execution", principal=principal,
                                                  after_event_seqs=cursor, include_content=True, ready=ready)
            owner = "agent-owner" if route in {"agent", "child"} else "task-owner"
            assert await anext(stream) == owner
            assert ready.is_set()
            assert calls[-1] == (owner, "execution", {
                "principal": principal, "after_event_seqs": cursor, "include_content": True, "ready": ready,
            })
            await stream.aclose()
            assert closed == [owner]
