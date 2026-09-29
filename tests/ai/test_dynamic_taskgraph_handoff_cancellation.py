#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation behavior for Runtime-owned Agent Task runners."""

import asyncio

import pytest

from linktools.ai.agent import AgentBindingContract
from linktools.ai.core import Principal
from linktools.ai.runtime._agent_task import RuntimeAgentTaskRunner
from linktools.ai.runtime._agent_task_input import AgentTaskInput
from linktools.ai.spec import AgentSpec
from linktools.ai.task import TaskNode, TaskNodeInvocation


@pytest.mark.asyncio
async def test_cancelled_wait_leaves_handed_off_execution_for_graph_recovery() -> None:
    class Execution:
        def __init__(self) -> None:
            self.wait_started = asyncio.Event()
            self.wait_cancelled = asyncio.Event()
            self.cancel_called = False
            self.execution_id = "execution"

        async def wait(self) -> object:
            self.wait_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.wait_cancelled.set()
                raise

        async def cancel(self) -> None:
            self.cancel_called = True

    class Control:
        def __init__(self) -> None:
            self.execution_id: str | None = None
            self.handed_off: list[str] = []

        async def bind_execution(self, execution_id: str) -> None:
            self.execution_id = execution_id

        async def handoff_execution(
            self,
            execution_id: str,
            *,
            occupies_concurrency: bool = True,
        ) -> None:
            assert occupies_concurrency is True
            self.handed_off.append(execution_id)

    async def unused(*args: object, **kwargs: object) -> object:
        raise AssertionError("literal Agent Task does not use this callback")

    async def hold(*args: object, **kwargs: object) -> None:
        del args, kwargs

    execution = Execution()

    async def start_execution(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return execution

    binding = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model"),
        model_contract={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
    )
    runner = RuntimeAgentTaskRunner(
        id="agent-task",
        revision=1,
        runtime_owner=object(),
        agent_id="agent",
        agent_revision=1,
        input_mode="literal",
        planning_default=False,
        thinking_default=False,
        binding_contract=binding.to_payload(),
        build_input=None,
        start_execution=start_execution,
        get_execution=unused,
        acquire_execution_hold=hold,
        release_execution_hold=hold,
        result_reader=unused,
        result_ref_reader=unused,
        get_prepared_input=unused,
        publish_prepared_input=unused,
        store_prepared_prompt=unused,
        restore_prepared_prompt=unused,
    )
    invocation = TaskNodeInvocation(
        node=TaskNode(
            "node",
            input=dict(AgentTaskInput("prompt", planning=False, thinking=False)),
        ),
        graph_id="graph",
        principal=Principal("principal", "tenant", "service"),
        correlation={},
        dependency_results={},
    )
    control = Control()
    running = asyncio.create_task(runner.run(invocation, control=control))

    await asyncio.wait_for(execution.wait_started.wait(), 1)
    assert control.execution_id == "execution"
    assert control.handed_off == ["execution"]

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert execution.wait_cancelled.is_set()
    assert not execution.cancel_called


@pytest.mark.asyncio
async def test_cancelled_start_finishes_durable_handoff_before_propagating_cancel() -> None:
    class Execution:
        execution_id = "execution"

        async def wait(self) -> object:
            raise AssertionError("cancelled start must not enter execution wait")

    class Control:
        def __init__(self) -> None:
            self.execution_id: str | None = None
            self.handed_off: list[str] = []

        async def bind_execution(self, execution_id: str) -> None:
            self.execution_id = execution_id

        async def handoff_execution(
            self,
            execution_id: str,
            *,
            occupies_concurrency: bool = True,
        ) -> None:
            assert occupies_concurrency is True
            self.handed_off.append(execution_id)

    started = asyncio.Event()
    release = asyncio.Event()
    released_holds: list[tuple[str, str]] = []

    async def start_execution(*args: object, **kwargs: object) -> object:
        del args
        assert kwargs["dependency_hold_id"] == "task:graph:node"
        started.set()
        await release.wait()
        return Execution()

    async def unused(*args: object, **kwargs: object) -> object:
        raise AssertionError("literal Agent Task does not use this callback")

    async def acquire_hold(*args: object, **kwargs: object) -> None:
        del args, kwargs

    async def release_hold(
        execution_id: str,
        _principal: Principal,
        hold_id: str,
    ) -> None:
        released_holds.append((execution_id, hold_id))

    binding = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model"),
        model_contract={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
    )
    runner = RuntimeAgentTaskRunner(
        id="agent-task",
        revision=1,
        runtime_owner=object(),
        agent_id="agent",
        agent_revision=1,
        input_mode="literal",
        planning_default=False,
        thinking_default=False,
        binding_contract=binding.to_payload(),
        build_input=None,
        start_execution=start_execution,
        get_execution=unused,
        acquire_execution_hold=acquire_hold,
        release_execution_hold=release_hold,
        result_reader=unused,
        result_ref_reader=unused,
        get_prepared_input=unused,
        publish_prepared_input=unused,
        store_prepared_prompt=unused,
        restore_prepared_prompt=unused,
    )
    invocation = TaskNodeInvocation(
        node=TaskNode(
            "node",
            input=dict(AgentTaskInput("prompt", planning=False, thinking=False)),
        ),
        graph_id="graph",
        principal=Principal("principal", "tenant", "service"),
        correlation={},
        dependency_results={},
    )
    control = Control()
    running = asyncio.create_task(runner.run(invocation, control=control))

    await asyncio.wait_for(started.wait(), 1)
    running.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await running

    assert control.execution_id == "execution"
    assert control.handed_off == ["execution"]
    assert released_holds == [("execution", "task:graph:node")]
