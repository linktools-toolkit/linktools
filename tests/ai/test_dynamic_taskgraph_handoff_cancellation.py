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
async def test_cancelled_start_defers_binding_to_idempotent_recovery() -> None:
    class Control:
        execution_id: str | None = None

        async def bind_execution(self, execution_id: str) -> None:
            raise AssertionError(f"unexpected execution binding: {execution_id}")

        async def handoff_execution(
            self,
            execution_id: str,
            *,
            occupies_concurrency: bool = True,
        ) -> None:
            raise AssertionError(
                f"unexpected execution handoff: {execution_id}/{occupies_concurrency}"
            )

    started = asyncio.Event()

    async def start_execution(*args: object, **kwargs: object) -> object:
        del args
        assert kwargs["dependency_hold_id"] == "task:graph:node"
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled start must not return")

    async def unused(*args: object, **kwargs: object) -> object:
        raise AssertionError("literal Agent Task does not use this callback")

    async def hold(*args: object, **kwargs: object) -> None:
        del args, kwargs

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
    running = asyncio.create_task(runner.run(invocation, control=Control()))

    await asyncio.wait_for(started.wait(), 1)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running


@pytest.mark.asyncio
async def test_cancelled_handoff_keeps_hold_until_recovery_rebinds() -> None:
    from linktools.ai.core import ExecutionStatus

    class Result:
        status = ExecutionStatus.SUCCEEDED
        output = {"value": "ok"}

    class Execution:
        execution_id = "execution"

        async def wait(self) -> object:
            from linktools.ai.runtime import WaitResult
            return WaitResult(Result(), None)

    class BlockingControl:
        def __init__(self) -> None:
            self.execution_id: str | None = None
            self.handoff_started = asyncio.Event()

        async def bind_execution(self, execution_id: str) -> None:
            self.execution_id = execution_id

        async def handoff_execution(
            self,
            execution_id: str,
            *,
            occupies_concurrency: bool = True,
        ) -> None:
            assert execution_id == "execution"
            assert occupies_concurrency is True
            self.handoff_started.set()
            await asyncio.Event().wait()

    class RecoveryControl:
        execution_id = "execution"

        async def bind_execution(self, execution_id: str) -> None:
            raise AssertionError(f"unexpected rebind: {execution_id}")

        async def handoff_execution(
            self,
            execution_id: str,
            *,
            occupies_concurrency: bool = True,
        ) -> None:
            assert execution_id == "execution"
            assert occupies_concurrency is True

    execution = Execution()
    acquired: list[tuple[str, str]] = []
    released: list[tuple[str, str]] = []

    async def start_execution(*args: object, **kwargs: object) -> object:
        del args
        assert kwargs["dependency_hold_id"] == "task:graph:node"
        return execution

    async def get_execution(
        execution_id: str,
        _principal: Principal,
    ) -> object:
        assert execution_id == "execution"
        return execution

    async def acquire_hold(
        execution_id: str,
        _principal: Principal,
        hold_id: str,
    ) -> None:
        acquired.append((execution_id, hold_id))

    async def release_hold(
        execution_id: str,
        _principal: Principal,
        hold_id: str,
    ) -> None:
        released.append((execution_id, hold_id))

    async def unused(*args: object, **kwargs: object) -> object:
        raise AssertionError("literal Agent Task does not use this callback")

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
        get_execution=get_execution,
        acquire_execution_hold=acquire_hold,
        release_execution_hold=release_hold,
        result_reader=unused,
        result_ref_reader=unused,
        get_prepared_input=unused,
        publish_prepared_input=unused,
        store_prepared_prompt=unused,
        restore_prepared_prompt=unused,
    )
    node = TaskNode(
        "node",
        input=dict(AgentTaskInput("prompt", planning=False, thinking=False)),
    )
    principal = Principal("principal", "tenant", "service")
    first = TaskNodeInvocation(
        node=node,
        graph_id="graph",
        principal=principal,
        correlation={},
        dependency_results={},
    )
    control = BlockingControl()
    running = asyncio.create_task(runner.run(first, control=control))

    await asyncio.wait_for(control.handoff_started.wait(), 1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert released == []

    recovered = TaskNodeInvocation(
        node=node,
        graph_id="graph",
        principal=principal,
        correlation={},
        dependency_results={},
        execution_id="execution",
    )
    result = await runner.run(recovered, control=RecoveryControl())

    assert result.execution_id == "execution"
    assert acquired == [("execution", "task:graph:node")]
    assert released == [("execution", "task:graph:node")]
