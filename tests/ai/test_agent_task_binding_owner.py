#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-node binding keeps producer dispatch and captured contracts together."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from linktools.ai.agent import AgentBindingContract, bind_output
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime.agent_execution import AgentExecutionPort
from linktools.ai.spec import AgentSpec
from linktools.ai.task import Task, TaskNode, TaskNodeInvocation
from linktools.ai.task.engine._agent import AgentTaskRunner


@pytest.mark.asyncio
async def test_agent_task_binding_preserves_snapshot_and_distinct_port_surfaces() -> None:
    binding = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model", tool_retries=3, output_retries=3),
        model_contract={"route_id": "model", "model_identity": "test:model"},
        selected=(), subagents=(), output_mode="text", output_schema=bind_output().schema_definition,
    )
    reads = []
    starts = []
    gets = []
    principal = Principal("caller", "tenant", "service")
    execution = SimpleNamespace(execution_id="execution")

    class Port(AgentExecutionPort):
        @property
        def binding_contract(self) -> AgentBindingContract:
            reads.append("binding")
            return binding

        async def start(self, prompt: object, **kwargs: object) -> SimpleNamespace:
            starts.append((prompt, kwargs))
            return execution

        async def get(self, execution_id: str, actor: Principal) -> SimpleNamespace:
            gets.append((execution_id, actor))
            return execution

        async def acquire_hold(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Task hold must retain its service dispatch")

    persistence = object()
    service = SimpleNamespace(
        acquire_dependency_hold=AsyncMock(return_value=True),
        inspect=AsyncMock(return_value="view"),
    )
    operations = SimpleNamespace(service=object(), ensure_open=Mock())
    captures = SimpleNamespace(record_invocation=AsyncMock())
    port = Port(
        persistence=persistence, compiled=SimpleNamespace(spec=binding.agent_spec),
        _binding_contract=None, compiler=None, service=service, sessions=None,
        operations=operations, watch_tree=None, resolve_binding=None,
        default_principal=principal, correlation={}, captures=captures,
    )
    task = Task.from_agent("agent-task", SimpleNamespace(execution_port=port))
    admissions = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
        graph_id="graph", budget_scope_id=None, tool_effects="live",
    )))
    executor = None

    def node_executor() -> SimpleNamespace:
        assert executor is not None
        return executor

    ports = {}
    bound = AgentTaskRunner.from_task(
        task, persistence=persistence, task_admissions=admissions, budgets=None,
        materializer=None, node_executor=node_executor, execution_ports=ports,
    )
    runner = bound.runner
    assert reads == ["binding", "binding"]
    operations.ensure_open.assert_called_once()
    assert runner.execution_port is port
    invocation = TaskNodeInvocation(
        TaskNode("node", input=AgentTaskInput("prompt")), "graph", principal, {}, {},
    )
    result = await runner._start_execution(
        invocation, "prompt", files=(), session_id=None, planning=False,
        thinking=False, idempotency_key="key", dependency_hold_id="hold",
    )
    assert result is execution
    assert reads == ["binding", "binding"]
    assert starts[0][1]["principal"] is principal
    assert starts[0][1]["requires_task_invocation_capture"] is True
    admissions.get.assert_awaited_once_with("graph", tenant_id="tenant")
    assert ports == {"execution": port}
    assert await runner._get_execution("execution", principal) is execution
    assert gets == [("execution", principal)]
    assert await runner._acquire_execution_hold("execution", principal, "hold") is None
    service.acquire_dependency_hold.assert_awaited_once_with(
        "execution", tenant_id="tenant", hold_id="hold",
    )
    assert await runner._inspect_execution("execution", principal) == "view"
    await runner._capture_invocation(execution, "execution", invocation)
    captures.record_invocation.assert_awaited_once_with("execution", invocation)
    executor = SimpleNamespace(restore_prepared_agent_prompt=AsyncMock(return_value="late"))
    assert runner._node_executor() is executor


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", ("disabled", "failure", "cancelled"))
async def test_invocation_capture_preserves_optional_owner_and_failure(capture: str) -> None:
    principal = Principal("caller", "tenant", "service")
    invocation = TaskNodeInvocation(
        TaskNode("node", input=AgentTaskInput("prompt")), "graph", principal, {}, {},
    )
    error = asyncio.CancelledError() if capture == "cancelled" else AIError(ErrorCode.STORAGE_UNAVAILABLE)
    captures = None if capture == "disabled" else SimpleNamespace(record_invocation=AsyncMock(side_effect=error))
    port = AgentExecutionPort(
        persistence=None, compiled=None, _binding_contract=None, compiler=None,
        service=None, sessions=None, operations=None, watch_tree=None, resolve_binding=None,
        default_principal=principal, correlation={}, captures=captures,
    )
    if capture == "disabled":
        assert await port.record_invocation("execution", invocation) is None
    else:
        with pytest.raises(type(error)) as raised:
            await port.record_invocation("execution", invocation)
        assert raised.value is error
        captures.record_invocation.assert_awaited_once_with("execution", invocation)


@pytest.mark.parametrize("same_owner", (False, True))
@pytest.mark.parametrize("closed", (False, True))
def test_producer_binding_checks_identity_before_open_state(same_owner: bool, closed: bool) -> None:
    class Owner:
        def __eq__(self, other: object) -> bool:
            raise AssertionError("producer binding requires identity, not equality")

    owner = Owner()
    failure = AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    ensure_open = Mock(side_effect=failure if closed else None)
    port = AgentExecutionPort(
        persistence=owner, compiled=None, _binding_contract=None, compiler=None,
        service=object(), sessions=None,
        operations=SimpleNamespace(service=object(), ensure_open=ensure_open),
        watch_tree=None, resolve_binding=None, default_principal=None,
        correlation={}, captures=None,
    )
    candidate = owner if same_owner else Owner()
    if not same_owner or closed:
        with pytest.raises(AIError) as raised:
            port.validate_persistence(candidate)
        if not same_owner:
            assert raised.value.code is ErrorCode.RUNTIME_SERVICE_MISMATCH
        else:
            assert raised.value is failure
    else:
        assert port.validate_persistence(candidate) is None
    assert ensure_open.call_count == int(same_owner)
