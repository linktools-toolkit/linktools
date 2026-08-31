#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for reliable mixed TaskGraph nodes."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.storage import StoredPayload
from linktools.ai.task import (
    LocalTaskGraphLauncher,
    TaskFunction,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
    TaskNodeContext,
    TaskNodeRunControl,
    TaskNodeRunResult,
)
from linktools.ai.workspace import Workspace, trusted_workspace_principal
from pydantic_ai.models.test import TestModel


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
        route_id: "str | None" = None,
    ) -> _TaskTestModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _TaskTestModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TaskTestModelBinding()


async def _echo_task(context: TaskNodeContext[None]) -> JsonValue:
    if not context.dependencies:
        return {"value": context.input.get("value")}
    dependency = next(iter(context.dependencies.values()))
    return {
        "upstream": dependency.output,
        "execution_id": dependency.execution_id,
    }


@pytest.mark.asyncio
async def test_task_handler_versions_are_exact_and_reserved_namespace_is_closed() -> None:
    group = CapabilityGroup[None]("application")
    v1 = TaskFunction[None]("example.echo", 1, _echo_task)
    v2 = TaskFunction[None]("example.echo", 2, _echo_task)

    group.task(v1)
    group.task(v2)
    frozen = await group.freeze()

    assert {(item.kind, item.id) for item in frozen} == {
        ("task", "example.echo@1"),
        ("task", "example.echo@2"),
    }
    with pytest.raises(AIError) as duplicate:
        group.task(TaskFunction[None]("example.echo", 1, _echo_task))
    assert duplicate.value.code is ErrorCode.CAPABILITY_CONFLICT
    with pytest.raises(ValueError):
        TaskFunction[None]("linktools.ai.custom", 1, _echo_task)


@pytest.mark.asyncio
async def test_task_result_commit_preserves_early_execution_binding() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-result-regression", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("result-graph", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
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
            execution_id="execution",
        )
        payload = StoredPayload.inline_json({"ok": True})

        terminal = await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest=payload.digest,
            result_payload=payload,
        )

        assert terminal.execution_id == "execution"
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.status is TaskStatus.SUCCEEDED
        assert snapshot.node_states[0].execution_id == "execution"
        assert snapshot.node_states[0].result_digest == payload.digest
        results = await repository.get_results(
            graph.graph_id,
            ("node",),
            tenant_id="tenant",
        )
        assert results["node"].payload == payload
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_runtime_executes_custom_agent_custom_graph_and_persists_each_result(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.echo", 1, _echo_task)
    application.task(handler)
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = RuntimeState.in_memory()

    async with Runtime.open(
        workspace,
        models=_TaskTestModels(),  # type: ignore[arg-type]
        state=state,
        capabilities=(application,),
    ) as runtime:
        first = handler.node("custom-first", input={"value": "seed"})
        agent = runtime.agent("default").task(
            "agent",
            "Return a short test response.",
            dependencies=("custom-first",),
        )
        last = handler.node("custom-last", dependencies=("agent",))
        graph = TaskGraph("mixed-graph", (first, agent, last))

        result = await runtime.run_graph_and_wait(
            graph,
            idempotency_key="mixed-graph-run-0001",
            timeout_seconds=10,
        )

        assert result.status is TaskStatus.SUCCEEDED
        assert all(node.status is TaskStatus.SUCCEEDED for node in result.node_results)
        first_output = await runtime.read_task_result(graph.graph_id, "custom-first")
        agent_output = await runtime.read_task_result(graph.graph_id, "agent")
        last_output = await runtime.read_task_result(graph.graph_id, "custom-last")
        assert first_output == {"value": "seed"}
        assert isinstance(last_output, dict)
        assert last_output["upstream"] == agent_output
        assert isinstance(last_output["execution_id"], str)
        persisted = await state.task.tasks.get_results(
            graph.graph_id,
            ("custom-first", "agent", "custom-last"),
            tenant_id="default",
        )
        assert set(persisted) == {"custom-first", "agent", "custom-last"}


class _BindingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.control: TaskNodeRunControl | None = None

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: object,
        dependency_results: object,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del node, graph_id, principal, dependency_results
        self.control = control
        self.entered.set()
        await self.release.wait()
        payload = StoredPayload.inline_json({"done": True})
        return TaskNodeRunResult(payload.digest, result_payload=payload)

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: object,
        dependency_results: object,
    ) -> None:
        del node, graph_id, principal, dependency_results


@pytest.mark.asyncio
async def test_local_activity_generation_does_not_lose_pre_wait_bind_signal() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-observation-regression", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        graph = TaskGraph("observation-graph", (TaskNode("node"),))
        repository = state.task.tasks
        principal = trusted_workspace_principal("tenant")
        request = TaskGraphRequest(
            graph,
            principal,
            idempotency_key="observation-graph-run-0001",
        )
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, graph)
        runner = _BindingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="worker")
        await launcher.start(TaskGraphLaunch(graph, principal, TaskGraphLimits()))
        await asyncio.wait_for(runner.entered.wait(), 1)

        generation = launcher.graph_activity_generation(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert generation is not None
        assert runner.control is not None
        await runner.control.bind_execution("execution")
        await asyncio.wait_for(
            launcher.wait_graph_activity(
                graph.graph_id,
                tenant_id="tenant",
                after_generation=generation,
            ),
            0.2,
        )
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.RUNNING
        assert snapshot.node_states[0].execution_id == "execution"
        runner.release.set()
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_runtime_shutdown_leaves_running_custom_task_recoverable(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    state_root = tmp_path / "state"
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.block", 1, blocking_task)
    application.task(handler)
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = RuntimeState.filesystem(state_root)
    graph = TaskGraph("shutdown-graph", (handler.node("node"),))

    async with Runtime.open(
        workspace,
        models=_TaskTestModels(),  # type: ignore[arg-type]
        state=state,
        capabilities=(application,),
    ) as runtime:
        await runtime.run_graph(
            graph,
            idempotency_key="shutdown-graph-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 1)
        snapshot = await runtime.task.inspect_graph_state(
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert snapshot.node_states[0].status is TaskStatus.RUNNING

    assert cancelled.is_set()
    probe = RuntimeState.filesystem(state_root)
    await probe.initialize(namespace=workspace.workspace_id, tenant_id="default")
    try:
        snapshot = await probe.task.tasks.snapshot_graph(
            graph.graph_id,
            tenant_id="default",
        )
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.RUNNING
        assert snapshot.node_states[0].owner is not None
        assert snapshot.node_states[0].lease_expires_at is not None
    finally:
        await probe.close()
