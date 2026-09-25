#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for reliable mixed TaskGraph nodes."""

import asyncio
from pathlib import Path

import pytest
from ._task_test_helpers import admit_graph
from linktools.ai.agent import AgentBindingContract
from linktools.ai.capability import CapabilityContribution, CapabilityGroup, TaskExpansionContext
from linktools.ai.core import (
    JsonValue,
    Principal,
    PrincipalKind,
    TaskStatus,
    WorkspaceFileInput,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._agent_task import _dependency_identity_payload
from linktools.ai.runtime.state import RuntimeDomain, SnapshotLimits
from linktools.ai.runtime.state._codec import (
    _encode_persisted_domain,
    decode_domain,
    iter_runtime_object_refs,
)
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import InMemoryObjectStore, StoredPayload, read_object
from linktools.ai.task import (
    LocalTaskGraphLauncher,
    TaskDependency,
    TaskDependencyState,
    TaskFunction,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
    TaskNodeContext,
    TaskNodeInvocation,
    TaskEffectResolution,
    TaskInputSupplyRequest,
    TaskExpanderRef,
    TaskNodeRunControl,
    TaskNodeRunResult,
    TaskResultRef,
)
from linktools.ai.workspace import Workspace
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import BinaryContent


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_input", (False, True))
@pytest.mark.parametrize("dynamic", (False, True))
async def test_graph_freezes_attachments_before_dependencies_finish(
    tmp_path: Path,
    workspace_input: bool,
    dynamic: bool,
) -> None:
    gate = asyncio.Event()
    started = asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        if context.node_id == "plan":
            return "ready"
        started.set()
        await gate.wait()
        return "ready"

    source = tmp_path / "input.txt"
    source.write_text("accepted content", encoding="utf-8")
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.hold", 1, hold)
    application.task(handler, effect_policy="none")
    attachment = (
        WorkspaceFileInput("input.txt", identifier="source")
        if workspace_input
        else BinaryContent(
            data=b"accepted content",
            media_type="text/plain",
            identifier="source",
        )
    )

    class Expand:
        id = "example.attachments"
        revision = 1

        def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            return (
                handler.node("hold"),
                context.agent_task(
                    "default",
                    "consumer",
                    ("inspect", attachment),
                    dependencies=("hold",),
                ),
            )

    application.task_expander(Expand())
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    workspace = Workspace.load(tmp_path)
    storage_root = tmp_path / "state"
    state = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "attachment-test",
        models=_TaskTestModels(),
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        node = runtime.agent("default").task(
            "consumer",
            ("inspect", attachment),
            dependencies=("hold",),
        )
        nodes = (
            (handler.node("plan", expander=TaskExpanderRef("example.attachments", 1)),)
            if dynamic
            else (handler.node("hold"), node)
        )
        graph = TaskGraph("attachment-graph", nodes)
        run = await runtime.start_graph(graph, idempotency_key="attachment-1")
        await asyncio.wait_for(started.wait(), 10)
        source.unlink()
        repeated = await runtime.start_graph(graph, idempotency_key="attachment-1")
        assert repeated.graph_id == run.graph_id
        snapshot = await state.task.tasks.scheduler_state(
            run.graph_id,
            tenant_id=runtime.tenant_id,
        )
        frozen_node = next(n for n in snapshot.nodes if n.node_id == "consumer")
        stored = decode_domain(
            frozen_node.input["user_prompt"]["value"], StoredUserInput
        )
        assert stored.payload.ref is not None
        frozen_bytes = await read_object(
            state.object_store(RuntimeDomain.TASK),
            stored.payload.ref.key,
            expected_digest=stored.payload.ref.digest,
            expected_size=stored.payload.ref.size,
        )
        refs = tuple(
            iter_runtime_object_refs(
                _encode_persisted_domain(frozen_node),
                default_domain=RuntimeDomain.TASK,
            )
        )
        assert (RuntimeDomain.TASK, stored.payload.ref) in refs
        with pytest.raises(AIError) as rejected:
            await runtime.start_graph(
                TaskGraph("forged-input", (handler.node("hold"), frozen_node)),
                idempotency_key="forged-input-1",
            )
        assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
        gate.set()
        completed = await run.wait(timeout_seconds=10)
        assert completed.status is TaskStatus.SUCCEEDED
        consumer = next(n for n in completed.node_results if n.node_id == "consumer")
        record = await state.execution.executions.get(
            consumer.execution_id, tenant_id=runtime.tenant_id,
        )
        assert record.stored_user_input.view["attachments"] == stored.view["attachments"]
        assert record.stored_user_input.view["files"] == stored.view["files"]

    archive = InMemoryObjectStore("archive")
    read_state = RuntimeStorage.filesystem(storage_root)
    await read_state.initialize(
        namespace="attachment-test", tenant_id="default", read_only=True
    )
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        reference = await read_state.export_snapshot(
            object_store=archive, limits=limits
        )
    finally:
        await read_state.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        reference, object_store=archive, root=restored_root, limits=limits
    )
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(
        namespace="attachment-test", tenant_id="default", read_only=True
    )
    try:
        assert frozen_bytes == await read_object(
            restored.object_store(RuntimeDomain.TASK),
            stored.payload.ref.key,
            expected_digest=stored.payload.ref.digest,
            expected_size=stored.payload.ref.size,
        )
    finally:
        await restored.close()


class _TaskTestModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:task"
    vision = False
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
        route_id: "str | None" = None,
    ) -> _TaskTestModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _TaskTestModelBinding.contract:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TaskTestModelBinding()


def test_direct_task_contribution_uses_group_task_defaults() -> None:
    handler = TaskFunction[None]("example.direct", 1, _echo_task)
    direct = CapabilityContribution.from_task(handler)

    group = CapabilityGroup[None]("application")
    group.task(handler)
    captured = asyncio.run(group.capture())
    registered = next(
        item
        for item in captured.contributions
        if item.kind == "task" and item.id == handler.id
    )

    assert direct.contract == registered.contract
    assert direct.contract["effect_policy"] == "non_replay_safe"


def test_task_registration_returns_original_handler() -> None:
    group = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.echo", 1, _echo_task)

    registered = group.task(handler, effect_policy="none")

    assert registered is handler
    assert registered.node("node").node_id == "node"


async def _echo_task(context: TaskNodeContext[None]) -> JsonValue:
    if not context.dependencies:
        return {"value": context.input.get("value")}
    name = next(iter(context.dependencies))
    dependency = context.dependencies[name]
    return {
        "upstream": await context.read_dependency(name),
        "execution_id": dependency.execution_id,
    }


def test_all_terminal_identity_includes_input_refs_without_execution_identity() -> None:
    digest = "b" * 64
    node = TaskNode(
        "consumer",
        dependencies=("failed",),
        input_refs={
            "evidence": TaskResultRef(
                "namespace",
                "tenant",
                "source-graph",
                "source-node",
                digest,
            )
        },
        dependency_policy="all_terminal",
    )
    states = {
        "failed": TaskDependencyState(
            TaskStatus.FAILED,
            error_code=ErrorCode.REQUEST_FIELD_INVALID.value,
            error_digest="a" * 64,
        )
    }
    first = _dependency_identity_payload(
        node,
        {"evidence": TaskDependency("source-node", digest, "execution-1")},
        states,
    )
    second = _dependency_identity_payload(
        node,
        {"evidence": TaskDependency("source-node", digest, "execution-2")},
        states,
    )

    assert first == second
    assert first == [
        {"node_id": "evidence", "result_digest": digest},
        {
            "node_id": "failed",
            "status": TaskStatus.FAILED.value,
            "error_code": ErrorCode.REQUEST_FIELD_INVALID.value,
            "error_digest": "a" * 64,
        },
    ]


def test_task_dependency_state_exposes_only_terminal_semantics() -> None:
    failed = TaskDependencyState(
        TaskStatus.FAILED,
        error_code=ErrorCode.REQUEST_FIELD_INVALID.value,
        error_digest="a" * 64,
    )
    assert failed.to_payload() == {
        "status": TaskStatus.FAILED.value,
        "error_code": ErrorCode.REQUEST_FIELD_INVALID.value,
        "error_digest": "a" * 64,
    }
    assert "execution_id" not in failed.to_payload()



def test_all_terminal_uses_a_distinct_persisted_task_node_wire() -> None:
    default_node = TaskNode("default")
    terminal_node = TaskNode("terminal", dependency_policy="all_terminal")

    default_wire = _encode_persisted_domain(default_node)
    terminal_wire = _encode_persisted_domain(terminal_node)

    assert default_wire["$dataclass"] == "task_node"
    assert "dependency_policy" not in default_wire["fields"]
    assert terminal_wire["$dataclass"] == "task_node_terminal"
    assert "dependency_policy" not in terminal_wire["fields"]


@pytest.mark.asyncio
async def test_all_terminal_tasks_run_after_failed_and_blocked_dependencies() -> None:
    observed: dict[str, TaskStatus] = {}

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def collect(context: TaskNodeContext[None]) -> JsonValue:
        observed.update(
            {key: value.status for key, value in context.dependency_states.items()}
        )
        for name in context.dependency_states:
            with pytest.raises(AIError) as raised:
                await context.read_dependency(name)
            assert raised.value.code is ErrorCode.TASK_DEPENDENCY_FAILED
        return "collected"

    group = CapabilityGroup[None]("application")
    failure = group.task(TaskFunction("example.fail", 1, fail), effect_policy="none")
    collector = group.task(TaskFunction("example.collect", 1, collect), effect_policy="none")
    group.agent(
        "default", model_route="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    async with Runtime.open(
        "terminal-dependencies",
        models=_TaskTestModels(),
        capabilities=(group,),
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph(
                "graph",
                (
                    failure.node("failed"),
                    failure.node("blocked", dependencies=("failed",)),
                    collector.node(
                        "collect",
                        dependencies=("failed", "blocked"),
                        dependency_policy="all_terminal",
                    ),
                    runtime.agent("default").task(
                        "summary",
                        "Summarize upstream states",
                        dependencies=("failed", "blocked", "collect"),
                        dependency_policy="all_terminal",
                    ),
                ),
            ),
            idempotency_key="terminal-dependencies",
        )
        result = await run.wait(timeout_seconds=10)
    assert observed == {"failed": TaskStatus.FAILED, "blocked": TaskStatus.BLOCKED}
    assert {node.node_id: node.status for node in result.node_results} == {
        "failed": TaskStatus.FAILED,
        "blocked": TaskStatus.BLOCKED,
        "collect": TaskStatus.SUCCEEDED,
        "summary": TaskStatus.SUCCEEDED,
    }


class _TestTaskExpander:
    def __init__(self, expander_id: str, revision: int = 1) -> None:
        self.id = expander_id
        self.revision = revision

    def expand(self, context: object) -> tuple[TaskNode, ...]:
        del context
        return ()


class _ApplicationGraphExpander:
    id = "application.graph"
    revision = 1

    def __init__(self, handler: TaskFunction[None]) -> None:
        self._handler = handler

    def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        reference = TaskExpanderRef(self.id, self.revision)
        source_id = context.source_node.node_id
        if source_id == "application-root":
            return (
                self._handler.node(
                    "child-b",
                    input={"value": "b"},
                    dependencies=("child-a",),
                ),
                self._handler.node(
                    "disconnected",
                    input={"value": "disconnected"},
                ),
                self._handler.node(
                    "child-a",
                    input={"value": "a"},
                    expander=reference,
                ),
            )
        if source_id == "child-a":
            return (
                self._handler.node(
                    "grandchild",
                    input={"value": "grandchild"},
                    dependencies=("child-a",),
                ),
            )
        return ()


class _AgentGraphExpander:
    id = "application.agent-graph"
    revision = 1

    def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        return (
            context.agent_task(
                "worker",
                "agent-child",
                "return a child result",
            ),
        )


@pytest.mark.asyncio
async def test_task_handler_revisions_are_exact_and_reserved_namespace_is_closed() -> (
    None
):
    group = CapabilityGroup[None]("application")
    v1 = TaskFunction[None]("example.echo", 1, _echo_task)
    v2 = TaskFunction[None]("example.echo", 2, _echo_task)

    group.task(v1, effect_policy="none")
    group.task(v2, effect_policy="none")
    snapshot = await group.capture()

    assert {
        (item.kind, item.id, item.revision)
        for item in snapshot.contributions
        if item.kind == "task"
    } == {
        ("task", "example.echo", 1),
        ("task", "example.echo", 2),
    }
    with pytest.raises(AIError) as duplicate:
        group.task(TaskFunction[None]("example.echo", 1, _echo_task))
    assert duplicate.value.code is ErrorCode.CAPABILITY_CONFLICT
    with pytest.raises(ValueError):
        TaskFunction[None]("linktools.ai.custom", 1, _echo_task)

    assert group.task_expander(_TestTaskExpander("application.expand"))
    with pytest.raises(AIError) as expander_error:
        group.task_expander(_TestTaskExpander("linktools.ai.expand"))
    assert expander_error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_runtime_accepts_multiple_task_handler_revisions() -> None:
    group = CapabilityGroup[None]("application")
    v1 = TaskFunction[None]("example.runtime-revision", 1, _echo_task)
    v2 = TaskFunction[None]("example.runtime-revision", 2, _echo_task)
    group.task(v1, effect_policy="none")
    group.task(v2, effect_policy="none")
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "task-revisions",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(group,),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph(
                "task-revisions",
                (v1.node("v1"), v2.node("v2")),
            ),
            idempotency_key="task-revisions-run-0001",
        )
        result = await run.wait(timeout_seconds=10)

    assert result.status is TaskStatus.SUCCEEDED
    assert {item.node_id for item in result.node_results} == {"v1", "v2"}


@pytest.mark.asyncio
async def test_task_result_commit_preserves_early_execution_binding() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-result-regression", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("result-graph", (TaskNode("node"),))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )
        payload = StoredPayload.inline_json({"ok": True})

        terminal = await repository.complete(
            None,
            tenant_id="tenant",
            execution_id="execution",
            result_digest=payload.digest,
            graph_id=graph.graph_id,
            node_id="node",
        )

        assert terminal.execution_id == "execution"
        snapshot = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.status is TaskStatus.SUCCEEDED
        assert snapshot.node_states[0].execution_id == "execution"
        assert snapshot.node_states[0].result_digest == payload.digest
        results = await repository.get_results(
            graph.graph_id,
            ("node",),
            tenant_id="tenant",
        )
        assert results["node"].execution_id == "execution"
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
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        first = handler.node("custom-first", input={"value": "seed"})
        agent = runtime.agent("default").task(
            "agent",
            "Return a short test response.",
            dependencies=("custom-first",),
        )
        last = handler.node("custom-last", dependencies=("agent",))
        graph = TaskGraph("mixed-graph", (first, agent, last))

        result = await runtime.run_graph(
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


@pytest.mark.asyncio
async def test_runtime_expands_application_and_agent_tasks_across_batches(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.echo", 1, _echo_task)
    application.task(handler, effect_policy="none")
    application.task_expander(_ApplicationGraphExpander(handler))
    application.task_expander(_AgentGraphExpander())
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    application.agent(
        "worker",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    app_reference = TaskExpanderRef("application.graph", 1)
    agent_reference = TaskExpanderRef("application.agent-graph", 1)
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        graph = TaskGraph(
            "dynamic-expansion",
            (
                handler.node(
                    "application-root",
                    input={"value": "root"},
                    expander=app_reference,
                ),
                handler.node("empty-root", expander=app_reference),
                runtime.agent("default").task(
                    "agent-root",
                    "return a root result",
                    expander=agent_reference,
                ),
            ),
        )

        result = await runtime.run_graph(
            graph,
            idempotency_key="dynamic-expansion-run-0001",
            timeout_seconds=10,
        )

        assert result.status is TaskStatus.SUCCEEDED, result.node_results
        assert {node.node_id for node in result.node_results} == {
            "agent-child",
            "agent-root",
            "application-root",
            "child-a",
            "child-b",
            "disconnected",
            "empty-root",
            "grandchild",
        }
        snapshot = await runtime.graph.state(
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert [node.node_id for node in snapshot.nodes] == sorted(
            node.node_id for node in snapshot.nodes
        )
        agent_child = next(
            node for node in snapshot.nodes if node.node_id == "agent-child"
        )
        binding = AgentBindingContract.from_payload(
            agent_child.input["binding_contract"]
        )
        assert binding.agent_spec.id == "worker"
        assert snapshot.node_states[-1].status is TaskStatus.SUCCEEDED
        child_a_state = next(
            state for state in snapshot.node_states if state.node_id == "child-a"
        )
        assert child_a_state.execution_id is not None
        assert await runtime.read_task_result(
            graph.graph_id,
            "grandchild",
        ) == {
            "upstream": await runtime.read_task_result(graph.graph_id, "child-a"),
            "execution_id": child_a_state.execution_id,
        }
        events = await state.task.tasks.list_events(
            graph.graph_id,
            tenant_id="default",
            after_sequence=0,
            limit=100,
        )
        expanded = {
            event.source_node_id: event.added_node_ids
            for event in events.items
            if event.event_type.value == "GRAPH_EXPANDED"
        }
        assert expanded == {
            "agent-root": ("agent-child",),
            "application-root": ("child-a", "child-b", "disconnected"),
            "child-a": ("grandchild",),
        }



class _EffectOutput(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_public_graph_start_canonicalizes_registered_task_semantics() -> None:
    async def valid_output(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"value": "ok"}

    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.public-start", 1, valid_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
    )
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "task-public-start",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(application,),
    ) as runtime:
        request = TaskGraphRequest(
            TaskGraph("task-public-start", (handler.node("node"),)),
            runtime.default_principal,
            "task-public-start-0001",
        )
        await runtime.graph.start(request)
        graph_state = await state.task.tasks.graph_state(
            "task-public-start",
            tenant_id=runtime.default_principal.tenant_id,
        )

    assert graph_state is not None
    node = graph_state.nodes[0]
    assert node.effect_policy == "non_replay_safe"
    assert node.output_contract is not None
    assert node.output_contract["mode"] == "structured"


async def _invalid_effect_output(context: TaskNodeContext[None]) -> JsonValue:
    del context
    return {"wrong": True}


@pytest.mark.asyncio
async def test_non_replay_safe_applied_resolution_is_owned_by_execution(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.effect-applied", 1, _invalid_effect_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
    )
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph("effect-applied", (handler.node("node"),)),
            idempotency_key="effect-applied-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED

        snapshot = await runtime.graph.state(
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_state = snapshot.node_states[0]
        assert node_state.status is TaskStatus.RECOVERY_REQUIRED
        assert node_state.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_state.fence,
            TaskEffectResolution("applied", {"value": "recovered"}),
            idempotency_key="effect-applied-resolution-0001",
        )

        assert resolved.status is TaskStatus.SUCCEEDED
        assert await run.result("node") == {"value": "recovered"}
        execution = await runtime.execution.result(
            node_state.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.status.value == "SUCCEEDED"
        events = await state.execution.events.list(
            node_state.execution_id,
            tenant_id="default",
            after_sequence=0,
            limit=100,
        )
        assert events.items[-1].payload["task_effect"] == "applied"
        assert events.items[-1].payload["task_effect_value_digest"] == canonical_sha256(
            {"value": "recovered"}
        )


@pytest.mark.asyncio
async def test_non_replay_safe_invalid_applied_value_preserves_effect_fact(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.effect-invalid", 1, _invalid_effect_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
    )
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph("effect-invalid", (handler.node("node"),)),
            idempotency_key="effect-invalid-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED
        snapshot = await runtime.graph.state(
            run.graph_id,
            principal=runtime.default_principal,
        )
        node_state = snapshot.node_states[0]
        assert node_state.execution_id is not None

        resolved = await run.resolve_effect(
            "node",
            node_state.fence,
            TaskEffectResolution("applied", {"wrong": True}),
            idempotency_key="effect-invalid-resolution-0001",
        )

        assert resolved.status is TaskStatus.FAILED
        execution = await runtime.execution.result(
            node_state.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.error_code == ErrorCode.OUTPUT_CONTRACT_INVALID.value
        assert execution.safe_error_details["task_effect"] == "applied"
        assert execution.safe_error_details["task_effect_value_digest"] == canonical_sha256(
            {"wrong": True}
        )


@pytest.mark.asyncio
async def test_not_applied_retries_same_execution_once(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    calls = 0

    async def flaky_effect(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal calls
        del context
        calls += 1
        if calls == 1:
            raise RuntimeError("effect outcome is unknown")
        return {"ok": True}

    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.effect-retry", 1, flaky_effect)
    application.task(handler, effect_policy="non_replay_safe")
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph(
                "effect-retry",
                (handler.node("node", max_attempts=2, retry_delay_seconds=0),),
            ),
            idempotency_key="effect-retry-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED
        before = await runtime.graph.state(
            run.graph_id,
            principal=runtime.default_principal,
        )
        state_before = before.node_states[0]
        assert state_before.execution_id is not None

        resumed = await run.resolve_effect(
            "node",
            state_before.fence,
            TaskEffectResolution("not_applied"),
            idempotency_key="effect-retry-resolution-0001",
        )
        assert resumed.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
        final = await run.wait(timeout_seconds=10)

        assert final.status is TaskStatus.SUCCEEDED
        assert calls == 2
        after = await runtime.graph.state(
            run.graph_id,
            principal=runtime.default_principal,
        )
        assert after.node_states[0].execution_id == state_before.execution_id
        execution = await runtime.execution.inspect(
            state_before.execution_id,
            principal=runtime.default_principal,
        )
        assert execution.task_attempt == 2


@pytest.mark.asyncio
async def test_deferred_input_is_committed_by_execution_and_allows_json_null(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        run = await runtime.start_graph(
            TaskGraph(
                "deferred-input",
                (
                    TaskNode(
                        "input",
                        input={
                            "task_id": "linktools.ai.input",
                            "task_revision": 1,
                        },
                    ),
                ),
            ),
            idempotency_key="deferred-input-run-0001",
        )
        waiting = await run.wait(timeout_seconds=10)
        node_result = waiting.node_results[0]
        assert node_result.status is TaskStatus.WAITING
        assert node_result.execution_id is not None

        request = TaskInputSupplyRequest(
            runtime.default_principal,
            node_result.execution_id,
            None,
            "deferred-input-value-0001",
        )
        resolved = await run.resume("input", request)

        assert resolved.status is TaskStatus.SUCCEEDED
        assert await run.result("input") is None

        replay = await run.resume("input", request)
        assert replay.status is TaskStatus.SUCCEEDED

        same = await runtime.execution.supply_task_input(
            node_result.execution_id,
            principal=runtime.default_principal,
            value=None,
        )
        assert same.status.value == "SUCCEEDED"

        with pytest.raises(AIError) as raised:
            await runtime.execution.supply_task_input(
                node_result.execution_id,
                principal=runtime.default_principal,
                value={"different": True},
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT


class _BindingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.control: TaskNodeRunControl | None = None

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation
        self.control = control
        self.entered.set()
        await self.release.wait()
        payload = StoredPayload.inline_json({"done": True})
        return TaskNodeRunResult(payload.digest)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation


@pytest.mark.asyncio
async def test_local_activity_generation_does_not_lose_pre_wait_handoff_signal() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-observation-regression", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        graph = TaskGraph("observation-graph", (TaskNode("node"),))
        repository = state.task.tasks
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        request = TaskGraphRequest(
            graph,
            principal,
            idempotency_key="observation-graph-run-0001",
        )
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, graph)
        runner = _BindingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="worker")
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )
        await asyncio.wait_for(runner.entered.wait(), 1)

        generation = launcher.graph_activity_generation(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert generation is not None
        assert runner.control is not None
        await runner.control.handoff_execution("execution")
        await asyncio.wait_for(
            launcher.wait_graph_activity(
                graph.graph_id,
                tenant_id="tenant",
                after_generation=generation,
            ),
            0.2,
        )
        snapshot = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.WAITING
        assert snapshot.node_states[0].execution_id == "execution"
        runner.release.set()
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_waiting_recovery_reestablishes_hold_until_task_commit() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-waiting-hold", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    calls: list[str] = []
    try:
        graph = TaskGraph("waiting-hold", (TaskNode("node"),))
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        request = TaskGraphRequest(
            graph,
            principal,
            idempotency_key="waiting-hold-submit-0001",
        )
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            graph,
        )
        lease = await state.task.tasks.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )

        payload = StoredPayload.inline_json({"done": True})

        class WaitingRunner:
            async def run(
                self,
                invocation: TaskNodeInvocation,
                *,
                control: TaskNodeRunControl,
            ) -> TaskNodeRunResult:
                del invocation, control
                raise AssertionError("WAITING recovery must not start a task")

            async def wait_bound(
                self,
                invocation: TaskNodeInvocation,
                execution_id: str,
            ) -> TaskNodeRunResult:
                del invocation
                calls.append(f"wait:{execution_id}")
                return TaskNodeRunResult(
                    payload.digest,
                    execution_id=execution_id,
                )

            async def cancel(self, invocation: TaskNodeInvocation) -> None:
                del invocation

        async def acquire(
            execution_id: str,
            *,
            tenant_id: str,
            hold_id: str,
        ) -> None:
            del tenant_id
            calls.append(f"acquire:{execution_id}:{hold_id}")

        async def release(
            execution_id: str,
            *,
            tenant_id: str,
            hold_id: str,
        ) -> None:
            del tenant_id
            snapshot = await state.task.tasks.graph_state(
                graph.graph_id,
                tenant_id="tenant",
            )
            assert snapshot is not None
            assert snapshot.node_states[0].status is TaskStatus.SUCCEEDED
            calls.append(f"release:{execution_id}:{hold_id}")

        launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            WaitingRunner(),
            owner="worker",
            acquire_execution_hold=acquire,
            release_execution_hold=release,
        )
        await launcher.start(
            TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits())
        )

        async def wait_terminal() -> None:
            while True:
                snapshot = await state.task.tasks.graph_state(
                    graph.graph_id,
                    tenant_id="tenant",
                )
                assert snapshot is not None
                if snapshot.status is TaskStatus.SUCCEEDED:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_terminal(), 1)
        assert calls == [
            "acquire:execution:task:waiting-hold:node",
            "wait:execution",
            "release:execution:task:waiting-hold:node",
        ]
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
    storage_root = tmp_path / "state"
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
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model_route="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    state = RuntimeStorage.filesystem(storage_root)
    graph = TaskGraph("shutdown-graph", (handler.node("node"),))

    async with Runtime.open(
        "default",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        await runtime.start_graph(
            graph,
            idempotency_key="shutdown-graph-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)
        snapshot = await runtime.graph.state(
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert snapshot.node_states[0].status is TaskStatus.RUNNING

    assert cancelled.is_set()
    probe = RuntimeStorage.filesystem(storage_root)
    await probe.initialize(namespace="default", tenant_id="default")
    try:
        snapshot = await probe.task.tasks.graph_state(
            graph.graph_id,
            tenant_id="default",
        )
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.RUNNING
        assert snapshot.node_states[0].owner is not None
        assert snapshot.node_states[0].lease_expires_at is not None
    finally:
        await probe.close()
