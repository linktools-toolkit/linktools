#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for reliable mixed TaskGraph nodes."""

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from ._task_test_helpers import (
    CapabilityGroup,
    TaskFunction,
    admit_graph,
    agent_task_definition,
    agent_task_node,
    clear_task_test_state,
    get_task_graph_run,
    run_task_graph,
    register_task_definition,
    start_task_graph,
    start_task_request,
    task_engine,
    task_graph_cancel,
    task_graph_resume,
    task_graph_state,
    task_graph_wait,
    task_result,
)
from linktools.ai.agent import restore_output
from linktools.ai.core import (
    JsonValue,
    Principal,
    PrincipalKind,
    TaskStatus,
    WorkspaceFileInput,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    AgentTaskInput,
    AgentTaskInputContext,
    Runtime,
    RuntimeStorage,
)
from linktools.ai.runtime._agent_task import (
    _agent_task_input_identity,
    _dependency_identity_payload,
)
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
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphState,
    TaskNode,
    TaskNodeContext,
    TaskNodeInvocation,
    Task,
    TaskRef,
    TaskEffectResolution,
    TaskInputSupplyRequest,
    TaskExpansionContext,
    TaskExpander,
    TaskExpanderRef,
    TaskNodeRunControl,
    TaskNodeRunResult,
    TaskNodeRunner,
    TaskResultRef,
)
from linktools.ai.task import _local as task_local
from linktools.ai.workspace import Workspace
from pydantic import BaseModel
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.test import TestModel


@pytest.fixture(autouse=True)
def _reset_task_registry() -> None:
    clear_task_test_state()


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

        def __init__(self, consumer: TaskNode) -> None:
            self.consumer = consumer

        def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            del context
            return (
                handler.node("hold"),
                self.consumer,
            )

    application.agent(
        "default",
        model="default",
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
        node = agent_task_node(runtime,
            "consumer",
            ("inspect", attachment),
            dependencies=("hold",),
        )
        if dynamic:
            application.task_expander(Expand(node))
        nodes = (
            (handler.node("plan", expander=TaskExpanderRef("example.attachments", 1)),)
            if dynamic
            else (handler.node("hold"), node)
        )
        graph = TaskGraph("attachment-graph", nodes)
        run = await start_task_graph(runtime, graph, idempotency_key="attachment-1")
        await asyncio.wait_for(started.wait(), 10)
        source.unlink()
        repeated = await start_task_graph(runtime, graph, idempotency_key="attachment-1")
        assert repeated.graph_id == run.graph_id
        snapshot = await state.task.tasks.scheduler_state(
            run.graph_id,
            tenant_id=runtime.tenant_id,
        )
        frozen_node = next(n for n in snapshot.nodes if n.node_id == "consumer")
        stored_prompt = frozen_node.input.get("prompt")
        assert isinstance(stored_prompt, Mapping)
        assert stored_prompt.get("kind") == "stored-user-content-v1"
        stored = decode_domain(stored_prompt.get("value"), StoredUserInput)
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
            await start_task_graph(runtime,
                TaskGraph("forged-input", (handler.node("hold"), frozen_node)),
                idempotency_key="forged-input-1",
            )
        assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
        gate.set()
        completed = await run.wait(timeout_seconds=10)
        assert completed.status is TaskStatus.SUCCEEDED
        consumer = next(n for n in completed.node_results if n.node_id == "consumer")
        assert consumer.status is TaskStatus.SUCCEEDED
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


class _ThinkingTaskTestModelBinding(_TaskTestModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(
            profile={
                "supports_thinking": True,
                "thinking_always_enabled": False,
            }
        )


class _ThinkingTaskTestModels(_TaskTestModels):
    def resolve(self, route_id: str) -> _ThinkingTaskTestModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _ThinkingTaskTestModelBinding()

    def restore(
        self,
        payload: dict[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _ThinkingTaskTestModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _TaskTestModelBinding.contract:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _ThinkingTaskTestModelBinding()


class _StructuredTaskTestModelBinding(_TaskTestModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(
            custom_output_args={"value": "accepted"},
            profile={"supports_json_schema_output": False},
        )


class _StructuredTaskTestModels(_TaskTestModels):
    def resolve(self, route_id: str) -> _StructuredTaskTestModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _StructuredTaskTestModelBinding()

    def restore(
        self,
        payload: dict[str, JsonValue],
        *,
        route_id: "str | None" = None,
    ) -> _StructuredTaskTestModelBinding:
        super().restore(payload, route_id=route_id)
        return _StructuredTaskTestModelBinding()


class _ContractTaskRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del invocation, control
        self.calls += 1
        return TaskNodeRunResult(
            canonical_sha256({"value": "accepted"}),
            "contract-runner-execution",
        )

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        del invocation
        return TaskNodeRunResult(
            canonical_sha256({"value": "accepted"}),
            execution_id,
        )

    async def supply_input(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        value: JsonValue,
    ) -> TaskNodeRunResult:
        del invocation, execution_id, value
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def resolve_effect(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        resolution: TaskEffectResolution,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id, resolution
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        del invocation


def test_task_definitions_keep_explicit_identity_and_contract() -> None:
    definition = Task("example.direct", _echo_task, effect_policy="none")

    assert (definition.ref.id, definition.ref.revision) == ("example.direct", 1)
    assert definition.contract["type"] == "function"
    assert definition.contract["effect_policy"] == "none"
    assert TaskNode("node", task=definition).task == definition.ref


def test_agent_task_input_keeps_but_excludes_unknown_additive_fields() -> None:
    value = dict(
        AgentTaskInput(
            "prompt",
            parameters={"task_id": "business"},
            planning=False,
            thinking=False,
        )
    )
    value["metadata"] = {"source": "host"}

    restored = AgentTaskInput.from_mapping(value)

    assert restored.prompt == "prompt"
    assert restored.parameters == {"task_id": "business"}
    assert restored["metadata"] == {"source": "host"}
    assert "metadata" not in restored.execution_payload()


def test_agent_task_input_authoring_defaults_and_durable_required_fields() -> None:
    authored = AgentTaskInput.from_authoring({"prompt": "prompt"})
    assert set(authored) == {
        "kind",
        "version",
        "prompt",
        "parameters",
        "files",
        "session_id",
        "memory_scope",
        "planning",
        "thinking",
    }
    assert authored.planning is None
    assert authored.thinking is None

    durable = dict(
        AgentTaskInput("prompt", planning=False, thinking=False)
    )
    assert AgentTaskInput.from_mapping(durable).thinking is False
    assert AgentTaskInput.from_mapping(durable).planning is False
    for field in durable:
        missing = dict(durable)
        del missing[field]
        with pytest.raises(AIError) as error:
            AgentTaskInput.from_mapping(missing)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", 1),
        ("version", "1"),
        ("prompt", []),
        ("parameters", []),
        ("files", "file.txt"),
        ("session_id", 1),
        ("memory_scope", 1),
        ("planning", None),
        ("thinking", None),
    ),
)
def test_agent_task_input_durable_known_fields_fail_closed(
    field: str,
    value: JsonValue,
) -> None:
    durable = dict(AgentTaskInput("prompt", planning=False, thinking=False))
    durable[field] = value

    with pytest.raises(AIError) as error:
        AgentTaskInput.from_mapping(durable)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_agent_task_input_unknown_durable_version_is_unsupported() -> None:
    durable = dict(AgentTaskInput("prompt", planning=False, thinking=False))
    durable["version"] = 2

    with pytest.raises(AIError) as error:
        AgentTaskInput.from_mapping(durable)

    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_recovery_preflight_preserves_unknown_agent_input_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await release.wait()
        return {"ready": True}

    application = CapabilityGroup[None]("application")
    gate = TaskFunction[None]("test.recovery-version-gate", 1, hold)
    application.task(gate, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    async with Runtime.open(
        "recovery-version",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
        agent_task = runtime.tasks.from_agent(
            "test.recovery-version-agent",
            runtime.agents.get("default"),
        )
        agent_node = TaskNode(
            "agent",
            ("gate",),
            task=agent_task,
            input=AgentTaskInput("prompt"),
        )
        engine = runtime.tasks.bind(gate, agent_task)
        graph = TaskGraph(
            "recovery-version",
            (gate.node("gate"), agent_node),
        )
        run = await engine.start(
            graph,
            idempotency_key="recovery-version-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)

        preflight = runtime._graph_service._preflight
        original_validate_recovery = preflight.validate_recovery

        def corrupting_validate_recovery(state: TaskGraphState) -> None:
            nodes = []
            for node in state.nodes:
                if node.node_id == "agent":
                    input_value = dict(node.input)
                    input_value["version"] = 2
                    node = TaskNode(
                        node.node_id,
                        node.dependencies,
                        task=node.task,
                        input=input_value,
                        budget_cost=node.budget_cost,
                        expander=node.expander,
                        input_refs=node.input_refs,
                        timeout_seconds=node.timeout_seconds,
                        max_attempts=node.max_attempts,
                        retry_delay_seconds=node.retry_delay_seconds,
                        output_type=node.output_type,
                        output_contract=node.output_contract,
                        effect_policy=node.effect_policy,
                        reconcile=node.reconcile,
                        dependency_policy=node.dependency_policy,
                    )
                nodes.append(node)
            original_validate_recovery(replace(state, nodes=tuple(nodes)))

        monkeypatch.setattr(
            preflight,
            "validate_recovery",
            corrupting_validate_recovery,
        )
        with pytest.raises(AIError) as unsupported:
            await runtime.tasks.bind(gate, agent_task).recover_pending()
        assert unsupported.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

        release.set()
        result = await run.wait(timeout_seconds=10)
        assert result.status is TaskStatus.SUCCEEDED


def test_agent_task_input_identity_excludes_additive_fields() -> None:
    authored = dict(
        AgentTaskInput("prompt", planning=False, thinking=False)
    )
    plain = AgentTaskInput.from_mapping(authored)
    authored["metadata"] = {"source": "host"}
    extended = AgentTaskInput.from_mapping(authored)
    invocation = TaskNodeInvocation(
        TaskNode("node", task=TaskRef("example.agent", 1)),
        "graph",
        Principal("principal", "tenant"),
        {},
        {},
    )

    assert _agent_task_input_identity(
        invocation,
        plain,
        task_id="example.agent",
        task_revision=1,
        binding_digest="a" * 64,
    ) == _agent_task_input_identity(
        invocation,
        extended,
        task_id="example.agent",
        task_revision=1,
        binding_digest="a" * 64,
    )


@pytest.mark.asyncio
async def test_runtime_agent_task_binding_is_owned_by_its_runtime() -> None:
    first_capabilities = CapabilityGroup[None]("runtime-owner-first")
    first_capabilities.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    second_capabilities = CapabilityGroup[None]("runtime-owner-second")
    second_capabilities.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    async with Runtime.open(
        "runtime-owner-first",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(first_capabilities,),
    ) as first_runtime:
        async with Runtime.open(
            "runtime-owner-second",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.in_memory(),
            capabilities=(second_capabilities,),
        ) as second_runtime:
            agent = first_runtime.agents.get("default")
            first_task = first_runtime.tasks.from_agent("example.agent.first", agent)
            second_task = first_runtime.tasks.from_agent("example.agent.second", agent)
            first_runtime.tasks.bind(first_task, second_task)

            with pytest.raises(AIError) as foreign_runtime:
                second_runtime.tasks.bind(first_task)
            assert foreign_runtime.value.code is ErrorCode.RUNTIME_SERVICE_MISMATCH

            changed_contract = dict(first_task.contract)
            changed_config = dict(changed_contract["config"])
            changed_config["agent_id"] = "edited"
            changed_contract["config"] = changed_config
            edited_task = Task.from_runner(
                first_task.id,
                first_task.runner,  # type: ignore[arg-type]
                contract=changed_contract,
            )
            with pytest.raises(AIError) as edited_declaration:
                first_runtime.tasks.bind(edited_task)
            assert edited_declaration.value.code is ErrorCode.REQUEST_FIELD_INVALID

            class FakeAgentRunner:
                pass

            fake_agent_task = Task.from_runner(
                "example.agent.fake",
                FakeAgentRunner(),  # type: ignore[arg-type]
                contract={
                    "version": 1,
                    "type": "agent",
                    "effect_policy": "none",
                    "output_contract": {"kind": "json"},
                    "reconcile": False,
                    "config": {"agent_id": "default", "agent_revision": 1},
                },
            )
            with pytest.raises(AIError) as fake_runner:
                first_runtime.tasks.bind(fake_agent_task)
            assert fake_runner.value.code is ErrorCode.REQUEST_FIELD_INVALID

            incomplete_input_graph = TaskGraph(
                "incomplete-agent-task-input",
                (
                    TaskNode(
                        "node",
                        task=first_task,
                        input={"kind": "agent-task-input", "version": 1},
                    ),
                ),
            )
            with pytest.raises(AIError) as invalid_input:
                await first_runtime.tasks.bind(first_task).start(
                    incomplete_input_graph,
                    idempotency_key="incomplete-agent-input-run-0001",
                )
            assert invalid_input.value.code is ErrorCode.REQUEST_FIELD_INVALID
            assert await first_runtime._task_admissions.get(
                incomplete_input_graph.graph_id,
                tenant_id=first_runtime.tenant_id,
            ) is None


@pytest.mark.asyncio
async def test_task_results_page_reads_only_page_states_and_preserves_null(
    monkeypatch,
) -> None:
    async def return_null(_context: TaskNodeContext[None]) -> JsonValue:
        return None

    tasks = tuple(
        Task(f"example.result-{name}", return_null, effect_policy="none")
        for name in ("a", "b", "c")
    )
    graph = TaskGraph(
        "bounded-task-results",
        tuple(
            TaskNode(f"node-{name}", task=task)
            for name, task in zip(("c", "a", "b"), tasks)
        ),
    )
    state = RuntimeStorage.in_memory()
    async with Runtime.open(
        "bounded-task-results",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
    ) as runtime:
        run = await runtime.tasks.bind(*tasks).start(
            graph,
            idempotency_key="bounded-task-results-run-0001",
        )
        completed = await run.wait(timeout_seconds=10)
        assert completed.status is TaskStatus.SUCCEEDED

        repository = state.task.tasks
        original_get_node_states = repository.get_node_states
        page_reads: list[tuple[str, ...]] = []

        async def get_node_states(
            graph_id: str,
            node_ids: tuple[str, ...],
            *,
            tenant_id: str,
        ):
            page_reads.append(node_ids)
            return await original_get_node_states(
                graph_id,
                node_ids,
                tenant_id=tenant_id,
            )

        async def reject_full_state(*args, **kwargs):
            del args, kwargs
            raise AssertionError("results() must not load full graph state")

        monkeypatch.setattr(repository, "get_node_states", get_node_states)
        monkeypatch.setattr(repository, "graph_state", reject_full_state)

        first = await run.results(limit=2)
        assert first.next_cursor is not None
        assert [item.node_id for item in first.items] == ["node-a", "node-b"]
        assert all(item.output is None and not item.content_included for item in first.items)

        second = await run.results(cursor=first.next_cursor, limit=2)
        assert second.next_cursor is None
        assert [item.node_id for item in second.items] == ["node-c"]
        assert page_reads == [("node-a", "node-b"), ("node-c",)]

        content_page = await run.results(limit=1, include_content=True)
        assert content_page.items[0].output is None
        assert content_page.items[0].content_included is True
        assert page_reads[-1] == ("node-a",)


@pytest.mark.asyncio
async def test_agent_task_thinking_false_and_none_resolve_before_model_calls() -> None:
    application = CapabilityGroup[None]("thinking-agent")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
        thinking="high",
    )
    state = RuntimeStorage.in_memory()
    async with Runtime.open(
        "agent-task-thinking",
        models=_ThinkingTaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(application,),
    ) as runtime:
        task = runtime.tasks.from_agent("example.thinking-worker", runtime.agents.get())
        graph = TaskGraph(
            "agent-task-thinking-graph",
            (
                TaskNode(
                    "explicit-false",
                    task=task,
                    input=AgentTaskInput("false request", thinking=False),
                ),
                TaskNode(
                    "default-thinking",
                    task=task,
                    input=AgentTaskInput("default request"),
                ),
            ),
        )
        run = await runtime.tasks.bind(task).start(
            graph,
            idempotency_key="agent-task-thinking-run-0001",
        )
        result = await run.wait(timeout_seconds=10)

        assert result.status is TaskStatus.SUCCEEDED, result.node_results
        state_view = await run.state(include_content=True)
        inputs = {node.node_id: node.input for node in state_view.nodes}
        assert inputs["explicit-false"]["thinking"] is False
        assert inputs["default-thinking"]["thinking"] == "high"

        for node_id, expected in (("explicit-false", False), ("default-thinking", "high")):
            node_result = next(item for item in result.node_results if item.node_id == node_id)
            assert node_result.execution_id is not None
            execution = await state.execution.executions.get(
                node_result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            assert execution.thinking == expected
            interactions = await runtime.history.model_interactions(
                node_result.execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=10,
            )
            assert len(interactions.items) == 1
            assert interactions.items[0].status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_task_view_rejects_duplicate_exact_definitions() -> None:
    definition = Task("example.echo", _echo_task)
    async with Runtime.open(
        "duplicate-task-view",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        with pytest.raises(AIError) as duplicate:
            runtime.tasks.bind(definition, definition)
    assert duplicate.value.code is ErrorCode.BINDING_CONFLICT


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

    assert default_wire["$dataclass"] == "task_node@2"
    assert "dependency_policy" not in default_wire["fields"]
    assert terminal_wire["$dataclass"] == "task_node_terminal@2"
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
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    async with Runtime.open(
        "terminal-dependencies",
        models=_TaskTestModels(),
        capabilities=(group,),
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        run = await start_task_graph(runtime,
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
                    agent_task_node(runtime,
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

    def __init__(self, task: Task[None]) -> None:
        self._task = task

    def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        del context
        return (
            TaskNode(
                "agent-child",
                task=self._task,
                input=AgentTaskInput(
                    "return a child result",
                    files=("context.txt",),
                    session_id="expander-session",
                    memory_scope="expander-memory",
                ),
                timeout_seconds=12,
                max_attempts=2,
                retry_delay_seconds=0.25,
            ),
        )


@pytest.mark.asyncio
async def test_task_handler_revisions_are_exact_and_reserved_namespace_is_closed() -> (
    None
):
    v1 = TaskFunction[None]("example.echo", 1, _echo_task)
    v2 = TaskFunction[None]("example.echo", 2, _echo_task)
    async with Runtime.open(
        "task-revision-view",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        view = runtime.tasks.bind(v1, v2)
        assert view is not None
        with pytest.raises(AIError) as duplicate:
            runtime.tasks.bind(v1, TaskFunction[None]("example.echo", 1, _echo_task))
        assert duplicate.value.code is ErrorCode.BINDING_CONFLICT
    with pytest.raises(ValueError):
        Task("linktools.ai.custom", _echo_task)

    assert TaskExpander("application.expand", _TestTaskExpander("application.expand").expand)
    with pytest.raises(ValueError):
        TaskExpander("linktools.ai.expand", _TestTaskExpander("linktools.ai.expand").expand)


@pytest.mark.asyncio
async def test_graph_nodes_store_only_the_exact_task_reference() -> None:
    handler = TaskFunction[None]("example.mutable-handler", 1, _echo_task)
    node = handler.node("node")
    assert node.task == handler.ref
    async with Runtime.open(
        "frozen-task-identity",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        run = await start_task_graph(
            runtime,
            TaskGraph("frozen-task-identity", (node,)),
            idempotency_key="frozen-task-identity-0001",
        )
        result = await run.wait(timeout_seconds=10)
        assert await run.result("node") == {"value": None}

    assert result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_runtime_accepts_multiple_task_handler_revisions() -> None:
    group = CapabilityGroup[None]("application")
    v1 = TaskFunction[None]("example.runtime-revision", 1, _echo_task)
    v2 = TaskFunction[None]("example.runtime-revision", 2, _echo_task)
    group.task(v1, effect_policy="none")
    group.task(v2, effect_policy="none")
    group.task_expander(_TestTaskExpander("example.runtime-expand", 1))
    group.task_expander(_TestTaskExpander("example.runtime-expand", 2))
    state = RuntimeStorage.in_memory()

    async with Runtime.open(
        "task-revisions",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(group,),
    ) as runtime:
        run = await start_task_graph(runtime,
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
        model="default",
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
        agent = agent_task_node(runtime,
            "agent",
            "Return a short test response.",
            dependencies=("custom-first",),
        )
        last = handler.node("custom-last", dependencies=("agent",))
        graph = TaskGraph("mixed-graph", (first, agent, last))

        result = await run_task_graph(runtime,
            graph,
            idempotency_key="mixed-graph-run-0001",
            timeout_seconds=10,
        )

        assert result.status is TaskStatus.SUCCEEDED
        assert all(node.status is TaskStatus.SUCCEEDED for node in result.node_results)
        first_output = await task_result(runtime, graph.graph_id, "custom-first")
        agent_output = await task_result(runtime, graph.graph_id, "agent")
        last_output = await task_result(runtime, graph.graph_id, "custom-last")
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
async def test_projected_agent_input_persists_only_declared_source_and_final_input() -> None:
    application = CapabilityGroup[None]("application")
    source = TaskFunction[None]("example.projected-source", 1, _echo_task)
    application.task(source, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    calls: list[str] = []

    async def build_input(context: AgentTaskInputContext) -> str:
        assert context.input == {"request": "review"}
        calls.append(context.node_id)
        value = await context.result("source")
        assert await context.result("source") == value
        return f"{context.prompt}: {value['value']}"

    state = RuntimeStorage.in_memory()
    async with Runtime.open(
        "projected-input",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(application,),
    ) as runtime:
        agent_task = runtime.tasks.from_agent(
            "example.projected-agent",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            "projected-input-graph",
            (
                source.node("source", input={"value": "accepted"}),
                TaskNode(
                    "consumer",
                    ("source",),
                    task=agent_task,
                    input=AgentTaskInput(
                        "base prompt",
                        parameters={"request": "review"},
                        thinking=False,
                    ),
                ),
            ),
        )
        run = await runtime.tasks.bind(source, agent_task).start(
            graph,
            idempotency_key="projected-input-run-0001",
        )
        result = await run.wait(timeout_seconds=10)

        assert result.status is TaskStatus.SUCCEEDED, result.node_results
        assert calls == ["consumer"]
        prepared = await state.task.tasks.get_prepared_input(
            graph.graph_id,
            "consumer",
            tenant_id=runtime.tenant_id,
        )
        assert prepared is not None
        assert len(prepared.source_refs) == 1
        source_name, source_ref = prepared.source_refs[0]
        assert source_name == "source"
        source_record = await state.task.tasks.get_results(
            graph.graph_id,
            ("source",),
            tenant_id=runtime.tenant_id,
        )
        assert source_ref.result_digest == source_record["source"].result_digest
        consumer_result = next(
            node for node in result.node_results if node.node_id == "consumer"
        )
        execution = await state.execution.executions.get(
            consumer_result.execution_id,
            tenant_id=runtime.tenant_id,
        )
        persisted_graph = await state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert persisted_graph is not None
        persisted_consumer = next(
            node for node in persisted_graph.nodes if node.node_id == "consumer"
        )
        assert persisted_consumer.input["thinking"] is False
        assert execution.thinking is False
        assert execution.stored_user_input is not None
        assert execution.stored_user_input.codec == "text"
        assert execution.stored_user_input.payload.decode() == "base prompt: accepted"


@pytest.mark.asyncio
async def test_invalid_graph_request_does_not_reserve_task_definitions() -> None:
    async def run_first(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"owner": "first"}

    async def run_second(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"owner": "second"}

    first = Task("example.admission-owner", run_first, effect_policy="none")
    second = Task(
        "example.admission-owner",
        run_second,
        effect_policy="replay_safe",
    )
    async with Runtime.open(
        "task-admission-owner",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        invalid_graph = TaskGraph(
            "task-admission-owner-graph",
            (TaskNode("node", task=first),),
        )
        with pytest.raises(AIError) as invalid_request:
            await runtime.tasks.bind(first).start(
                invalid_graph,
                idempotency_key="",
            )
        assert invalid_request.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID

        valid_graph = TaskGraph(
            invalid_graph.graph_id,
            (TaskNode("node", task=second),),
        )
        result = await runtime.tasks.bind(second).start(
            valid_graph,
            idempotency_key="task-admission-owner-valid-0001",
        )
        completed = await result.wait(timeout_seconds=10)

    assert completed.status is TaskStatus.SUCCEEDED
    assert completed.node_results[0].status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_duplicate_graph_start_keeps_the_original_task_definition_owner() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def run_first(context: TaskNodeContext[None]) -> JsonValue:
        del context
        calls.append("first")
        entered.set()
        await release.wait()
        return {"owner": "first"}

    async def run_second(context: TaskNodeContext[None]) -> JsonValue:
        del context
        calls.append("second")
        return {"owner": "second"}

    first = Task("example.concurrent-owner", run_first, effect_policy="none")
    second = Task("example.concurrent-owner", run_second, effect_policy="none")
    graph = TaskGraph(
        "concurrent-owner-graph",
        (TaskNode("node", task=first),),
    )
    async with Runtime.open(
        "concurrent-task-owner",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    ) as runtime:
        first_run = await runtime.tasks.bind(first).start(
            graph,
            idempotency_key="concurrent-owner-graph-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)
        repeated = await runtime.tasks.bind(second).start(
            graph,
            idempotency_key="concurrent-owner-graph-0001",
        )
        release.set()
        first_result = await first_run.wait(timeout_seconds=10)
        repeated_result = await repeated.wait(timeout_seconds=10)

    assert first_result.status is TaskStatus.SUCCEEDED
    assert repeated_result.status is TaskStatus.SUCCEEDED
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_runner_task_contract_is_persisted_and_validated_after_reopen(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "runner-contract-state"
    runner = _ContractTaskRunner()
    schema = _EffectOutput.model_json_schema()
    task = Task.from_runner(
        "example.contract-runner",
        runner,  # type: ignore[arg-type]
        contract={
            "version": 1,
            "type": "example.runner",
            "effect_policy": "none",
            "output_contract": {"kind": "schema", "schema": schema},
            "reconcile": False,
        },
    )
    graph = TaskGraph(
        "runner-contract-graph",
        (
            TaskNode.wait("input"),
            TaskNode("runner", ("input",), task=task),
        ),
    )
    state = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "runner-contract",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
    ) as runtime:
        await runtime.tasks.bind(task).start(
            graph,
            idempotency_key="runner-contract-graph-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)
        initial = await state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert initial is not None
        runner_node = next(node for node in initial.nodes if node.node_id == "runner")
        assert runner_node.output_contract == {
            "mode": "structured",
            "schema": schema,
        }
        assert runner_node.reconcile is False

    recovered_storage = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "runner-contract",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=recovered_storage,
    ) as runtime:
        engine = runtime.tasks.bind(task)
        await engine.recover_pending()
        resumed = await engine.get(graph.graph_id)
        await resumed.resume(
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                wait_id,
                {"value": "input"},
                "runner-contract-input-0001",
            ),
        )
        completed = await resumed.wait(timeout_seconds=10)
        recovered_state = await recovered_storage.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.tenant_id,
        )
        assert recovered_state is not None
        runner_node = next(
            node for node in recovered_state.nodes if node.node_id == "runner"
        )
        assert runner_node.output_contract == {
            "mode": "structured",
            "schema": schema,
        }
        assert runner_node.reconcile is False

    assert completed.status is TaskStatus.SUCCEEDED
    assert runner.calls == 1


@pytest.mark.asyncio
async def test_projected_workers_issue_distinct_requests_and_persist_history(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    attachment = b"PROJECTED_ATTACHMENT_PAYLOAD"
    (workspace_root / "brief.txt").write_bytes(attachment)
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    models = _TaskTestModels()
    state = RuntimeStorage.in_memory()

    async def read_source(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {
            "customer": "Harbor-ALPHA",
            "queue": "Queue-BETA",
            "unprojected": "SECRET-UNPROJECTED-SOURCE-FIELD",
        }

    source = Task("example.dual-worker-source", read_source, effect_policy="none")

    async def build_input(context: AgentTaskInputContext) -> str:
        value = await context.result("source")
        assert isinstance(value, dict)
        if context.input["worker"] == "left":
            return f"left projection customer={value['customer']}"
        return f"right projection queue={value['queue']}"

    async with Runtime.open(
        "dual-projected-workers",
        models=models,  # type: ignore[arg-type]
        storage=state,
        capabilities=(
            CapabilityGroup("workspace", workspace=workspace),
            application,
        ),
    ) as runtime:
        left = runtime.tasks.from_agent(
            "example.dual-worker-left",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        right = runtime.tasks.from_agent(
            "example.dual-worker-right",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            "dual-projected-workers-graph",
            (
                TaskNode("source", task=source),
                TaskNode(
                    "left",
                    ("source",),
                    task=left,
                    input=AgentTaskInput(
                        "left request",
                        parameters={"worker": "left"},
                        files=("brief.txt",),
                    ),
                ),
                TaskNode(
                    "right",
                    ("source",),
                    task=right,
                    input=AgentTaskInput(
                        "right request",
                        parameters={"worker": "right"},
                    ),
                ),
            ),
        )
        run = await runtime.tasks.bind(source, left, right).start(
            graph,
            idempotency_key="dual-projected-workers-run-0001",
        )
        result = await run.wait(timeout_seconds=10)

        assert result.status is TaskStatus.SUCCEEDED, result.node_results

        source_record = (
            await state.task.tasks.get_results(
                graph.graph_id,
                ("source",),
                tenant_id=runtime.tenant_id,
            )
        )["source"]
        node_results = {node.node_id: node for node in result.node_results}
        for node_id, marker in (
            ("left", "Harbor-ALPHA"),
            ("right", "Queue-BETA"),
        ):
            prepared = await state.task.tasks.get_prepared_input(
                graph.graph_id,
                node_id,
                tenant_id=runtime.tenant_id,
            )
            assert prepared is not None
            assert len(prepared.source_refs) == 1
            source_name, source_ref = prepared.source_refs[0]
            assert source_name == "source"
            assert source_ref.result_digest == source_record.result_digest

            execution_id = node_results[node_id].execution_id
            assert execution_id is not None
            interactions = await runtime.history.model_interactions(
                execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=100,
            )
            assert interactions.next_cursor is None
            assert len(interactions.items) == 1
            interaction = interactions.items[0]
            assert interaction.status == "SUCCEEDED"
            assert interaction.content_included is True
            assert marker in str(interaction.request)
            other_marker = "Queue-BETA" if node_id == "left" else "Harbor-ALPHA"
            assert other_marker not in str(interaction.request)
            assert "SECRET-UNPROJECTED-SOURCE-FIELD" not in str(interaction.request)

            history = await runtime.history.history(
                execution_id,
                principal=runtime.default_principal,
                include_content=True,
                limit=100,
            )
            assert history.next_cursor is None
            assert any(
                item.content_included and marker in str(item.content)
                for item in history.items
            )


@pytest.mark.asyncio
async def test_projected_agent_file_is_reused_from_prepared_input_on_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    attachment = b"RECOVERED_PROJECTED_ATTACHMENT"
    attachment_path = workspace_root / "brief.txt"
    attachment_path.write_bytes(attachment)
    workspace = Workspace.load(workspace_root)
    storage_root = tmp_path / "projected-file-state"
    monkeypatch.setattr(task_local, "_LEASE_SECONDS", 1)
    application = CapabilityGroup[None]("application")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    models = _TaskTestModels()
    builds: list[str] = []
    entered_start = asyncio.Event()

    async def source_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"ticket": "RECOVERY-317"}

    async def build_input(context: AgentTaskInputContext) -> str:
        builds.append(context.node_id)
        source = await context.result("source")
        assert isinstance(source, dict)
        return f"inspect projected ticket {source['ticket']}"

    source = Task("example.projected-file-source", source_task, effect_policy="none")
    state = RuntimeStorage.filesystem(storage_root)
    graph_id = "projected-file-recovery-graph"

    async with Runtime.open(
        "projected-file-recovery",
        models=models,  # type: ignore[arg-type]
        storage=state,
        capabilities=(
            CapabilityGroup("workspace", workspace=workspace),
            application,
        ),
    ) as runtime:
        worker = runtime.tasks.from_agent(
            "example.projected-file-worker",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        graph = TaskGraph(
            graph_id,
            (
                TaskNode("source", task=source),
                TaskNode(
                    "consumer",
                    ("source",),
                    task=worker,
                    input=AgentTaskInput(
                        "base request",
                        files=("brief.txt",),
                    ),
                ),
            ),
        )
        original_start = runtime._start_for_agent

        async def pause_before_execution_start(
            *args: object,
            **kwargs: object,
        ) -> object:
            entered_start.set()
            await asyncio.Event().wait()
            return await original_start(*args, **kwargs)  # type: ignore[arg-type]

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_start_for_agent", pause_before_execution_start)
            await runtime.tasks.bind(source, worker).start(
                graph,
                idempotency_key="projected-file-recovery-run-0001",
            )
            await asyncio.wait_for(entered_start.wait(), 10)
            prepared = await state.task.tasks.get_prepared_input(
                graph.graph_id,
                "consumer",
                tenant_id=runtime.tenant_id,
            )
            assert prepared is not None
            file_views = prepared.stored_user_input.view["files"]
            assert isinstance(file_views, list) and len(file_views) == 1
            assert file_views[0]["digest"] == hashlib.sha256(attachment).hexdigest()
            assert builds == ["consumer"]
            await runtime.close()

    attachment_path.unlink()
    await asyncio.sleep(1.1)
    recovered_storage = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "projected-file-recovery",
        models=models,  # type: ignore[arg-type]
        storage=recovered_storage,
        capabilities=(
            CapabilityGroup("workspace", workspace=workspace),
            application,
        ),
    ) as runtime:
        worker = runtime.tasks.from_agent(
            "example.projected-file-worker",
            runtime.agents.get("default"),
            build_input=build_input,
        )
        engine = runtime.tasks.bind(source, worker)
        await engine.recover_pending()
        recovered_run = await engine.get(graph.graph_id)
        completed = await recovered_run.wait(timeout_seconds=10)

        assert completed.status is TaskStatus.SUCCEEDED, completed.node_results
        assert builds == ["consumer"]
        consumer_result = next(
            node for node in completed.node_results if node.node_id == "consumer"
        )
        assert consumer_result.execution_id is not None
        interactions = await runtime.history.model_interactions(
            consumer_result.execution_id,
            principal=runtime.default_principal,
            include_content=True,
            limit=100,
        )
        assert interactions.next_cursor is None
        assert len(interactions.items) == 1
        assert interactions.items[0].status == "SUCCEEDED"
        assert "RECOVERY-317" in str(interactions.items[0].request)
        assert "brief.txt" in str(interactions.items[0].request)
        history = await runtime.history.history(
            consumer_result.execution_id,
            principal=runtime.default_principal,
            include_content=True,
            limit=100,
        )
        assert any(
            item.content_included and "RECOVERY-317" in str(item.content)
            for item in history.items
        )


@pytest.mark.asyncio
async def test_runtime_expands_application_and_agent_tasks_across_batches(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "context.txt").write_text("context", encoding="utf-8")
    workspace = Workspace.load(workspace_root)
    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.echo", 1, _echo_task)
    application.task(handler, effect_policy="none")
    application.task_expander(_ApplicationGraphExpander(handler))
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )
    application.agent(
        "worker",
        model="default",
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
        worker_task = agent_task_definition(runtime, "worker", "test.agent.worker")
        application.task_expander(_AgentGraphExpander(worker_task))
        await runtime.agents.get("worker").create_session(
            "expander-session",
            idempotency_key="expander-session-create-0001",
        )
        graph = TaskGraph(
            "dynamic-expansion",
            (
                handler.node(
                    "application-root",
                    input={"value": "root"},
                    expander=app_reference,
                ),
                handler.node("empty-root", expander=app_reference),
                agent_task_node(runtime,
                    "agent-root",
                    "return a root result",
                    expander=agent_reference,
                ),
            ),
        )

        result = await run_task_graph(runtime,
            graph,
            idempotency_key="dynamic-expansion-run-0001",
            timeout_seconds=10,
        )

        assert result.status is TaskStatus.SUCCEEDED, result.node_results
        assert await task_result(runtime,
            graph.graph_id,
            "agent-child",
        ) is not None
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
        snapshot = await task_graph_state(runtime,
            graph.graph_id,
            principal=runtime.default_principal,
        )
        assert [node.node_id for node in snapshot.nodes] == sorted(
            node.node_id for node in snapshot.nodes
        )
        agent_child = next(
            node for node in snapshot.nodes if node.node_id == "agent-child"
        )
        assert agent_child.input_refs == {}
        assert agent_child.timeout_seconds == 12
        assert agent_child.max_attempts == 2
        assert agent_child.retry_delay_seconds == 0.25
        assert agent_child.task == worker_task.ref
        assert "binding_contract" not in agent_child.input
        child_state = next(
            state for state in snapshot.node_states if state.node_id == "agent-child"
        )
        assert child_state.execution_id is not None
        child_execution = await runtime.executions.inspect(
            child_state.execution_id,
            principal=runtime.default_principal,
        )
        assert child_execution.session_id == "expander-session"
        persisted = await state.task.tasks.graph_state(
            graph.graph_id,
            tenant_id=runtime.default_principal.tenant_id,
        )
        assert persisted is not None
        persisted_agent_child = next(
            node for node in persisted.nodes if node.node_id == "agent-child"
        )
        persisted_input = AgentTaskInput.from_mapping(persisted_agent_child.input)
        assert persisted_input.files == ("context.txt",)
        assert persisted_input.session_id == "expander-session"
        assert persisted_input.memory_scope == "expander-memory"
        assert snapshot.node_states[-1].status is TaskStatus.SUCCEEDED
        child_a_state = next(
            state for state in snapshot.node_states if state.node_id == "child-a"
        )
        assert child_a_state.execution_id is not None
        assert await task_result(runtime,
            graph.graph_id,
            "grandchild",
        ) == {
            "upstream": await task_result(runtime, graph.graph_id, "child-a"),
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


async def _wait_for_input_execution(runtime: Runtime, graph_id: str) -> str:
    async def wait() -> str:
        while True:
            snapshot = await task_graph_state(runtime,
                graph_id,
                principal=runtime.default_principal,
            )
            input_state = next(
                state for state in snapshot.node_states if state.node_id == "input"
            )
            if input_state.status is TaskStatus.WAITING:
                assert input_state.execution_id is not None
                return input_state.execution_id
            await asyncio.sleep(0)

    return await asyncio.wait_for(wait(), timeout=10)


@pytest.mark.asyncio
async def test_reopened_expansion_uses_captured_candidates_and_supports_nesting(
    tmp_path: Path,
) -> None:
    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        return {"node": context.node_id}

    handler = TaskFunction[None]("example.reopened-expansion", 1, run_task)
    unused = TaskFunction[None]("example.unused-expansion", 1, run_task)
    expander_reference = TaskExpanderRef("example.reopened-expander", 1)

    class Expander:
        id = expander_reference.id
        revision = expander_reference.revision

        def expand(self, context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            if context.source_node.node_id == "root":
                return (
                    handler.node(
                        "child",
                        dependencies=("root",),
                        expander=expander_reference,
                    ),
                )
            if context.source_node.node_id == "child":
                return (handler.node("grandchild", dependencies=("child",)),)
            return ()

    expander = Expander()
    graph = TaskGraph(
        "reopened-expansion",
        (
            TaskNode.wait("input"),
            handler.node(
            "root",
            dependencies=("input",),
            expander=expander_reference,
        ),
        ),
    )
    storage_root = tmp_path / "state"
    initial_application = CapabilityGroup[None]("application")
    initial_application.task(handler, effect_policy="none")
    initial_application.task(unused, effect_policy="none")
    initial_application.task_expander(expander)
    async with Runtime.open(
        "reopened-expansion",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.filesystem(storage_root),
        capabilities=(initial_application,),
    ) as runtime:
        await start_task_graph(runtime,
            graph,
            idempotency_key="reopened-expansion-run-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)

    application = CapabilityGroup[None]("application")
    application.task(handler, effect_policy="none")
    application.task(unused, effect_policy="replay_safe")
    application.task_expander(expander)
    async with Runtime.open(
        "reopened-expansion",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.filesystem(storage_root),
        capabilities=(application,),
    ) as runtime:
        principal = runtime.default_principal
        await task_graph_resume(runtime,
            graph.graph_id,
            "input",
            TaskInputSupplyRequest(
                principal,
                wait_id,
                {"value": "seed"},
                "reopened-expansion-input-0001",
            ),
        )
        result = await task_graph_wait(runtime,
            graph.graph_id,
            principal=principal,
            timeout_seconds=10,
        )

    assert result.status is TaskStatus.SUCCEEDED
    assert {node.node_id for node in result.node_results} == {
        "input",
        "root",
        "child",
        "grandchild",
    }


@pytest.mark.asyncio
async def test_reopened_expansion_rejects_a_new_runtime_task_candidate(
    tmp_path: Path,
) -> None:
    async def run_task(context: TaskNodeContext[None]) -> JsonValue:
        return {"node": context.node_id}

    required = TaskFunction[None]("example.reopened-required", 1, run_task)
    added = TaskFunction[None]("example.added-after-capture", 1, run_task)
    expander = TaskExpander(
        "example.added-candidate-expander",
        lambda _context: (added.node("added"),),
    )
    graph = TaskGraph(
        "reopened-added-candidate",
        (
            TaskNode.wait("input"),
            required.node(
                "root",
                dependencies=("input",),
                expander=expander.ref,
            ),
        ),
    )
    storage_root = tmp_path / "state"
    async with Runtime.open(
        "reopened-added-candidate",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.filesystem(storage_root),
    ) as runtime:
        await runtime.tasks.bind(required, expander).start(
            graph,
            idempotency_key="reopened-added-candidate-run-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)

    async with Runtime.open(
        "reopened-added-candidate",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.filesystem(storage_root),
    ) as runtime:
        principal = runtime.default_principal
        run = await runtime.tasks.bind(required, added, expander).get(
            graph.graph_id,
            principal=principal,
        )
        await run.resume(
            "input",
            TaskInputSupplyRequest(
                principal,
                wait_id,
                {"value": "seed"},
                "reopened-added-candidate-input-0001",
            ),
        )
        result = await run.wait(timeout_seconds=10)

    root = next(node for node in result.node_results if node.node_id == "root")
    assert result.status is TaskStatus.FAILED
    assert root.error_code == ErrorCode.BINDING_NOT_REGISTERED.value
    assert all(node.node_id != "added" for node in result.node_results)



class _EffectOutput(BaseModel):
    value: str


class _ChangedEffectOutput(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_public_graph_start_canonicalizes_registered_task_semantics() -> None:
    async def valid_output(context: TaskNodeContext[None]) -> JsonValue:
        del context
        return {"value": "ok"}

    async def reconcile(
        context: TaskNodeContext[None],
    ) -> TaskEffectResolution:
        del context
        return TaskEffectResolution("unknown")

    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.public-start", 1, valid_output)
    application.task(
        handler,
        effect_policy="non_replay_safe",
        output_type=_EffectOutput,
        reconcile=reconcile,
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
        await start_task_request(runtime, request)
        graph_state = await state.task.tasks.graph_state(
            "task-public-start",
            tenant_id=runtime.default_principal.tenant_id,
        )

    assert graph_state is not None
    node = graph_state.nodes[0]
    assert node.effect_policy == "non_replay_safe"
    assert node.output_contract is not None
    assert node.output_contract["mode"] == "structured"
    assert node.reconcile is True


@pytest.mark.asyncio
async def test_persisted_node_output_contracts_survive_runtime_reopen(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = Workspace.load(workspace_root)
    storage_root = tmp_path / "state"

    async def copy_input(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("input")

    application = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.typed-node", 1, copy_input)
    application.task(handler, effect_policy="none")
    application.agent(
        "default",
        model="default",
        allow_tools=(),
        allow_skills=(),
        allow_subagents=(),
    )

    state = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "typed-contract-reopen",
        models=_StructuredTaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        graph = TaskGraph(
            "typed-contract-reopen",
            (
                TaskNode.wait("input", output_type=_EffectOutput),
                TaskNode(
                    "typed-custom",
                    dependencies=("input",),
                    task=handler,
                    output_type=_EffectOutput,
                ),
                agent_task_node(runtime,
                    "typed-agent",
                    "return a structured result",
                    dependencies=("typed-custom",),
                    output_type=_EffectOutput,
                ),
            ),
        )
        await start_task_graph(runtime,
            graph,
            idempotency_key="typed-contract-reopen-0001",
        )
        wait_id = await _wait_for_input_execution(runtime, graph.graph_id)
        invalid_custom_graph = TaskGraph(
            "typed-custom-invalid-reopen",
            (
                TaskNode.wait("input"),
                TaskNode(
                    "typed-custom-invalid",
                    dependencies=("input",),
                    task=handler,
                    output_type=_ChangedEffectOutput,
                ),
            ),
        )
        await start_task_graph(runtime,
            invalid_custom_graph,
            idempotency_key="typed-custom-invalid-reopen-0001",
        )
        invalid_custom_wait_id = await _wait_for_input_execution(
            runtime,
            invalid_custom_graph.graph_id,
        )

    state = RuntimeStorage.filesystem(storage_root)
    async with Runtime.open(
        "typed-contract-reopen",
        models=_StructuredTaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
    ) as runtime:
        recovered = await task_graph_state(runtime,
            "typed-contract-reopen",
            principal=runtime.default_principal,
        )
        contracts = {node.node_id: node.output_contract for node in recovered.nodes}
        assert all(
            contracts[node_id] is not None
            and contracts[node_id]["mode"] == "structured"
            for node_id in ("input", "typed-custom", "typed-agent")
        )
        agent_contract = contracts["typed-agent"]
        assert agent_contract is not None
        restored_agent_output = restore_output(
            agent_contract["mode"],
            agent_contract["schema"],
        )
        with pytest.raises(AIError) as invalid_agent_output:
            restored_agent_output.validate_payload({"wrong": True})
        assert invalid_agent_output.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED

        with pytest.raises(AIError) as invalid_input:
            await task_graph_resume(runtime,
                "typed-contract-reopen",
                "input",
                TaskInputSupplyRequest(
                    runtime.default_principal,
                    wait_id,
                    {"wrong": True},
                    "typed-contract-invalid-input-0001",
                ),
            )
        assert invalid_input.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

        await task_graph_resume(runtime,
            "typed-contract-reopen",
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                wait_id,
                {"value": "accepted"},
                "typed-contract-valid-input-0001",
            ),
        )
        result = await task_graph_wait(runtime,
            "typed-contract-reopen",
            principal=runtime.default_principal,
            timeout_seconds=10,
        )
        assert result.status is TaskStatus.SUCCEEDED
        output = await task_result(runtime,
            "typed-contract-reopen",
            "typed-agent",
        )
        _EffectOutput.model_validate(output)
        await task_graph_resume(runtime,
            "typed-custom-invalid-reopen",
            "input",
            TaskInputSupplyRequest(
                runtime.default_principal,
                invalid_custom_wait_id,
                {"value": "not-an-integer"},
                "typed-custom-invalid-input-0001",
            ),
        )
        invalid_custom_result = await task_graph_wait(runtime,
            "typed-custom-invalid-reopen",
            principal=runtime.default_principal,
            timeout_seconds=10,
        )

    invalid_custom = next(
        node
        for node in invalid_custom_result.node_results
        if node.node_id == "typed-custom-invalid"
    )
    assert invalid_custom_result.status is TaskStatus.FAILED
    assert invalid_custom.error_code == ErrorCode.OUTPUT_VALIDATION_FAILED.value


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
        model="default",
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
        run = await start_task_graph(runtime,
            TaskGraph("effect-applied", (handler.node("node"),)),
            idempotency_key="effect-applied-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED

        snapshot = await task_graph_state(runtime,
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
        execution = await runtime.executions.result(
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
        model="default",
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
        run = await start_task_graph(runtime,
            TaskGraph("effect-invalid", (handler.node("node"),)),
            idempotency_key="effect-invalid-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED
        snapshot = await task_graph_state(runtime,
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
        execution = await runtime.executions.result(
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
        model="default",
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
        run = await start_task_graph(runtime,
            TaskGraph(
                "effect-retry",
                (handler.node("node", max_attempts=2, retry_delay_seconds=0),),
            ),
            idempotency_key="effect-retry-run-0001",
        )
        initial = await run.wait(timeout_seconds=10)
        assert initial.status is TaskStatus.RECOVERY_REQUIRED
        before = await task_graph_state(runtime,
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
        after = await task_graph_state(runtime,
            run.graph_id,
            principal=runtime.default_principal,
        )
        assert after.node_states[0].execution_id == state_before.execution_id
        execution = await runtime.executions.inspect(
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
        model="default",
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
        run = await start_task_graph(runtime,
            TaskGraph(
                "deferred-input",
                (
                    TaskNode.wait("input"),
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
async def test_runtime_recovery_rejects_same_revision_task_semantic_drift(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "semantic-drift-state"
    entered = asyncio.Event()

    async def blocking_task(context: TaskNodeContext[None]) -> JsonValue:
        del context
        entered.set()
        await asyncio.Event().wait()
        return {"value": "unreachable"}

    async def reconcile(
        context: TaskNodeContext[None],
    ) -> TaskEffectResolution:
        del context
        return TaskEffectResolution("unknown")

    first = CapabilityGroup[None]("application")
    handler = TaskFunction[None]("example.semantic-drift", 1, blocking_task)
    first.task(
        handler,
        effect_policy="none",
        output_type=_EffectOutput,
        reconcile=reconcile,
    )
    state = RuntimeStorage.filesystem(storage_root)
    graph = TaskGraph("semantic-drift", (handler.node("node"),))

    async with Runtime.open(
        "semantic-drift",
        models=_TaskTestModels(),  # type: ignore[arg-type]
        storage=state,
        capabilities=(first,),
    ) as runtime:
        await start_task_graph(runtime,
            graph,
            idempotency_key="semantic-drift-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)

    reconcile_removed = CapabilityGroup[None]("application")
    reconcile_removed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        output_type=_EffectOutput,
    )
    with pytest.raises(AIError) as reconcile_error:
        async with Runtime.open(
            "semantic-drift",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.filesystem(storage_root),
            capabilities=(reconcile_removed,),
        ) as runtime:
            await task_engine(runtime).recover_pending()
    assert reconcile_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert reconcile_error.value.safe_details["reason"] == "task_reconcile_changed"

    output_changed = CapabilityGroup[None]("application")
    output_changed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        output_type=_ChangedEffectOutput,
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as output_error:
        async with Runtime.open(
            "semantic-drift",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.filesystem(storage_root),
            capabilities=(output_changed,),
        ) as runtime:
            await task_engine(runtime).recover_pending()
    assert output_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert output_error.value.safe_details["reason"] == "task_output_contract_changed"

    output_removed = CapabilityGroup[None]("application")
    output_removed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="none",
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as removed_output_error:
        async with Runtime.open(
            "semantic-drift",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.filesystem(storage_root),
            capabilities=(output_removed,),
        ) as runtime:
            await task_engine(runtime).recover_pending()
    assert removed_output_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert (
        removed_output_error.value.safe_details["reason"]
        == "task_output_contract_changed"
    )

    effect_changed = CapabilityGroup[None]("application")
    effect_changed.task(
        TaskFunction[None]("example.semantic-drift", 1, blocking_task),
        effect_policy="replay_safe",
        output_type=_EffectOutput,
        reconcile=reconcile,
    )
    with pytest.raises(AIError) as effect_error:
        async with Runtime.open(
            "semantic-drift",
            models=_TaskTestModels(),  # type: ignore[arg-type]
            storage=RuntimeStorage.filesystem(storage_root),
            capabilities=(effect_changed,),
        ) as runtime:
            await task_engine(runtime).recover_pending()
    assert effect_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert effect_error.value.safe_details["reason"] == "task_effect_changed"


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
        model="default",
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
        await start_task_graph(runtime,
            graph,
            idempotency_key="shutdown-graph-run-0001",
        )
        await asyncio.wait_for(entered.wait(), 10)
        snapshot = await task_graph_state(runtime,
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
