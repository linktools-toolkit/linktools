#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task.engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Historical reexecution imports the accepted framework context, not live state."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

from pathlib import Path
from linktools.ai.capability import ExecutionCapability

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.capability import TraceCapability
from linktools.ai.inputs import AgentTaskInputContext, CaptureInputRequest
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.persistence import PersistenceDomain
from .test_task_mixed_node_reliability import _TaskTestModelBinding


class _NoToolsBinding(_TaskTestModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(call_tools=[])


class _NoToolsModels(_NoToolsBinding):
    pass


@pytest.mark.asyncio
async def test_captured_session_context_ignores_later_turns_and_metadata(tmp_path: Path) -> None:
    group = RuntimeFeatures("context")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(TraceCapability(), scope="runtime")
    group.agent("default", model="default", system_prompt="original-only behavior", allow_tools=(), allow_skills=(), allow_subagents=())
    group.agent("candidate", model="default", system_prompt="candidate-only behavior", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('captured-session', storage=storage) as persistence_42, Runtime.attach(persistence=persistence_42, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_42, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        agent = runtime.agents.get()
        session = await agent.create_session("source-session", principal=principal, metadata={"note": "accepted metadata"})
        first = await session.start("first historical turn")
        await first.wait()
        second = await session.start("second input")
        await second.wait()
        captured = await runtime_engine.inputs.capture_input(second.execution_id, CaptureInputRequest(principal, "context-session-0001"))
        value = await runtime_engine.inputs.read_agent(captured, principal=principal)
        assert value.input_context is not None
        assert "first historical turn" in value.input_context.history.decode()
        assert "second input" not in value.input_context.history.decode()
        third = await session.start("later production turn")
        await third.wait()
        replay_principal = Principal("new-owner", runtime.tenant_id)
        replay = await runtime.agents.get("candidate").start(value.prompt, input_context=value.input_context, principal=replay_principal)
        assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
        interaction = (await replay.model_interactions(include_content=True)).items[0]
        assert "candidate-only behavior" in str(interaction.request)
        assert "original-only behavior" not in str(interaction.request)
        record = await runtime._state.execution.executions.get(replay.execution_id, tenant_id=principal.tenant_id)
        assert record.session_id is None and record.parent_execution_id is None and record.context_imported
        assert record.principal_id == replay_principal.principal_id
        recapture = await runtime_engine.inputs.capture_input(replay.execution_id, CaptureInputRequest(replay_principal, "context-replay-0001"))
        restored = await runtime_engine.inputs.read_agent(recapture, principal=replay_principal)
        assert restored.input_context.digest == value.input_context.digest
        assert "later production turn" not in restored.input_context.history.decode()
        assert restored.input_context.session_metadata["note"] == "accepted metadata"


@pytest.mark.asyncio
async def test_captured_memory_preserves_versions_in_new_execution_scope(tmp_path: Path) -> None:
    group = RuntimeFeatures("context-memory")
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('captured-memory', storage=storage) as persistence_78, Runtime.attach(persistence=persistence_78, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_78, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        memory = RuntimeMemoryStore(runtime._state.memory, object_store=runtime._state.object_store(PersistenceDomain.MEMORY),
                                    namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id="setup", memory_scope="production")
        original = await memory.write("facts.txt", "old content", expected_version=None)
        source = await runtime.agents.get().start("question", memory_scope="production", principal=principal)
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        captured = await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(principal, "memory-context-0001"))
        value = await runtime_engine.inputs.read_agent(captured, principal=principal)
        assert value.input_context.memory["facts.txt"]["content"] == "old content"
        await memory.write("facts.txt", "new production content", expected_version=original.version)
        replay = await runtime.agents.get().start(value.prompt, input_context=value.input_context, principal=principal)
        assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(replay.execution_id, tenant_id=principal.tenant_id)
        isolated = RuntimeMemoryStore(runtime._state.memory, object_store=runtime._state.object_store(PersistenceDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id=replay.execution_id, memory_scope=record.memory_scope)
        frozen = await isolated.read("facts.txt", max_chars=1000)
        assert frozen.content == "old content" and frozen.version == original.version
        await isolated.write("facts.txt", "isolated edit", expected_version=frozen.version)
        assert (await memory.read("facts.txt", max_chars=1000)).content == "new production content"


@pytest.mark.asyncio
async def test_captured_agent_task_and_graph_restore_context_after_restart(tmp_path: Path) -> None:
    from linktools.ai.inputs import AgentTaskInput, CaptureGraphRequest
    from linktools.ai.task import TaskGraph, TaskNode

    group = RuntimeFeatures("context-graph")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    principal = Principal("owner", "default")
    async with Persistence.open('captured-graph-context', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_109, Runtime.attach(persistence=persistence_109, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_109, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        agent = runtime.agents.get()
        session = await agent.create_session("graph-session", principal=principal)
        previous = await session.start("remember the earlier turn")
        await previous.wait()
        task = Task.from_agent("context.agent", agent)
        run = await runtime_engine.with_definitions(task).start(TaskGraph("context-source", (TaskNode("agent", task=task,
            input=AgentTaskInput("graph question", session_id="graph-session")),)), principal=principal, idempotency_key="context-graph-source")
        await run.wait()
        graph_capture = await runtime_engine.inputs.capture_graph("context-source", CaptureGraphRequest(principal, "context-graph-capture"))
        template = await runtime_engine.inputs.read_graph(graph_capture, principal=principal)
        assert template.context_policy == "captured"
        assert template.nodes[0].input["capture_context"]["history"]
        latest = await session.start("this later turn must not leak")
        await latest.wait()
    async with Persistence.open('captured-graph-context', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_124, Runtime.attach(persistence=persistence_124, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_124, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("context.agent", runtime.agents.get())
        template = await runtime_engine.inputs.read_graph(graph_capture, principal=principal)
        rerun = await runtime_engine.with_definitions(task).start(TaskGraph("context-replay", template.nodes), principal=principal, idempotency_key="context-graph-replay")
        await rerun.wait()
        execution = await rerun.execution("agent")
        capture = await runtime_engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(principal, "context-graph-rerun-capture"))
        value = await runtime_engine.inputs.read_agent(capture, principal=principal)
        assert "remember the earlier turn" in value.input_context.history.decode()
        assert "this later turn must not leak" not in value.input_context.history.decode()
        assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).session_id is None


@pytest.mark.asyncio
async def test_captured_repository_instructions_never_refresh_from_workspace(tmp_path: Path) -> None:
    from linktools.ai.workspace import Workspace

    work = tmp_path / "workspace"
    work.mkdir()
    instructions = work / "AGENTS.md"
    instructions.write_text("Historical repository instruction", encoding="utf-8")
    group = workspace_features("context-instructions", workspace=Workspace.load(work))
    group.capability(TraceCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    async with Persistence.open('captured-instructions', storage=storage) as persistence_149, Runtime.attach(persistence=persistence_149, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_149, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        source = await runtime.agents.get().start("question", principal=principal)
        await source.wait()
        capture = await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(principal, "instructions-capture"))
        value = await runtime_engine.inputs.read_agent(capture, principal=principal)
        instructions.write_text("Changed live repository instruction", encoding="utf-8")
        rerun = await runtime.agents.get().start(value.prompt, input_context=value.input_context, principal=principal)
        assert (await rerun.wait()).result.status is ExecutionStatus.SUCCEEDED
        interactions = await rerun.model_interactions(include_content=True)
        assert "Historical repository instruction" in str(interactions.items[0].request)
        assert "Changed live repository instruction" not in str(interactions.items[0].request)


@pytest.mark.asyncio
async def test_large_captured_context_survives_portable_storage_snapshot(tmp_path: Path) -> None:
    from linktools.ai.execution import ExecutionInputContext
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    group = RuntimeFeatures("portable-context")
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    principal = Principal("owner", "default")
    baseline = ExecutionInputContext.from_messages((), memory={
        "first.txt": {"content": "a" * 40000, "version": "m1:" + "1" * 64},
        "second.txt": {"content": "b" * 40000, "version": "m1:" + "2" * 64},
    })
    source_root = tmp_path / "source"
    storage = PersistenceStorage.filesystem(source_root)
    async with Persistence.open('portable-context', storage=storage) as persistence_178, Runtime.attach(persistence=persistence_178, features=model_features(_NoToolsModels(), group)) as runtime:
        execution = await runtime.agents.get().start("question", input_context=baseline, principal=principal)
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
        assert record.input_context.payload.kind == "inline"
        manifest = record.input_context.payload.decode()
        assert manifest["kind"] == "execution-context"
        from linktools.ai.inputs import ExecutionContextArchive
        contents = await ExecutionContextArchive(
            runtime._state.execution.executions.state_store,
            runtime._state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace=runtime.namespace, tenant_id=principal.tenant_id,
            object_reader=lambda reference: runtime._state.read_archive_object(
                PersistenceDomain.EXECUTION, reference,
            ),
        )._open()
        memory_contents = await contents.content_records(tuple(
            reference["content_id"] for reference in manifest["memory"].values()
        ))
        assert len(memory_contents) == 2
        assert all(content.payload.kind == "object" for content in memory_contents)
        assert sum(content.raw_size for content in memory_contents) >= 80000
    archive = InMemoryObjectStore("context-snapshot")
    storage = PersistenceStorage.filesystem(source_root)
    storage = Persistence(storage)
    await storage.initialize(namespace="portable-context", tenant_id=principal.tenant_id, read_only=True)
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        snapshot = await storage.export_snapshot(object_store=archive, limits=limits)
    finally:
        await storage.close()
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=archive, root=restored_root, limits=limits)
    async with Persistence.open('portable-context', storage=PersistenceStorage.from_root(restored_root)) as persistence_211, Runtime.attach(persistence=persistence_211, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_211, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        capture = await runtime_engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(principal, "restored-context"))
        imported = await runtime_engine.inputs.read_agent(capture, principal=principal)
        assert imported.input_context.digest == baseline.digest
        rerun = await runtime.agents.get().start(imported.prompt, input_context=imported.input_context, principal=principal)
        assert (await rerun.wait()).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_reprojection_uses_captured_history_without_reopening_source_session(tmp_path: Path) -> None:
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.task import TaskGraph, TaskNode, TaskStatus

    group = RuntimeFeatures("reproject-context")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def first_builder(context: AgentTaskInputContext) -> str:
        return "original projected question"

    async def replacement_builder(context: AgentTaskInputContext) -> str:
        return "replacement projected question"

    async with Persistence.open('reproject-context', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_234, Runtime.attach(persistence=persistence_234, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_234, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        session = await runtime.agents.get().create_session("context-session", principal=principal)
        first = await session.start("immutable earlier history")
        await first.wait()
        original = Task.from_agent("context.original", runtime.agents.get(), build_input=first_builder)
        candidate = Task.from_agent("context.candidate", runtime.agents.get(), build_input=replacement_builder)
        engine = runtime_engine.with_definitions(original, candidate)
        source = await engine.start(TaskGraph("projected-source", (TaskNode("node", task=original,
            input=AgentTaskInput(parameters={"example": 1}, session_id="context-session")),)), principal=principal, idempotency_key="projected-context-source")
        assert (await source.wait()).result.wait_status is TaskStatus.SUCCEEDED
        source_execution = await source.execution("node")
        captured = await runtime_engine.inputs.capture_input(source_execution.execution_id, CaptureInputRequest(principal, "projected-context-capture"))
        source_value = await runtime_engine.inputs.read_agent(captured, principal=principal)
        capture = await runtime_engine.inputs.task_input(captured, principal=principal, input_mode="reproject_input")
        replay = await engine.start(TaskGraph("projected-replay", (TaskNode("node", task=candidate, input_capture=capture),)), principal=principal, idempotency_key="projected-context-replay")
        assert (await replay.wait()).result.wait_status is TaskStatus.SUCCEEDED
        replay_execution = await replay.execution("node")
        replay_capture = await runtime_engine.inputs.capture_input(replay_execution.execution_id, CaptureInputRequest(principal, "projected-context-rerun"))
        value = await runtime_engine.inputs.read_agent(replay_capture, principal=principal)
        assert value.prompt == "replacement projected question"
        assert value.input_context.digest == source_value.input_context.digest
        assert (await runtime.executions.inspect(replay_execution.execution_id, principal=principal)).session_id is None


@pytest.mark.asyncio
async def test_custom_memory_without_consistent_read_view_reports_precise_unavailability(tmp_path: Path) -> None:
    from pydantic_ai_harness.memory import InMemoryStore
    from linktools.ai.errors import AIError, ErrorCode

    group = RuntimeFeatures("unsupported-context")
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    async with Persistence.open('unsupported-context', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_266, Runtime.attach(persistence=persistence_266, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_266, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        backend = runtime._execution_service.runtime_backend()
        backend._memory_store_factory = lambda *_args: InMemoryStore()
        source = await runtime.agents.get().start("question", memory_scope="custom", principal=principal)
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        with pytest.raises(AIError) as raised:
            await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(principal, "unsupported-context-capture"))
        assert raised.value.code is ErrorCode.INPUT_CONTEXT_UNAVAILABLE
        assert raised.value.safe_details["reason"] == "memory_store_has_no_consistent_capture"
        clean = await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(principal, "explicit-clean", "clean"))
        assert (await runtime_engine.inputs.read_agent(clean, principal=principal)).input_context is None


@pytest.mark.asyncio
async def test_reprojected_files_use_accepted_bytes_after_workspace_file_changes(tmp_path: Path) -> None:
    from pydantic_ai.messages import BinaryContent
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.task import TaskGraph, TaskNode, TaskStatus
    from linktools.ai.workspace import Workspace

    work = tmp_path / "workspace"
    work.mkdir()
    path = work / "input.txt"
    path.write_text("accepted file bytes", encoding="utf-8")
    group = workspace_features("file-context", workspace=Workspace.load(work))
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def builder(context: AgentTaskInputContext) -> str:
        return "projected prompt"

    async with Persistence.open('file-context', storage=PersistenceStorage.filesystem(tmp_path / 'state')) as persistence_297, Runtime.attach(persistence=persistence_297, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_297, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        task = Task.from_agent("files.projected", runtime.agents.get(), build_input=builder)
        engine = runtime_engine.with_definitions(task)
        source = await engine.start(TaskGraph("file-source", (TaskNode("node", task=task, input=AgentTaskInput(files=("input.txt",))),)),
                                    principal=principal, idempotency_key="file-context-source")
        assert (await source.wait()).result.wait_status is TaskStatus.SUCCEEDED
        execution = await source.execution("node")
        ref = await runtime_engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(principal, "file-context-capture"))
        path.write_text("changed live file", encoding="utf-8")
        replay_input = await runtime_engine.inputs.task_input(ref, principal=principal, input_mode="reproject_input")
        replay = await engine.start(TaskGraph("file-replay", (TaskNode("node", task=task, input_capture=replay_input),)),
                                   principal=principal, idempotency_key="file-context-replay")
        assert (await replay.wait()).result.wait_status is TaskStatus.SUCCEEDED
        replay_execution = await replay.execution("node")
        replay_ref = await runtime_engine.inputs.capture_input(replay_execution.execution_id, CaptureInputRequest(principal, "file-context-rerun"))
        value = await runtime_engine.inputs.read_agent(replay_ref, principal=principal)
        assert [item.data for item in value.prompt if isinstance(item, BinaryContent)] == [b"accepted file bytes"]


@pytest.mark.asyncio
async def test_model_memory_tool_reads_captured_content_after_production_changes(tmp_path: Path) -> None:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo
    from ._runtime_test_helpers import _UsageFunctionModel

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        last = messages[-1]
        if isinstance(last, ModelRequest):
            assert not any(isinstance(part, RetryPromptPart) for part in last.parts), repr(last)
        returned = next((part for part in last.parts if isinstance(part, ToolReturnPart)), None) if isinstance(last, ModelRequest) else None
        if returned is not None:
            return ModelResponse(parts=[TextPart("read:" + str(returned.content))])
        return ModelResponse(parts=[ToolCallPart("read_memory", {"file": "facts.md"}, tool_call_id="read-captured-memory")])

    class Binding(_TaskTestModelBinding):
        def materialize(self) -> _UsageFunctionModel:
            return _UsageFunctionModel(model)

    class Models(Binding):
        pass

    group = RuntimeFeatures("model-memory-context")
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('model-memory-context', storage=storage) as persistence_343, Runtime.attach(persistence=persistence_343, features=model_features(Models(), group)) as runtime, TaskEngine.open(persistence=persistence_343, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        memory = RuntimeMemoryStore(runtime._state.memory, object_store=runtime._state.object_store(PersistenceDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id="setup", memory_scope="production")
        original = await memory.write("memory/facts.md", "frozen memory value", expected_version=None)
        source = await runtime.agents.get().start("read memory", memory_scope="production", principal=principal)
        assert (await source.wait()).result.output == {"text": "read:frozen memory value"}
        ref = await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(principal, "model-memory-capture"))
        value = await runtime_engine.inputs.read_agent(ref, principal=principal)
        await memory.write("memory/facts.md", "changed production value", expected_version=original.version)
        replay = await runtime.agents.get().start(value.prompt, input_context=value.input_context, principal=principal)
        assert (await replay.wait()).result.output == {"text": "read:frozen memory value"}
        assert (await memory.read("memory/facts.md", max_chars=1000)).content == "changed production value"


@pytest.mark.asyncio
@pytest.mark.parametrize("rich_input", (False, True), ids=("text", "content"))
async def test_execution_capture_preserves_detached_accepted_provenance(
    tmp_path: Path, rich_input: bool,
) -> None:
    from pydantic_ai.messages import BinaryContent, TextContent
    from linktools.ai.model import MaterializedUserContent

    group = RuntimeFeatures("capture-provenance")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    principal = Principal("owner", "default")
    metadata = {"nested": {"labels": ["original"]}}
    prompt = (
        TextContent("hello", metadata=metadata),
        BinaryContent(b"original bytes", media_type="text/plain", identifier="evidence"),
    ) if rich_input else "hello"
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    expected_view = None
    async with Persistence.open('capture-provenance', storage=storage) as persistence_377, Runtime.attach(persistence=persistence_377, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_377, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        execution = await runtime.agents.get().start(prompt, principal=principal)
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        metadata["nested"]["labels"].append("author annotation")
        reference = await runtime_engine.inputs.capture_input(
            execution.execution_id, CaptureInputRequest(principal, "capture-provenance"),
        )
        captured = await runtime_engine.inputs.read_agent(reference, principal=principal)
        if rich_input:
            assert isinstance(captured.prompt, MaterializedUserContent)
            expected_view = dict(captured.prompt.view)
            text = next(item for item in captured.prompt if isinstance(item, TextContent) and item.content == "hello")
            assert text.metadata == {"nested": {"labels": ["original"]}}
            text.metadata["nested"]["labels"].append("consumer annotation")
            captured.prompt.view["attachments"].append({"display_annotation": "local"})
        else:
            assert captured.prompt == "hello"
        repeated = await runtime_engine.inputs.capture_input(
            execution.execution_id, CaptureInputRequest(principal, "capture-provenance"),
        )
        assert repeated == reference

    async with Persistence.open('capture-provenance', storage=storage) as persistence_399, Runtime.attach(persistence=persistence_399, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_399, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        restored = await runtime_engine.inputs.read_agent(reference, principal=principal)
        if rich_input:
            assert isinstance(restored.prompt, MaterializedUserContent)
            assert restored.prompt.view == expected_view
            text = next(item for item in restored.prompt if isinstance(item, TextContent) and item.content == "hello")
            assert text.metadata == {"nested": {"labels": ["original"]}}
            assert [item.data for item in restored.prompt if isinstance(item, BinaryContent)] == [b"original bytes"]
        else:
            assert restored.prompt == "hello"
