#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable historical inputs are executable without their source graph."""

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from linktools.ai.core import JsonValue, Principal, TaskStatus, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInputContext, CaptureInputRequest, CaptureGraphRequest, Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain, input_capture_key
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeResultRef
from .test_task_mixed_node_reliability import _TaskTestModels


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_invocation_fields", (False, True))
async def test_captured_task_dependencies_outlive_source_execution(tmp_path: Path, extra_invocation_fields: bool) -> None:
    async def source(context: TaskNodeContext[None]) -> JsonValue:
        return {"number": 17}

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        return {"value": (await context.read_dependency("alias"))["number"], "input": dict(context.input)}

    producer = Task("capture.source", source, effect_policy="none")
    consumer = Task("capture.target", target, effect_policy="none")
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("input-capture", models=_TaskTestModels(), storage=storage) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        engine = runtime.tasks.bind(producer, consumer)
        source_run = await engine.start(TaskGraph("source-graph", (
            TaskNode("source", task=producer),
            TaskNode("target", ("source",), task=consumer, input={"original": True},
                     input_refs={"alias": TaskNodeResultRef("source")}),
        )), idempotency_key="capture-source-graph-0001", principal=principal)
        result = await source_run.wait()
        assert result.status is TaskStatus.SUCCEEDED
        execution = await source_run.execution("target")
        if extra_invocation_fields:
            objects = storage.object_store(RuntimeDomain.TASK)
            key = input_capture_key("input-capture", principal.tenant_id, "invocation", execution.execution_id)
            stat = await objects.stat(key)
            payload = json.loads(b"".join([chunk async for chunk in objects.open(key)]))
            payload["dependency_results"] = {"$mapping": [["source", {
                "$dataclass": "task_dependency_result",
                "fields": {"result_digest": "0" * 64, "execution_id": "unused"},
            }]]}
            data = canonical_json_bytes(payload)

            async def chunks() -> AsyncIterator[bytes]:
                yield data

            assert await objects.delete_object(key, expected_digest=stat.digest)
            await objects.put(key, chunks(), expected_size=len(data), expected_digest=canonical_sha256(payload))
        capture = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(principal, "capture-task-input-0001", "clean"))
        rerun = await engine.start(TaskGraph("rerun-graph", (
            TaskNode("target", task=consumer, input_capture=capture),
        )), idempotency_key="capture-rerun-graph-0001", principal=principal)
        assert (await rerun.wait()).status is TaskStatus.SUCCEEDED
        assert await rerun.result("target") == {"value": 17, "input": {"original": True}}
        assert await source_run.result("target") == await rerun.result("target")


@pytest.mark.asyncio
async def test_capture_permission_denied_by_default() -> None:
    async with Runtime.open("capture-default-deny", models=_TaskTestModels(), storage=RuntimeStorage.in_memory()) as runtime:
        with pytest.raises(AIError) as raised:
            await runtime.executions.capture_input("unknown", CaptureInputRequest(runtime.default_principal, "capture-denied-0001", "clean"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_agent_capture_fixed_and_reproject_inputs(tmp_path: Path) -> None:
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.runtime import AgentTaskInput

    group = CapabilityGroup("capture")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    calls = []

    async def build(context: AgentTaskInputContext) -> str:
        calls.append(context.input["name"])
        return "Hello " + context.input["name"]

    async with Runtime.open("agent-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get(), build_input=build)
        engine = runtime.tasks.bind(task)
        run = await engine.start(TaskGraph("agent-source", (TaskNode("a", task=task,
            input=AgentTaskInput(parameters={"name": "Ada"})),)), principal=principal, idempotency_key="agent-capture-source-0001")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        execution = await run.execution("a")
        capture = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(principal, "agent-capture-input-0001", "clean"))
        assert (await runtime._input_captures.read_agent(capture, principal=principal)).prompt == "Hello Ada"
        fixed = await runtime._input_captures.task_input(capture, principal=principal)
        rerun = await engine.start(TaskGraph("agent-fixed", (TaskNode("a", task=task, input_capture=fixed),)), principal=principal, idempotency_key="agent-capture-fixed-0001")
        assert (await rerun.wait()).status is TaskStatus.SUCCEEDED
        assert calls == ["Ada"]
        reproject = await runtime._input_captures.task_input(capture, principal=principal, input_mode="reproject_input")
        rerun = await engine.start(TaskGraph("agent-reproject", (TaskNode("a", task=task, input_capture=reproject),)), principal=principal, idempotency_key="agent-capture-reproject-0001")
        assert (await rerun.wait()).status is TaskStatus.SUCCEEDED
        assert calls == ["Ada", "Ada"]
        restored = await runtime.tasks.from_agent_capture("capture.restored", capture, principal=principal)
        restored_run = await runtime.tasks.bind(restored).start(TaskGraph("agent-restored", (TaskNode("a", task=restored,
            input=AgentTaskInput("different case")),)), principal=principal, idempotency_key="agent-capture-restored-0001")
        assert (await restored_run.wait()).status is TaskStatus.SUCCEEDED
        restored_execution = await restored_run.execution("a")
        restored_capture = await runtime.executions.capture_input(restored_execution.execution_id, CaptureInputRequest(principal, "agent-capture-restored-input-0001", "clean"))
        assert (await runtime._input_captures.read_agent(restored_capture, principal=principal)).prompt == "different case"
        contextual = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(principal, "agent-captured-context-0001"))
        assert (await runtime._input_captures.read_agent(contextual, principal=principal)).input_context.model_messages() == ()


@pytest.mark.asyncio
async def test_graph_capture_keeps_frozen_results_across_restart(tmp_path: Path) -> None:
    async def source(context: TaskNodeContext[None]) -> JsonValue:
        return "frozen value"

    async def read(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("frozen")

    producer = Task("frozen.source", source, effect_policy="none")
    consumer = Task("frozen.consumer", read, effect_policy="none")
    async with Runtime.open("graph-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        engine = runtime.tasks.bind(producer, consumer)
        first = await engine.start(TaskGraph("external", (
            TaskNode("value", task=producer),
            TaskNode("read", ("value",), task=consumer, input_refs={"frozen": TaskNodeResultRef("value")}),
        )), principal=principal, idempotency_key="capture-external-0001")
        assert (await first.wait()).status is TaskStatus.SUCCEEDED
        source_execution = await first.execution("read")
        input_ref = await runtime.executions.capture_input(source_execution.execution_id,
            CaptureInputRequest(principal, "frozen-case-0001", "clean"))
        capture_service = runtime._input_captures
        second = await engine.start(TaskGraph("consumer", (TaskNode("read", task=consumer, input_capture=input_ref),)), principal=principal, idempotency_key="capture-consumer-0001")
        assert (await second.wait()).status is TaskStatus.SUCCEEDED
        graph_capture = await runtime.tasks.capture_graph("consumer", CaptureGraphRequest(principal, "capture-graph-0001"))
        template = await capture_service.read_graph(graph_capture, principal=principal)
        assert template.task_contracts
    async with Runtime.open("graph-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        template = await runtime._input_captures.read_graph(graph_capture, principal=principal)
        replay = await runtime.tasks.bind(consumer).start(TaskGraph("consumer-replay", template.nodes), principal=principal, idempotency_key="capture-consumer-replay-0001")
        assert (await replay.wait()).status is TaskStatus.SUCCEEDED
        assert await replay.result("read") == "frozen value"


@pytest.mark.asyncio
async def test_capture_preserves_failed_dependency_states(tmp_path: Path) -> None:
    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def collect(context: TaskNodeContext[None]) -> JsonValue:
        with pytest.raises(AIError) as raised:
            await context.read_dependency("failed")
        assert raised.value.code is ErrorCode.TASK_DEPENDENCY_FAILED
        return {name: value.status.value for name, value in context.dependency_states.items()}

    failure = Task("capture.failure", fail, effect_policy="none")
    collector = Task("capture.collect", collect, effect_policy="none")
    async with Runtime.open("failed-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        engine = runtime.tasks.bind(failure, collector)
        source = await engine.start(TaskGraph("failed-source", (
            TaskNode("failed", task=failure, failure_policy="isolate"),
            TaskNode("blocked", ("failed",), task=failure, failure_policy="isolate"),
            TaskNode("collect", ("failed", "blocked"), task=collector, dependency_policy="all_terminal"),
        )), principal=principal, idempotency_key="capture-failure-0001")
        await source.wait()
        execution = await source.execution("collect")
        reference = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(principal, "failed-input-0001", "clean"))
        replay = await engine.start(TaskGraph("failed-replay", (TaskNode("collect", task=collector, input_capture=reference),)),
            principal=principal, idempotency_key="failed-replay-0001")
        assert (await replay.wait()).status is TaskStatus.SUCCEEDED
        assert await replay.result("collect") == {"failed": "FAILED", "blocked": "BLOCKED"}


@pytest.mark.asyncio
async def test_subagent_capture_runs_as_independent_execution(tmp_path: Path) -> None:
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.runtime import AgentTaskInput, ExecutionRequest

    group = CapabilityGroup("capture-child")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    group.agent("child", model="default", system_prompt="Captured child instructions", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("subagent-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        parent = await runtime.agents.get().start("parent input", principal=principal)
        await parent.wait()
        child_agent = runtime.agents.get("child")
        compiled = runtime._compiled_agent(child_agent.id, child_agent.revision, child_agent.compiled)
        binding = runtime._compiler.bind_subagent(compiled)
        child = await runtime._execution_service.start_subagent(binding.binding_digest,
            ExecutionRequest(user_prompt="accepted child input", principal=principal, idempotency_key="child-input-0001",
                             memory_scope=None, mode="run", planning=False, thinking=False),
            parent_execution_id=parent.execution_id, root_execution_id=parent.execution_id,
            parent_invocation_id="source-invocation", binding_contract=binding.binding_contract)
        await runtime.executions.wait(child.execution_id, principal=principal)
        reference = await runtime.executions.capture_input(child.execution_id,
            CaptureInputRequest(principal, "capture-child-input-0001"))
        value = await runtime._input_captures.read_agent(reference, principal=principal)
        assert value.prompt == "accepted child input"
        assert value.source_invocation_id == "source-invocation"
        restored = await runtime.tasks.from_agent_capture("restored.child", reference, principal=principal)
        run = await runtime.tasks.bind(restored).start(TaskGraph("child-standalone", (
            TaskNode("child", task=restored, input=AgentTaskInput(value.prompt, input_context=value.input_context)),
        )), principal=principal, idempotency_key="child-standalone-0001")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        execution = await run.execution("child")
        fresh = await runtime.executions.inspect(execution.execution_id, principal=principal)
        source = await runtime.executions.inspect(child.execution_id, principal=principal)
        assert fresh.parent_execution_id is None
        assert fresh.agent_id == "child"
        assert source.parent_execution_id == parent.execution_id
        assert fresh.execution_id != child.execution_id


@pytest.mark.asyncio
async def test_snapshot_copies_capture_closure_and_rejects_missing_body(tmp_path: Path) -> None:
    import json
    from linktools.ai.core import canonical_json_bytes, canonical_sha256
    from linktools.ai.runtime.state import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore, ObjectRef

    async def producer(context: TaskNodeContext[None]) -> JsonValue:
        return "owned result"

    async def consumer(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("input")

    make = Task("snapshot.make", producer, effect_policy="none")
    use = Task("snapshot.use", consumer, effect_policy="none")
    root = tmp_path / "source"
    principal = Principal("capture", "default")
    async with Runtime.open("snapshot-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(root)) as runtime:
        engine = runtime.tasks.bind(make, use)
        first = await engine.start(TaskGraph("source", (
            TaskNode("make", task=make),
            TaskNode("use", ("make",), task=use, input_refs={"input": TaskNodeResultRef("make")}),
        )), principal=principal, idempotency_key="snapshot-source-0001")
        await first.wait()
        source = await first.execution("use")
        capture = await runtime.executions.capture_input(source.execution_id, CaptureInputRequest(principal, "snapshot-capture-0001", "clean"))
        second = await engine.start(TaskGraph("captured", (TaskNode("use", task=use, input_capture=capture),)),
            principal=principal, idempotency_key="snapshot-captured-0001")
        assert (await second.wait()).status is TaskStatus.SUCCEEDED
    archive = InMemoryObjectStore("capture-archive")
    storage = RuntimeStorage.filesystem(root)
    await storage.initialize(namespace="snapshot-capture", tenant_id="default", read_only=True)
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    try:
        snapshot = await storage.export_snapshot(object_store=archive, limits=limits)
    finally:
        await storage.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(snapshot, object_store=archive, root=restored_root, limits=limits)
    async with Runtime.open("snapshot-capture", models=_TaskTestModels(), storage=RuntimeStorage.from_root(restored_root)) as runtime:
        replay = await runtime.tasks.bind(use).start(TaskGraph("fresh", (TaskNode("use", task=use, input_capture=capture),)),
            principal=principal, idempotency_key="snapshot-fresh-0001")
        assert (await replay.wait()).status is TaskStatus.SUCCEEDED
        assert await replay.result("use") == "owned result"
    chunks = [chunk async for chunk in archive.open(snapshot.key)]
    manifest = json.loads(b"".join(chunks))
    manifest["objects"] = [item for item in manifest["objects"] if not item["source"]["key"].startswith("v1/input-capture/result/")]
    data = canonical_json_bytes(manifest)
    digest = canonical_sha256(manifest)

    async def damaged_chunks() -> AsyncIterator[bytes]:
        yield data

    await archive.put("damaged", damaged_chunks(), expected_size=len(data), expected_digest=digest)
    with pytest.raises(AIError) as raised:
        await RuntimeStorage.restore_snapshot(ObjectRef(archive.store_id, "damaged", digest, len(data)), object_store=archive,
                                              root=tmp_path / "damaged", limits=limits)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_task_capture_preserves_raw_input_and_projects_only_when_requested(tmp_path: Path) -> None:
    counts = {"source": 0, "candidate": 0}

    def original(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        counts["source"] += 1
        return {"value": value["value"] + 1}

    def candidate(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        counts["candidate"] += 1
        return {"value": value["value"] + 10}

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    source = Task("raw.source", echo, normalize=original, effect_policy="none")
    target = Task("raw.candidate", echo, normalize=candidate, effect_policy="none")
    async with Runtime.open("raw-capture", models=_TaskTestModels(), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        principal = Principal("capture", runtime.tenant_id)
        engine = runtime.tasks.bind(source, target)
        run = await engine.start(TaskGraph("raw-source", (TaskNode("node", task=source, input={"value": 1}),)),
            principal=principal, idempotency_key="raw-source-0001")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        execution = await run.execution("node")
        capture = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(principal, "raw-capture-0001", "clean"))
        contract = await runtime._input_captures.read_task(capture, principal=principal)
        assert dict(contract.original_input) == {"value": 1}
        assert dict(contract.input) == {"value": 2}
        fixed = await engine.start(TaskGraph("raw-fixed", (TaskNode("node", task=target, input_capture=capture),)),
            principal=principal, idempotency_key="raw-fixed-0001")
        assert (await fixed.wait()).status is TaskStatus.SUCCEEDED
        assert await fixed.result("node") == {"value": 2}
        assert counts["candidate"] == 0
        reproject = await runtime._input_captures.task_input(capture, principal=principal, input_mode="reproject_input")
        rerun = await engine.start(TaskGraph("raw-reproject", (TaskNode("node", task=target, input_capture=reproject),)),
            principal=principal, idempotency_key="raw-reproject-0001")
        assert (await rerun.wait()).status is TaskStatus.SUCCEEDED
        assert await rerun.result("node") == {"value": 11}
        assert counts["candidate"] == 1


@pytest.mark.asyncio
async def test_capture_reads_keep_owner_and_storage_tenant_authorization() -> None:
    async with Runtime.open("capture-owner", models=_TaskTestModels(), storage=RuntimeStorage.in_memory()) as runtime:
        owner = Principal("owner", runtime.tenant_id)
        other = Principal("other", runtime.tenant_id)
        capture = await runtime._input_captures.create_agent_input("private prompt", principal=owner, idempotency_key="owner-input-0001")
        with pytest.raises(AIError) as raised:
            await runtime._input_captures.read_agent(capture, principal=other)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as raised:
            await runtime._input_captures.create_agent_input("foreign", principal=Principal("owner", "different"), idempotency_key="foreign-input-0001")
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_graph_capture_authorizes_admitted_owner_before_copying_inputs() -> None:
    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    task = Task("private.graph", echo, effect_policy="none")
    async with Runtime.open("graph-capture-owner", models=_TaskTestModels(), storage=RuntimeStorage.in_memory()) as runtime:
        owner = Principal("owner", runtime.tenant_id)
        other = Principal("other", runtime.tenant_id)
        run = await runtime.tasks.bind(task).start(TaskGraph("private-source", (TaskNode("input", task=task, input={"secret": "owner input"}),)),
                                                  principal=owner, idempotency_key="private-source-0001")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        execution = await run.execution("input")
        with pytest.raises(AIError) as raised:
            await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(other, "steal-input-0001", "clean"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as raised:
            await runtime.tasks.capture_graph("private-source", CaptureGraphRequest(other, "steal-graph-0001", context_policy="clean"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        captured = await runtime.tasks.capture_graph("private-source", CaptureGraphRequest(owner, "own-graph-0001", context_policy="clean"))
        assert (await runtime._input_captures.read_graph(captured, principal=owner)).nodes[0].input["secret"] == "owner input"
