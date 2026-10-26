#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.task import Task
"""Task input identity follows the resolved owner, never business JSON shape."""
from ._runtime_test_helpers import model_features

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction, JsonValue, Principal, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import (
    DefaultTaskGraphService, TaskGraph, TaskGraphAdmission, TaskGraphHandle,
    TaskGraphLaunch, TaskGraphRequest, TaskGraphTemplate, TaskNode, TaskRef,
)


class _Authorization:
    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        assert principal.tenant_id == resource.tenant_id


class _Launcher:
    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        return TaskGraphHandle(launch.graph_id)


def _business_input(text: str) -> dict[str, JsonValue]:
    return {
        "kind": "agent-task-input",
        "prompt": {
            "kind": "stored-user-content-v1",
            "source_intent_digest": "a" * 64,
            "business_text": text,
        },
    }


def _request(body: Mapping[str, JsonValue]) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph("business", (TaskNode("worker", task=TaskRef("business.worker", 1), input=body),)),
        Principal("user", "tenant"), "submit",
    )


@pytest.mark.asyncio
async def test_generic_submission_rejects_changed_agent_shaped_business_input() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="identity", tenant_id="tenant")
    try:
        service = DefaultTaskGraphService(state.task, _Authorization(), _Launcher())
        first = _request(_business_input("first"))
        second = _request(_business_input("second"))
        prepared = await service.prepare_submission(first)
        assert await service.prepare_submission(first) == prepared
        with pytest.raises(AIError) as caught:
            await service.prepare_submission(second)
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    finally:
        await state.close()


@pytest.mark.parametrize("resolved", (False, True))
def test_business_input_projection_preserves_normalized_and_template_semantics(resolved: bool) -> None:
    def node(text: str) -> TaskNode:
        if resolved:
            return TaskNode.from_resolved(
                "worker", task=TaskRef("business.worker", 1),
                input=_business_input(text), original_input={"request": "unchanged"},
            )
        return TaskNode("worker", task=TaskRef("business.worker", 1), input=_business_input(text))

    first, second = node("first"), node("second")
    assert first.to_mapping()["input"] == first.input
    assert first.to_mapping() != second.to_mapping()
    assert TaskGraphTemplate((first,)).to_mapping() != TaskGraphTemplate((second,)).to_mapping()
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.codec import encode_domain, iter_runtime_object_refs
    assert tuple(iter_runtime_object_refs(encode_domain(first), default_domain=PersistenceDomain.TASK)) == ()
    admissions = [TaskGraphAdmission.from_request(TaskGraphRequest(
        TaskGraph("business", (current,)), Principal("user", "tenant"), "submit",
    )) for current in (first, second)]
    assert admissions[0].initial_request_digest != admissions[1].initial_request_digest


@pytest.mark.asyncio
async def test_function_normalization_cannot_hide_changed_business_input() -> None:

    from linktools.ai.task import Task, TaskNodeContext

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    body = _business_input("first")
    task = Task("business.normalized", echo, effect_policy="none", normalize=lambda _: body)
    async with Persistence.open('normalized', storage=PersistenceStorage.in_memory()) as persistence_102, TaskEngine.open(persistence=persistence_102) as engine:
        engine = engine.with_definitions(task)
        graph = TaskGraph("normalized", (TaskNode("worker", task=task, input={"request": "unchanged"}),))
        first = await engine.prepare_submission(graph, idempotency_key="submit")
        body = _business_input("second")
        second = await engine.describe_submission(graph, idempotency_key="submit")
        assert first.graph.nodes[0].original_input == second.graph.nodes[0].original_input
        assert first.ref != second.ref
        with pytest.raises(AIError) as caught:
            await engine.prepare_submission(graph, idempotency_key="submit")
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("function", "runner"))
async def test_graph_capture_keeps_agent_shaped_function_and_runner_inputs(kind: str) -> None:

    from linktools.ai.inputs import CaptureGraphRequest
    from linktools.ai.task import Task, TaskNodeContext
    from .test_task_mixed_node_reliability import _ExecutionBackedContractRunner
    from .test_evaluation_consumers import CONTEXT, PRINCIPAL

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    async with Persistence.open('capture-json', storage=PersistenceStorage.in_memory(), tenant_id='evaluation-consumers') as persistence_127, TaskEngine.open(persistence=persistence_127, context=CONTEXT) as engine:
        task = Task("business.worker", echo, effect_policy="none") if kind == "function" else Task.from_runner(
            "business.worker", _ExecutionBackedContractRunner(engine._execution_service),
            contract={"version": 1, "type": "runner", "effect_policy": "none",
                      "output_contract": {"kind": "json"}, "reconcile": False, "config": {}},
        )
        engine = engine.with_definitions(task)
        body = _business_input("business")
        run = await engine.start(TaskGraph("captured", (TaskNode("worker", task=task, input=body),)),
                                 principal=PRINCIPAL, idempotency_key="submit")
        await run.wait(timeout_seconds=10)
        principal = PRINCIPAL
        reference = await engine.inputs.capture_graph("captured", CaptureGraphRequest(principal, "capture"))
        template = await engine.inputs.read_graph(reference, principal=principal)
        assert template.nodes[0].input == body
        assert template.to_mapping()["nodes"][0]["input"] == body


@pytest.mark.asyncio
async def test_agent_submission_identity_survives_materialization_and_reopen(tmp_path: Path) -> None:
    from pydantic_ai.messages import BinaryContent
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.persistence.codec import decode_domain, encode_domain
    from linktools.ai.persistence.contracts import StoredUserInput
    from linktools.ai.storage import StoredPayload
    from .test_evaluation_consumers import FixtureModels

    features = RuntimeFeatures("identity")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    input = AgentTaskInput(("inspect", BinaryContent(b"accepted bytes", media_type="text/plain")))
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('agent-identity', storage=storage) as persistence_158, Runtime.open(persistence=persistence_158, features=model_features(FixtureModels(), features)) as runtime, TaskEngine.open(persistence=persistence_158) as engine:
        task = Task.from_agent("user.named.worker", runtime.agents.get())
        graph = TaskGraph("agent", (TaskNode("worker", task=task, input=input),))
        engine = engine.with_definitions(task)
        described = await engine.describe_submission(graph, idempotency_key="submit")
        prepared = await engine.prepare_submission(graph, idempotency_key="submit")
        assert described.ref == prepared.ref
        assert described.graph.nodes[0].input != prepared.graph.nodes[0].input
        assert described.graph.nodes[0].to_mapping() == prepared.graph.nodes[0].to_mapping()
        prepared.admission.validate_graph(prepared.graph)
        restored = decode_domain(encode_domain(prepared.graph), TaskGraph)
        prepared.admission.validate_graph(restored)
        assert restored == prepared.graph
        node = prepared.graph.nodes[0]
        assert decode_domain(encode_domain(node), type(node)) == node
        body = dict(node.input)
        prompt = dict(body["prompt"])
        stored = decode_domain(prompt["value"], StoredUserInput)
        assert stored.payload.ref is not None
        relocated = replace(stored, payload=StoredPayload.object(replace(stored.payload.ref, key="another-location")))
        prompt["value"] = encode_domain(relocated)
        body["prompt"] = prompt
        wire = encode_domain(node)
        wire["fields"]["input"] = encode_domain(body)
        relocated_node = decode_domain(wire, TaskNode)
        assert relocated_node.input != node.input
        assert relocated_node.to_mapping() == node.to_mapping()
        prepared.admission.validate_graph(TaskGraph("agent", (relocated_node,)))
        identity = prepared.ref

    async with Persistence.open('agent-identity', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_188, Runtime.open(persistence=persistence_188, features=model_features(FixtureModels(), features)) as runtime, TaskEngine.open(persistence=persistence_188) as engine:
        task = Task.from_agent("user.named.worker", runtime.agents.get())
        graph = TaskGraph("agent", (TaskNode("worker", task=task, input=input),))
        reopened = await engine.with_definitions(task).prepare_submission(graph, idempotency_key="submit")
        assert reopened.ref == identity
        assert reopened.graph == prepared.graph
        reopened.admission.validate_graph(reopened.graph)


@pytest.mark.parametrize("semantics", (None, "unknown-input-v1", "agent-task-input-v1", [], {}, True, 1))
def test_task_wire_requires_supported_and_valid_explicit_input_semantics(semantics: JsonValue) -> None:
    from linktools.ai.persistence.codec import decode_domain, encode_domain

    wire = encode_domain(TaskNode("worker", task=TaskRef("business.worker", 1), input=_business_input("text")))
    if semantics is None:
        wire["fields"].pop("input_semantics")
    else:
        wire["fields"]["input_semantics"] = semantics
    with pytest.raises(AIError) as caught:
        decode_domain(wire, TaskNode)
    assert caught.value.code is (ErrorCode.STORAGE_VERSION_UNSUPPORTED if semantics == "unknown-input-v1"
                                 else ErrorCode.STORAGE_INTEGRITY_ERROR)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("describe", "prepared"))
async def test_explicit_agent_node_cannot_override_a_function_input_owner(boundary: str) -> None:

    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.persistence.codec import decode_domain, encode_domain
    from linktools.ai.task import Task, TaskGraphSubmission, TaskNodeContext

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    task = Task("business.worker", echo, effect_policy="none")
    input = AgentTaskInput("hello", planning=False, thinking=False)
    wire = encode_domain(TaskNode("worker", task=task, input=input))
    wire["fields"]["input_semantics"] = "agent-task-input-v1"
    node = decode_domain(wire, TaskNode)
    with pytest.raises(AIError) as wrong_type:
        decode_domain(encode_domain(TaskNode("worker", task=task, input=input)), type(node))
    assert wrong_type.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    async with Persistence.open('owner-check', storage=PersistenceStorage.in_memory()) as persistence_232, TaskEngine.open(persistence=persistence_232) as engine:
        engine = engine.with_definitions(task)
        graph = TaskGraph("owner", (node,))
        principal = Principal("owner", "default")
        submission = TaskGraphSubmission(engine.namespace, TaskGraphAdmission.from_request(
            TaskGraphRequest(graph, principal, "submit")), graph)
        with pytest.raises(AIError) as caught:
            if boundary == "describe":
                await engine.describe_submission(graph, idempotency_key="submit")
            else:
                await engine.start_prepared(submission)
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert await engine.persistence.task.admissions.submission_status(submission.ref) is None


@pytest.mark.asyncio
async def test_agent_capture_identity_is_bound_to_the_retained_input_digest() -> None:
    from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.persistence.codec import decode_domain, encode_domain
    from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels

    features = RuntimeFeatures("capture-identity")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    models = FixtureModels()
    async with Persistence.open('capture-identity', storage=PersistenceStorage.in_memory(), tenant_id='evaluation-consumers') as persistence_256, Runtime.open(persistence=persistence_256, context=CONTEXT, features=model_features(models, features)) as runtime, TaskEngine.open(persistence=persistence_256) as engine:
        task = Task.from_agent("user.named.worker", runtime.agents.get())
        engine = engine.with_definitions(task)
        source = await engine.start(TaskGraph("source", (TaskNode("worker", task=task, input=AgentTaskInput("accepted")),)),
                                    principal=PRINCIPAL, idempotency_key="source")
        await source.wait(timeout_seconds=10)
        execution = await source.execution("worker")
        reference = await engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "capture"))
        capture = await engine.inputs.task_input(reference, principal=PRINCIPAL)
        graph = TaskGraph("captured", (TaskNode("worker", task=task, input_capture=capture),))
        described = await engine.describe_submission(graph, principal=PRINCIPAL, idempotency_key="captured")
        node = described.graph.nodes[0]
        assert node.input == {}
        assert node.to_mapping()["input_capture"]["digest"] == capture.digest
        prepared = await engine.prepare_submission(graph, principal=PRINCIPAL, idempotency_key="captured")
        assert prepared.ref == described.ref
        prepared.admission.validate_graph(decode_domain(encode_domain(prepared.graph), TaskGraph))

        wire = encode_domain(node)
        wire["fields"]["input_capture"] = encode_domain(replace(capture, digest="f" * 64))
        altered = decode_domain(wire, TaskNode)
        assert altered.input == node.input
        assert altered.input_capture.source_execution_id == capture.source_execution_id
        altered_graph = TaskGraph("captured", (altered,))
        altered_admission = TaskGraphAdmission.from_request(TaskGraphRequest(altered_graph, PRINCIPAL, "captured"))
        assert altered_admission.initial_request_digest != described.admission.initial_request_digest
        with pytest.raises(AIError) as caught:
            await engine.prepare_submission(TaskGraph("changed-capture", (altered,)),
                                            principal=PRINCIPAL, idempotency_key="changed-capture")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        replay = await engine.start(graph, principal=PRINCIPAL, idempotency_key="captured")
        await replay.wait(timeout_seconds=10)
        assert models.prompts == ["accepted", "accepted"]
