#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Declared output schemas resolve once for root, child, and Task executions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.agent import AgentBindingContract, AgentCompiler, AssistantTextOutput, OutputBinding, bind_output
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AssetCapability
from linktools.ai.core import ExecutionStatus, JsonValue, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import create_model
from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.spec import AgentSpec, AgentSpecAdapter, AgentSpecCodec
from linktools.ai.storage import StorageOverlay
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel, is_tool_retry, model_features


class _Count(BaseModel):
    count: int


class _Label(BaseModel):
    label: str


def _compiler(spec: AgentSpec) -> AgentCompiler:
    return AgentCompiler(
        models={"default": create_model(protocol="openai", model="gpt-test")},
        candidates=(), agents={spec.id: spec},
    )


@pytest.mark.parametrize("fields", ({}, {"output_schema": None}))
def test_undeclared_output_remains_text(fields: dict[str, object]) -> None:
    codec = AgentSpecCodec()
    spec = codec.from_payload({"version": 1, "id": "agent", **fields})
    assert spec.output_schema is None
    assert codec.decode(codec.encode(spec)) == spec
    compiler = _compiler(spec)
    assert compiler.bind(compiler.compile(spec)).output_binding == bind_output()


def test_declared_schema_is_immutable_and_round_trips() -> None:
    schema = _Count.model_json_schema()
    spec = AgentSpec("agent", output_schema=schema)
    expected = _Count.model_json_schema()
    schema["properties"]["count"]["type"] = "string"
    spec.output_schema["properties"]["count"]["type"] = "boolean"
    assert spec.output_schema == expected
    codec = AgentSpecCodec()
    assert codec.decode(codec.encode(spec)) == spec
    assert codec.to_contract_payload(spec)["output_schema"] == expected
    compiler = _compiler(spec)
    expected_binding = OutputBinding.create("structured", expected)
    assert compiler.bind(compiler.compile(spec), output=None).output_binding == expected_binding
    assert compiler.bind_subagent(compiler.compile(spec)).output_binding == expected_binding


@pytest.mark.parametrize("schema", ([], False, "object", {"type": object()}))
def test_malformed_output_declarations_fail_closed(schema: object) -> None:
    with pytest.raises(AIError) as failed:
        AgentSpecCodec().from_payload({"version": 1, "id": "agent", "output_schema": schema})
    assert failed.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("schema", ({"type": "unknown"}, {"$ref": "https://invalid.example/schema"}))
def test_schema_binding_uses_existing_supported_schema_validation(schema: dict[str, JsonValue]) -> None:
    spec = AgentSpec("agent", output_schema=schema)
    compiler = _compiler(spec)
    with pytest.raises(AIError) as failed:
        compiler.bind(compiler.compile(spec))
    assert failed.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("schema", ({}, {"type": "string"}, {"type": "array", "items": {"type": "string"}},
                                    {"type": ["object", "null"], "properties": {}}))
def test_default_schema_must_be_executable_at_binding(schema: dict[str, JsonValue]) -> None:
    spec = AgentSpec("agent", output_schema=schema)
    compiler = _compiler(spec)
    compiled = compiler.compile(spec)
    for bind in (compiler.bind, compiler.bind_subagent):
        with pytest.raises(AIError) as failed:
            bind(compiled)
        assert failed.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    assert compiler.bind(compiled, output=_Label).output_binding == bind_output(_Label)


@pytest.mark.parametrize("output", (_Label, AssistantTextOutput, bind_output(_Label), bind_output()))
def test_explicit_outputs_override_agent_schema(output: type[BaseModel] | OutputBinding) -> None:
    spec = AgentSpec("agent", output_schema=_Count.model_json_schema())
    compiler = _compiler(spec)
    expected = output if isinstance(output, OutputBinding) else bind_output(output)
    assert compiler.bind(compiler.compile(spec), output=output).output_binding == expected
    assert compiler.bind_subagent(compiler.compile(spec), output=output).output_binding == expected


@pytest.mark.parametrize("output", (None, _Label, AssistantTextOutput))
def test_restore_uses_frozen_output_not_new_declaration(output: type[BaseModel] | None) -> None:
    spec = AgentSpec("agent", output_schema=_Count.model_json_schema(), output_retries=2)
    compiler = _compiler(spec)
    saved = compiler.bind(compiler.compile(spec), output=output)
    contract = AgentBindingContract.from_payload(saved.binding_contract.to_payload())
    changed = replace(spec, output_schema={"type": "object", "properties": {"changed": {"type": "boolean"}}})
    restored = _compiler(changed).restore(contract)
    assert restored.output_binding == saved.output_binding
    assert restored.binding_contract.to_payload() == saved.binding_contract.to_payload()
    assert restored.compiled_agent.spec.output_retries == 2


@pytest.mark.asyncio
async def test_asset_and_features_declarations_share_output_semantics() -> None:
    schema = _Count.model_json_schema()
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("agent", "reviewer/AGENT.md"), (
            "---\noutput_schema: " + json.dumps(schema) + "\noutput_retries: 2\n---\nreview\n"
        ).encode())
        asset_features = RuntimeFeatures("assets")
        asset_features.capability(AssetCapability(store), scope="runtime")
        captured = await asset_features.capture()
        asset_spec = next(item.value for item in captured.contributions if item.kind == "agent")
        features = RuntimeFeatures("declared")
        explicit = features.agent("reviewer", system_prompt="review\n", output_schema=schema, output_retries=2)
        assert asset_spec == explicit
        assert AgentSpecAdapter().from_mapping(
            {"output_schema": None}, logical_id="agent", defaults={"output_schema": schema}
        ).output_schema is None
        compiler = _compiler(asset_spec)
        binding = compiler.bind(compiler.compile(asset_spec))
        assert binding.output_binding == OutputBinding.create("structured", schema)
        assert binding.binding_contract.agent_spec.output_retries == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_root_and_task_defaults_use_native_schema_correction(tmp_path: Path) -> None:
    feedback: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not info.output_tools:
            return ModelResponse(parts=[TextPart("plain")])
        output = info.output_tools[0]
        if "label" in output.parameters_json_schema["properties"]:
            value = {"label": "explicit"}
        else:
            retries = [part for message in messages for part in message.parts if is_tool_retry(part)]
            if retries:
                part = retries[-1]
                feedback.append(part.model_response() if isinstance(part, RetryPromptPart) else part.model_response_str())
            value = {"count": 3 if retries else "invalid-count"}
        return ModelResponse(parts=[ToolCallPart(output.name, value)])

    features = RuntimeFeatures("default-output")
    features.agent("default", output_schema=_Count.model_json_schema(), output_retries=1,
                   allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('default-output', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_169, Runtime.open(persistence=persistence_169, features=model_features(_Models(_UsageFunctionModel(respond)), features)) as runtime, TaskEngine.open(persistence=persistence_169, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        agent = runtime.agents.get()
        for options, expected in (
            ({}, {"count": 3}), ({"output": None}, {"count": 3}),
            ({"output": _Label}, {"label": "explicit"}),
            ({"output": AssistantTextOutput}, {"text": "plain"}),
        ):
            result = (await agent.run("review", timeout_seconds=10, **options)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == expected
        task = Task.from_agent("review-task", agent)
        run = await runtime_engine.with_definitions(task).start(TaskGraph("default-output", (
            TaskNode("inherited", task=task, input=AgentTaskInput("review")),
            TaskNode("explicit", task=task, input=AgentTaskInput("review"), output_type=_Label),
            TaskNode("text", task=task, input=AgentTaskInput("review"), output_type=AssistantTextOutput),
        )), idempotency_key="default-output")
        assert (await run.wait(timeout_seconds=30)).result.status is TaskStatus.SUCCEEDED
        assert await run.result("inherited") == {"count": 3}
        assert await run.result("explicit") == {"label": "explicit"}
        assert await run.result("text") == {"text": "plain"}
        assert len(feedback) == 3
        assert all("count" in item and "expected integer" in item for item in feedback)
        assert all("invalid-count" not in item for item in feedback)


@pytest.mark.asyncio
async def test_delegated_child_returns_declared_structured_output() -> None:
    delegated: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.output_tools:
            output = info.output_tools[0]
            assert output.parameters_json_schema["properties"]["count"]["type"] == "integer"
            return ModelResponse(parts=[ToolCallPart(output.name, {"count": 3})])
        returned = [part for message in messages for part in message.parts
                    if isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"]
        if returned:
            delegated.append(returned[-1].content)
            return ModelResponse(parts=[TextPart("reviewed")])
        return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "review"})])

    features = RuntimeFeatures("child-output")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=("child",))
    features.agent("child", output_schema=_Count.model_json_schema(),
                   allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('child-output', storage=PersistenceStorage.in_memory()) as persistence_217, Runtime.open(persistence=persistence_217, features=model_features(_Models(_UsageFunctionModel(respond)), features)) as runtime:
        result = (await runtime.agents.get().run("delegate review", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "reviewed"}
        assert len(delegated) == 1
        assert delegated[0]["output"] == {"count": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("output", (None, AssistantTextOutput))
async def test_captured_default_or_explicit_text_survives_runtime_reopen(
    tmp_path: Path, output: type[BaseModel] | None,
) -> None:
    schemas: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        if not info.output_tools:
            schemas.append(None)
            return ModelResponse(parts=[TextPart("plain")])
        tool = info.output_tools[0]
        schemas.append(tool.parameters_json_schema)
        value = {"count": 3} if "count" in tool.parameters_json_schema["properties"] else {"label": "new"}
        return ModelResponse(parts=[ToolCallPart(tool.name, value)])

    def features(revision: int, schema: dict[str, JsonValue]) -> RuntimeFeatures:
        group = RuntimeFeatures("frozen-output")
        group.agent("default", revision=revision, output_schema=schema,
                    allow_tools=(), allow_skills=(), allow_subagents=())
        return model_features(_Models(_UsageFunctionModel(respond)), group)

    async with Persistence.open('frozen-output', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_251, Runtime.open(persistence=persistence_251, features=features(1, _Count.model_json_schema())) as runtime, TaskEngine.open(persistence=persistence_251, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("capture", runtime.tenant_id)
        source = await runtime.agents.get().start("review", output=output, principal=principal)
        original = (await source.wait(timeout_seconds=10)).result
        assert original.status is ExecutionStatus.SUCCEEDED
        capture = await runtime_engine.inputs.capture_input(
            source.execution_id, CaptureInputRequest(principal, "capture-default-output", "clean")
        )

    async with Persistence.open('frozen-output', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_263, Runtime.open(persistence=persistence_263, features=features(2, _Label.model_json_schema())) as runtime, TaskEngine.open(persistence=persistence_263, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        fresh = (await runtime.agents.get().run("review", timeout_seconds=10)).result
        assert fresh.output == {"label": "new"}
        restored_capture = await runtime_engine.inputs.read_agent(capture, principal=principal)
        restored_capture.require_importable()
        restored = Task.from_agent("restored-output", runtime.agents.restore(restored_capture.binding))
        run = await runtime_engine.with_definitions(restored).start(TaskGraph("restored-output", (
            TaskNode("review", task=restored, input=AgentTaskInput("review again")),
        )), principal=principal, idempotency_key="restored-output")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert await run.result("review") == original.output
        assert schemas[0] == schemas[2]
        assert schemas[0] != schemas[1]
