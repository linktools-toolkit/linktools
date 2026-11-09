#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Parallel native tool references keep their call ownership and durable lifecycle."""

import asyncio
import base64
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, ExecutionCapability, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import ReferencedContent
from linktools.ai.model.messages import decode_user_content_item
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_native_referenced_tool_delivery import _IMAGE, _USER_IMAGE, _SOURCE, _reference


@pytest.mark.asyncio
async def test_parallel_native_tool_outputs_keep_ids_after_reverse_completion_and_reopen(tmp_path: Path) -> None:
    references = {"a": _reference(_IMAGE), "b": _reference(_USER_IMAGE)}
    for reference in references.values():
        reference.origin = None
    completed, loaded, requests = [], [], []
    consumers = []
    historical_read = False
    second_finished = asyncio.Event()

    async def read_a(context: AgentContext[None]) -> ToolReturn:
        await second_finished.wait()
        completed.append("a")
        return ToolReturn(return_value=[{"owner": "a"}, references["a"]])

    async def read_b(context: AgentContext[None]) -> ToolReturn:
        completed.append("b")
        second_finished.set()
        return ToolReturn(return_value=[{"owner": "b"}, references["b"]])

    async def finish(context: AgentContext[None]) -> str:
        return "acknowledged"

    async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        loaded.append(content)
        consumers.append(context.execution_id)
        if historical_read:
            assert content.origin.execution_id == execution_id
            assert context.execution_id != execution_id
            assert content.origin.principal == context.principal
        return _IMAGE if content.ref == references["a"].ref else _USER_IMAGE

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        if historical_read:
            values = [item for message in messages if isinstance(message, ModelRequest)
                      for part in message.parts for item in (part.content if isinstance(part.content, list) else [])]
            assert [item.data for item in values if isinstance(item, BinaryContent)] == [_IMAGE, _USER_IMAGE]
            return ModelResponse(parts=[TextPart("historical originals checked")])
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("read_a", {}, tool_call_id="a"),
                                        ToolCallPart("read_b", {}, tool_call_id="b")])
        native = [part for message in messages if isinstance(message, ModelRequest)
                  for part in message.parts if isinstance(part, ToolReturnPart) and part.tool_call_id in ("a", "b")]
        assert [part.tool_call_id for part in native] == ["a", "b"]
        assert [part.content[0]["owner"] for part in native] == ["a", "b"]
        if len(requests) == 2:
            assert all(isinstance(part.content[1], BinaryContent) for part in native)
            assert [part.content[1].data for part in native] == [_IMAGE, _USER_IMAGE]
            return ModelResponse(parts=[ToolCallPart("finish", {}, tool_call_id="finish")])
        assert len(requests) == 3
        assert all(not isinstance(part.content[1], (BinaryContent, ReferencedContent)) for part in native)
        assert "private-reconstruction-parameter" not in " ".join(part.model_response_str() for part in native)
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("native-tool-ownership")
    for tool in (read_a, read_b, finish):
        group.tool(tool, effect_policy="replay_safe")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    group.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    group.agent("default", allow_tools=("read_a", "read_b", "finish"), allow_skills=(), allow_subagents=())
    models = _Models(_UsageFunctionModel(respond))
    models.contract = {**models.contract, "input_mime_types": ["image/png"]}
    features = model_features(models, group)
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open('native-tool-ownership', storage=storage) as persistence_74, Runtime.attach(persistence=persistence_74, features=features) as runtime:
        execution = await runtime.agents.get().start("Read both independent files")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        execution_id = execution.execution_id
        principal = runtime.default_principal
    assert completed == ["b", "a"]
    assert [value.ref for value in loaded] == [references["a"].ref, references["b"].ref]
    assert [value.origin.tool_call_id for value in loaded] == ["a", "b"]
    assert all(value.origin.execution_id == execution_id and value.origin.principal == principal
               and value.origin.agent_run_seq == 1 and value.origin.model_request_seq == 1 for value in loaded)
    assert loaded[0].origin.agent_run_id == loaded[1].origin.agent_run_id
    async with Persistence.open('native-tool-ownership', storage=storage) as persistence_81, Runtime.attach(persistence=persistence_81, features=features) as runtime:
        result = await runtime.executions.result(execution_id, principal=principal)
        assert result.status is ExecutionStatus.SUCCEEDED
        history = await runtime.executions.history(execution_id, principal=principal, include_content=True)
        def descriptors(value):
            if isinstance(value, dict):
                if value.get("kind") == "referenced-content":
                    yield decode_user_content_item({
                        **{key: item for key, item in value.items() if key != "mime_type"},
                        "media_type": value["mime_type"],
                    })
                else:
                    for item in value.values():
                        yield from descriptors(item)
            elif isinstance(value, list):
                for item in value:
                    yield from descriptors(item)
        reopened = list(descriptors([item.content for item in history.items]))
        by_call = {value.origin.tool_call_id: value for value in reopened}
        assert set(by_call) == {"a", "b"}
        assert [by_call[name].origin for name in ("a", "b")] == [value.origin for value in loaded]
        historical_read = True
        consumer = await runtime.agents.get().start([by_call["a"], by_call["b"]])
        assert (await consumer.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert consumers[-2:] == [consumer.execution_id, consumer.execution_id]
    assert completed == ["b", "a"] and len(loaded) == 4 and len(requests) == 4
    stored = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    for body in (_IMAGE, _USER_IMAGE):
        assert body not in stored and base64.b64encode(body) not in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["direct", "list", "return_value", "content", "content_tuple"])
async def test_native_capability_tools_bind_original_producer_before_hydration(tmp_path: Path, shape: str) -> None:
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.toolsets import FunctionToolset

    from linktools.ai.capability import ReferencedInputResult

    reference = _reference(_IMAGE)
    reference.origin = None
    loaded = []
    calls = []

    async def read_original():
        calls.append(True)
        if shape == "direct":
            return reference
        if shape == "list":
            return [{"description": "original"}, reference]
        if shape == "return_value":
            return ToolReturn(return_value=[reference, reference])
        return ToolReturn(return_value="original", content=(reference, reference)
                          if shape == "content_tuple" else [reference, reference])

    class NativeTools(AbstractCapability[AgentContext[None]]):
        id = "native-original-tools"

        def get_toolset(self):
            return FunctionToolset([read_original])

    async def load(context, content):
        pytest.fail("The batch loader must be selected")

    async def batch(context, contents):
        loaded.append((context, contents))
        assert all(value.origin is not None for value in contents)
        return [ReferencedInputResult(value, _IMAGE) for value in contents]

    async def respond(messages, info):
        if not loaded:
            return ModelResponse(parts=[ToolCallPart("read_original", {}, tool_call_id="native-call")])
        assert any(isinstance(item, BinaryContent) for message in messages if isinstance(message, ModelRequest)
                   for part in message.parts for item in (part.content if isinstance(part.content, list) else [part.content]))
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("native-reference-origin")
    group.capability(NativeTools(), scope="agent")
    group.capability(TraceCapability(), scope="runtime")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(ReferencedInputsCapability(load, authorized_load_many=batch), id=_SOURCE.id,
                     revision=_SOURCE.revision, scope="agent")
    group.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    models = _Models(_UsageFunctionModel(respond))
    models.contract = {**models.contract, "input_mime_types": ["image/png"]}
    async with Persistence.open("native-reference-origin", storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True)) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(models, group),
    ) as runtime:
        execution = await runtime.agents.get().start("Read the admitted original")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        assert len(loaded) == 1 and len(calls) == 1
        context, contents = loaded[0]
        assert context.execution_id == execution.execution_id
        assert len(contents) == (2 if shape in {"return_value", "content", "content_tuple"} else 1)
        assert all(value.origin.execution_id == execution.execution_id
                   and value.origin.principal == context.principal
                   and value.origin.agent_run_seq == 1 and value.origin.model_request_seq == 1
                   and value.origin.tool_call_id == "native-call"
                   and value.origin.tool_name == "read_original" for value in contents)
        assert reference.origin is None
