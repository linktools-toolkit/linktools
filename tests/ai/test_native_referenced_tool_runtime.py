#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel native tool references keep their call ownership and durable lifecycle."""

import asyncio
import base64
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, ExecutionCapability, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_native_referenced_tool_delivery import _IMAGE, _USER_IMAGE, _SOURCE, _reference


@pytest.mark.asyncio
async def test_parallel_native_tool_outputs_keep_ids_after_reverse_completion_and_reopen(tmp_path: Path) -> None:
    references = {"a": _reference(_IMAGE), "b": _reference(_USER_IMAGE)}
    completed, loaded, requests = [], [], []
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
        return _IMAGE if content == references["a"] else _USER_IMAGE

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
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
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("native-tool-ownership", storage=storage, features=features) as runtime:
        execution = await runtime.agents.get().start("Read both independent files")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        execution_id = execution.execution_id
        principal = runtime.default_principal
    assert completed == ["b", "a"] and loaded == [references["a"], references["b"]]
    async with Runtime.open("native-tool-ownership", storage=storage, features=features) as runtime:
        result = await runtime.executions.result(execution_id, principal=principal)
        assert result.status is ExecutionStatus.SUCCEEDED
    assert completed == ["b", "a"] and len(loaded) == 2 and len(requests) == 3
    stored = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    for body in (_IMAGE, _USER_IMAGE):
        assert body not in stored and base64.b64encode(body) not in stored
