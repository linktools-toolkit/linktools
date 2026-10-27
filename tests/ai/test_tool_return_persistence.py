#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-return persistence preserves JSON and explicit media provenance."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent as NativeAgent
from pydantic_ai.messages import (
    BinaryContent, ImageUrl, ModelMessage, ModelRequest, ModelResponse, TextPart,
    ToolCallPart, ToolReturn, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import decode_model_messages, encode_model_messages
from linktools.ai.persistence._model_interaction import project_public_messages
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy

def _message(content: object) -> tuple[ModelRequest, ...]:
    return (
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "tool",
                    content,
                    tool_call_id="call-1",
                )
            ]
        ),
    )


def _round_trip(content: object) -> object:
    decoded = decode_model_messages(encode_model_messages(_message(content)))
    part = decoded[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    return part.content


def _encoded_part(content: object) -> dict[str, object]:
    value = json.loads(encode_model_messages(_message(content)).decode("utf-8"))
    return value[0]["parts"][0]


def _tool_bridge() -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        None,  # type: ignore[arg-type]
        InMemoryObjectStore(),
        namespace="tool-return-persistence",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id="run",
        binding_digest="a" * 64,
        owner="worker",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )


def test_arbitrary_tool_result_round_trips_as_json_snapshot() -> None:
    class Result(BaseModel):
        count: int

    assert _round_trip(Result(count=3)) == {"count": 3}
    assert _round_trip(b"abc") == "YWJj"


def test_media_shaped_tool_json_round_trips_as_plain_mapping() -> None:
    value = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
    }
    decoded = decode_model_messages(encode_model_messages(_message(value)))
    part = decoded[0].parts[0]

    assert isinstance(part, ToolReturnPart)
    assert part.content == value
    assert isinstance(part.content, dict)
    assert project_public_messages(decoded)[0]["parts"][0]["content"] == value
    encoded = _encoded_part(value)["content"]
    assert encoded["contract"] == "linktools.tool-return"
    assert encoded["version"] == 1
    assert encoded["value"]["type"] == "mapping"
    assert encoded["value"]["items"]["kind"] == {
        "type": "scalar",
        "value": "binary",
    }


def test_nested_tool_json_and_binary_content_keep_distinct_types() -> None:
    nested_plain = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "ZGVm",
    }
    binary = BinaryContent(
        data=b"real-binary",
        media_type="application/octet-stream",
    )
    value = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
        "business_count": 2,
        "nested": nested_plain,
        "attachment": binary,
    }

    restored = _round_trip(value)

    assert isinstance(restored, dict)
    assert restored["kind"] == "binary"
    assert restored["business_count"] == 2
    assert restored["nested"] == nested_plain
    assert isinstance(restored["attachment"], BinaryContent)
    assert restored["attachment"].data == binary.data
    assert restored["attachment"].media_type == binary.media_type


def test_tool_return_wire_is_independent_of_mapping_insertion_order() -> None:
    first = {"z": {"value": 2}, "a": {"value": 1}}
    second = {"a": first["a"], "z": first["z"]}

    assert _encoded_part(first)["content"] == _encoded_part(second)["content"]


@pytest.mark.asyncio
async def test_durable_tool_result_recovery_preserves_media_shaped_json() -> None:
    value = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
        "business_count": 2,
    }
    bridge = _tool_bridge()
    decision = ToolOperationDecision("operation", "worker", 1, True)

    payload = await bridge._result_payload(decision, value)
    restored = await bridge._decode_result(  # type: ignore[arg-type]
        SimpleNamespace(
            tool_operation_id=decision.operation_id,
            result_payload=payload,
        )
    )

    assert restored == value
    assert isinstance(restored, dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    None,
    "Read the attached evidence.",
    ["Read the attached evidence."],
    [BinaryContent(data=b"synthetic-image", media_type="image/png")],
    [ImageUrl(url="https://example.com/evidence.png")],
])
async def test_native_tool_return_preserves_content_and_metadata_on_operation_replay(content: object) -> None:
    value = ToolReturn(return_value={"status": "read"}, content=content,
                       metadata={"source": "evidence"}, tools=["existing_tool"])
    bridge = _tool_bridge()
    payload = await bridge._result_payload(ToolOperationDecision("operation", "worker", 1, True), value)
    restored = await bridge._decode_result(SimpleNamespace(
        result_payload=payload, tool_operation_id="operation",
    ))
    assert isinstance(restored, ToolReturn)
    assert restored.return_value == value.return_value
    assert restored.content == value.content
    assert restored.metadata == value.metadata
    assert restored.tools == value.tools


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    "Explicit follow-up instructions.",
    [BinaryContent(data=b"synthetic-image", media_type="image/png")],
    [ImageUrl(url="https://example.com/evidence.png")],
])
async def test_replayed_tool_content_reaches_native_model_as_user_content(content: object) -> None:
    value = ToolReturn(return_value="receipt", content=content)
    bridge = _tool_bridge()
    payload = await bridge._result_payload(ToolOperationDecision("operation", "worker", 1, True), value)
    restored = await bridge._decode_result(SimpleNamespace(
        result_payload=payload, tool_operation_id="operation",
    ))
    requests = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("evidence", {}, tool_call_id="read-evidence")])
        parts = messages[-1].parts
        assert any(isinstance(part, ToolReturnPart) and part.content == "receipt" for part in parts)
        user_content = [part.content for part in parts if isinstance(part, UserPromptPart)]
        assert user_content == [content]
        return ModelResponse(parts=[TextPart("received")])

    async def evidence() -> ToolReturn:
        return restored

    agent = NativeAgent(FunctionModel(respond), tools=[evidence])
    result = await agent.run("Read the evidence")
    assert result.output == "received"
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [False, True])
async def test_tool_return_replay_preserves_portable_metadata_snapshot(with_image: bool) -> None:
    class EvidenceMetadata(BaseModel):
        captured_at: datetime
        labels: set[str]

    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    value = ToolReturn(
        return_value="receipt",
        content=[BinaryContent(data=b"synthetic-image", media_type="image/png")] if with_image else None,
        metadata={
            "captured_at": timestamp,
            "evidence": EvidenceMetadata(captured_at=timestamp, labels={"b", "a"}),
        },
    )
    bridge = _tool_bridge()
    payload = await bridge._result_payload(ToolOperationDecision("operation", "worker", 1, True), value)
    restored = await bridge._decode_result(SimpleNamespace(
        result_payload=payload, tool_operation_id="operation",
    ))

    assert isinstance(restored, ToolReturn)
    assert restored.return_value == value.return_value
    assert restored.content == value.content
    assert restored.metadata == {
        "captured_at": "2026-01-02T03:04:05Z",
        "evidence": {"captured_at": "2026-01-02T03:04:05Z", "labels": ["a", "b"]},
    }
