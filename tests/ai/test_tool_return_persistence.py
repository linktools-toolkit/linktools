#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-return persistence preserves JSON and explicit media provenance."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages
from linktools.ai.runtime._model_interaction import project_public_messages
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
        step_run_id="run",
        binding_digest="a" * 64,
        owner="worker",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )


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


