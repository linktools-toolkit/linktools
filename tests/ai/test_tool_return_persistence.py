#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-return persistence preserves JSON and explicit media provenance."""

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from linktools.ai.runtime._message import decode_model_messages, encode_model_messages
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy


def _round_trip(content: object) -> object:
    messages = (
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
    decoded = decode_model_messages(encode_model_messages(messages))
    part = decoded[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    return part.content


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

    restored = _round_trip(value)

    assert restored == value
    assert isinstance(restored, dict)


def test_nested_tool_json_and_binary_content_keep_distinct_types() -> None:
    plain = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
        "business_count": 2,
    }
    binary = BinaryContent(
        data=b"real-binary",
        media_type="application/octet-stream",
    )

    restored = _round_trip({"plain": plain, "binary": binary})

    assert isinstance(restored, dict)
    assert restored["plain"] == plain
    assert isinstance(restored["binary"], BinaryContent)
    assert restored["binary"].data == binary.data
    assert restored["binary"].media_type == binary.media_type


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
