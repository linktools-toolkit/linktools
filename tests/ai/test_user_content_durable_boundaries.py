#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import ExecutionLineageKind, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import _request_digest
from linktools.ai.runtime._input import (
    UserPromptTransport,
    _restore_user_prompt,
    prepare_user_prompt,
    user_prompt_transport,
)
from linktools.ai.runtime._local import _recovery_prompt_payload, _recovery_prompt_text
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
        binding_digest="a" * 64,
    )


def _rich_prompt() -> UserPromptTransport:
    return prepare_user_prompt(
        (
            "Inspect this attachment",
            BinaryContent(
                b"payload\n",
                media_type="text/plain",
                identifier="input.log",
            ),
        )
    )


def _request(prompt: str, *, codec: str) -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt=prompt,
        user_prompt_codec=codec,
        principal=Principal("user", "tenant", "service"),
        idempotency_key="same-key",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )


def _execution_request_digest(request: ExecutionRequest) -> str:
    return _request_digest(
        request,
        _binding().binding_digest,
        session_id=None,
        source_execution_id=None,
        base_execution_id=None,
        parent_execution_id=None,
        root_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
    )


def test_recovery_prompt_payload_preserves_rich_codec() -> None:
    prompt = _rich_prompt()

    payload = _recovery_prompt_payload(prompt)

    assert payload.encoding == "json"
    assert payload.decode() == {
        "codec": "pydantic-user-content-v1",
        "value": str(prompt),
    }
    restored_transport = _recovery_prompt_text(SimpleNamespace(user_prompt=payload))
    assert isinstance(restored_transport, UserPromptTransport)
    assert restored_transport.codec == prompt.codec
    restored = _restore_user_prompt(restored_transport)
    assert isinstance(restored, tuple)
    assert isinstance(restored[1], BinaryContent)


def test_recovery_text_payload_is_always_text() -> None:
    value = "linktools.ai:user-content:v1:" + "0" * 64 + "\n{}"
    payload = StoredPayload.inline_text(value)

    transport = _recovery_prompt_text(SimpleNamespace(user_prompt=payload))

    assert isinstance(transport, UserPromptTransport)
    assert transport.codec == "text"
    assert _restore_user_prompt(transport) == value


def test_recovery_unknown_prompt_codec_is_unsupported() -> None:
    payload = StoredPayload.inline_json(
        {"codec": "future-user-content-v2", "value": "payload"}
    )

    with pytest.raises(AIError) as raised:
        _recovery_prompt_text(SimpleNamespace(user_prompt=payload))

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_explicit_text_codec_keeps_request_digest_identity() -> None:
    value = '{"message":{"kind":"request"}}'
    first = _request(value, codec="text")
    second_transport = user_prompt_transport(value, "text")
    second = _request(second_transport, codec=second_transport.codec)

    assert _execution_request_digest(second) == _execution_request_digest(first)


def test_prompt_codec_participates_in_request_digest_identity() -> None:
    rich = _rich_prompt()
    rich_request = _request(rich, codec=rich.codec)
    same_bytes_as_text = _request(str(rich), codec="text")

    assert str(rich_request.user_prompt) == same_bytes_as_text.user_prompt
    assert _execution_request_digest(rich_request) != _execution_request_digest(
        same_bytes_as_text
    )
