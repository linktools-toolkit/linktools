#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._input import (
    UserPromptTransport,
    _restore_user_prompt,
    prepare_user_prompt,
)
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest, RuntimeObjectKeyFactory
from linktools.ai.runtime._local import _recovery_prompt_payload, _recovery_prompt_text
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import InMemoryObjectStore, StoredPayload
from linktools.ai.temporal._request import put_execution_request, read_execution_request


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        agent_digest="c" * 64,
        output_schema_id="test-output",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
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


@pytest.mark.asyncio
async def test_temporal_execution_request_preserves_rich_prompt_codec() -> None:
    store = InMemoryObjectStore("requests")
    key_factory = RuntimeObjectKeyFactory("prompt-codec-test")
    prompt = _rich_prompt()
    request = ExecutionRequest(
        prompt,
        Principal("user", "tenant", "service"),
        "temporal-prompt-codec",
    )

    request_ref = await put_execution_request(
        store,
        key_factory,
        request,
        binding=_binding(),
    )
    restored_request = await read_execution_request(
        store,
        key_factory,
        tenant_id="tenant",
        request_ref=request_ref,
    )

    assert isinstance(restored_request.user_prompt, UserPromptTransport)
    assert restored_request.user_prompt.codec == "pydantic-user-content-v1"
    restored = _restore_user_prompt(restored_request.user_prompt)
    assert isinstance(restored, tuple)
    assert restored[0] == "Inspect this attachment"
    assert isinstance(restored[1], BinaryContent)
    assert restored[1].identifier == "input.log"


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


def test_recovery_legacy_text_payload_is_always_text() -> None:
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
