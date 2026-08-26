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
    user_prompt_transport,
)
from linktools.ai.core import ExecutionLineageKind, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest, RuntimeObjectKeyFactory
from linktools.ai.runtime._execution import _request_digest
from linktools.ai.runtime._local import _recovery_prompt_payload, _recovery_prompt_text
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import InMemoryObjectStore, StoredPayload
from linktools.ai.temporal._request import (
    _execution_request_from_payload,
    put_execution_request,
    read_execution_request,
)


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


def _legacy_temporal_payload(user_prompt: str) -> dict[str, object]:
    return {
        "version": 1,
        "user_prompt": user_prompt,
        "principal": {
            "principal_id": "user",
            "tenant_id": "tenant",
            "kind": "service",
        },
        "idempotency_key": "legacy-temporal-prompt",
        "memory_scope": None,
        "planning": False,
        "thinking": False,
        "binding": _binding().to_payload(),
    }


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


def test_temporal_legacy_request_without_codec_defaults_to_text() -> None:
    value = "linktools.ai:user-content:v1:" + "0" * 64 + "\n{}"

    request, binding = _execution_request_from_payload(
        _legacy_temporal_payload(value)
    )

    assert binding == _binding()
    assert isinstance(request.user_prompt, UserPromptTransport)
    assert request.user_prompt.codec == "text"
    assert _restore_user_prompt(request.user_prompt) == value


def test_temporal_unknown_prompt_codec_is_unsupported() -> None:
    payload = _legacy_temporal_payload("payload")
    payload["user_prompt_codec"] = "future-user-content-v2"

    with pytest.raises(AIError) as raised:
        _execution_request_from_payload(payload)

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


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


def test_text_transport_keeps_historical_request_digest_identity() -> None:
    principal = Principal("user", "tenant", "service")
    value = '{"message":{"kind":"request"}}'
    historical = ExecutionRequest(value, principal, "same-key")
    explicit_text = ExecutionRequest(
        user_prompt_transport(value, "text"),
        principal,
        "same-key",
    )

    assert _execution_request_digest(explicit_text) == _execution_request_digest(
        historical
    )


def test_prompt_codec_participates_in_rich_request_digest_identity() -> None:
    principal = Principal("user", "tenant", "service")
    rich = _rich_prompt()
    rich_request = ExecutionRequest(rich, principal, "same-key")
    same_bytes_as_text = ExecutionRequest(str(rich), principal, "same-key")

    assert str(rich_request.user_prompt) == same_bytes_as_text.user_prompt
    assert _execution_request_digest(rich_request) != _execution_request_digest(
        same_bytes_as_text
    )
