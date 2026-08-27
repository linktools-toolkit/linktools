#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, UploadedFile

from linktools.ai.core import Principal, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._input import (
    UserPromptTransport,
    _restore_user_prompt,
    prepare_user_prompt,
    user_prompt_transport,
)

_FIXTURE = Path(__file__).with_name("fixtures") / "user_prompt_transport_v1_golden.json"


def _attachment_prompt():
    return (
        "Inspect this attachment",
        BinaryContent(
            b"level=error message=boom\n",
            media_type="text/plain",
            identifier="error.log",
        ),
    )


def test_native_user_content_transport_is_deterministic_and_round_trips() -> None:
    first = prepare_user_prompt(_attachment_prompt())
    second = prepare_user_prompt(_attachment_prompt())

    assert isinstance(first, UserPromptTransport)
    assert first.codec == "pydantic-user-content-v1"
    assert first == second

    restored = _restore_user_prompt(first)
    assert isinstance(restored, tuple)
    assert restored[0] == "Inspect this attachment"
    assert isinstance(restored[1], BinaryContent)
    assert restored[1].data == b"level=error message=boom\n"
    assert restored[1].media_type == "text/plain"
    assert restored[1].identifier == "error.log"


def test_frozen_v1_prompt_fixtures_remain_readable() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    legacy_text = fixture["legacy_text"]
    text_transport = user_prompt_transport(
        legacy_text["value"],
        legacy_text["codec"],
    )
    assert _restore_user_prompt(text_transport) == legacy_text["value"]

    rich_v1 = fixture["rich_v1_minimal"]
    rich_transport = user_prompt_transport(
        canonical_json_bytes(rich_v1["value"]).decode("utf-8"),
        rich_v1["codec"],
    )
    assert _restore_user_prompt(rich_transport) == ("Inspect historical input",)


def test_rich_transport_survives_runtime_text_suffix() -> None:
    transport = prepare_user_prompt(_attachment_prompt())
    suffix = "\n\nUpstream task results (JSON, keyed by task id):\n{\"scan\":{\"ok\":true}}"

    combined = transport + suffix
    assert isinstance(combined, UserPromptTransport)
    assert combined.codec == transport.codec

    restored = _restore_user_prompt(combined)
    assert isinstance(restored, tuple)
    assert restored[0] == "Inspect this attachment"
    assert isinstance(restored[1], BinaryContent)
    assert restored[1].identifier == "error.log"
    assert restored[2] == suffix


def test_plain_text_is_never_interpreted_as_transport_protocol() -> None:
    plain = "normal prompt"
    transport = prepare_user_prompt(plain)
    assert isinstance(transport, UserPromptTransport)
    assert transport.codec == "text"
    assert str(transport) == plain
    assert _restore_user_prompt(transport) == plain

    magic_looking_text = "linktools.ai:user-content:v1:" + "0" * 64 + "\n{}"
    transport = prepare_user_prompt(magic_looking_text)
    assert transport.codec == "text"
    assert str(transport) == magic_looking_text
    assert _restore_user_prompt(transport) == magic_looking_text


def test_malformed_rich_transport_fails_closed() -> None:
    transport = prepare_user_prompt(_attachment_prompt())
    malformed = user_prompt_transport(
        str(transport).replace('"message"', '"unexpected"', 1),
        transport.codec,
    )

    with pytest.raises(AIError) as raised:
        _restore_user_prompt(malformed)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_unknown_prompt_codec_is_unsupported() -> None:
    with pytest.raises(AIError) as raised:
        user_prompt_transport("payload", "future-user-content-v2")

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_uploaded_file_is_rejected_until_durable_lifecycle_is_defined() -> None:
    uploaded = UploadedFile("file-123", "openai", media_type="text/plain")

    with pytest.raises(AIError) as raised:
        prepare_user_prompt(("Inspect this file", uploaded))

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details == {
        "field": "user_prompt",
        "reason": "uploaded_file_not_durable",
    }


def test_execution_request_preserves_rich_prompt_transport_and_codec() -> None:
    transport = prepare_user_prompt(_attachment_prompt())
    request = ExecutionRequest(
        user_prompt=transport,
        user_prompt_codec=transport.codec,
        principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="user-content-request",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )

    assert request.user_prompt is transport
    assert request.user_prompt_codec == "pydantic-user-content-v1"
    restored = _restore_user_prompt(request.user_prompt)
    assert isinstance(restored, tuple)
    assert isinstance(restored[1], BinaryContent)
