#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.agent import prepare_user_prompt
from linktools.ai.agent._input import _restore_user_prompt
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest


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

    assert first == second

    restored = _restore_user_prompt(first)
    assert isinstance(restored, tuple)
    assert restored[0] == "Inspect this attachment"
    assert isinstance(restored[1], BinaryContent)
    assert restored[1].data == b"level=error message=boom\n"
    assert restored[1].media_type == "text/plain"
    assert restored[1].identifier == "error.log"


def test_plain_text_keeps_identity_and_reserved_prefixes_are_escaped() -> None:
    plain = "normal prompt"
    assert prepare_user_prompt(plain) == plain
    assert _restore_user_prompt(plain) == plain

    reserved = "linktools.ai:user-content:v1:" + "0" * 64 + "\n{}"
    transport = prepare_user_prompt(reserved)
    assert transport != reserved
    assert _restore_user_prompt(transport) == reserved


def test_tampered_user_content_transport_fails_closed() -> None:
    transport = prepare_user_prompt(_attachment_prompt())
    tampered = transport[:-1] + ("x" if transport[-1] != "x" else "y")

    with pytest.raises(AIError) as raised:
        _restore_user_prompt(tampered)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_execution_request_preserves_rich_prompt_transport() -> None:
    transport = prepare_user_prompt(_attachment_prompt())
    request = ExecutionRequest(
        user_prompt=transport,
        principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="user-content-request",
    )

    assert request.user_prompt == transport
    restored = _restore_user_prompt(request.user_prompt)
    assert isinstance(restored, tuple)
    assert isinstance(restored[1], BinaryContent)
