#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import pytest
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    LinkToolsUploadedFile,
    declared_uploaded_file_media_type,
)
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages
from pydantic_ai import RequestUsage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UploadedFile,
    UserPromptPart,
)


class _CustomReprUploadedFile(LinkToolsUploadedFile):
    def __repr__(self) -> str:
        return "uploaded-file"


def test_model_message_round_trip_is_canonical() -> None:
    messages = (ModelRequest(parts=[UserPromptPart(content="hello")]),)

    first = encode_model_messages(messages)
    second = encode_model_messages(messages)

    assert first == second
    assert first == canonical_json_bytes(json.loads(first.decode("utf-8")))
    assert decode_model_messages(first) == messages


def test_model_message_round_trip_preserves_usage_extensions() -> None:
    usage = RequestUsage(
        input_tokens=5,
        details={"reasoning_tokens": 3},
        future_tokens=42,
        label="original",
        zero_tokens=0,
    )
    messages = (ModelResponse(parts=[], usage=usage),)

    decoded = decode_model_messages(encode_model_messages(messages))

    assert len(decoded) == 1
    assert isinstance(decoded[0], ModelResponse)
    assert decoded[0].usage == usage
    assert decoded[0].usage.__dict__["future_tokens"] == 42
    assert decoded[0].usage.__dict__["label"] == "original"


def test_model_message_round_trip_does_not_infer_uploaded_file_media_type() -> None:
    messages = (
        ModelRequest(
            parts=[
                UserPromptPart(
                    [LinkToolsUploadedFile("report.png", "openai")]
                )
            ]
        ),
    )

    decoded = decode_model_messages(encode_model_messages(messages))

    uploaded = decoded[0].parts[0].content[0]
    assert isinstance(uploaded, UploadedFile)
    assert declared_uploaded_file_media_type(uploaded) is None


def test_model_message_round_trip_preserves_explicit_uploaded_file_media_type() -> None:
    messages = (
        ModelRequest(
            parts=[
                UserPromptPart(
                    [
                        LinkToolsUploadedFile(
                            "report.png",
                            "openai",
                            media_type="image/png",
                        )
                    ]
                )
            ]
        ),
    )

    decoded = decode_model_messages(encode_model_messages(messages))

    uploaded = decoded[0].parts[0].content[0]
    assert isinstance(uploaded, UploadedFile)
    assert declared_uploaded_file_media_type(uploaded) == "image/png"


def test_model_message_round_trip_preserves_media_type_with_custom_repr() -> None:
    messages = (
        ModelRequest(
            parts=[
                UserPromptPart(
                    [
                        _CustomReprUploadedFile(
                            "report.png",
                            "openai",
                            media_type="image/png",
                        )
                    ]
                )
            ]
        ),
    )

    decoded = decode_model_messages(encode_model_messages(messages))

    uploaded = decoded[0].parts[0].content[0]
    assert isinstance(uploaded, UploadedFile)
    assert declared_uploaded_file_media_type(uploaded) == "image/png"


def test_model_message_reader_accepts_pydantic_legacy_usage() -> None:
    raw = canonical_json_bytes(
        [
            {
                "parts": [],
                "usage": {
                    "requests": 0,
                    "request_tokens": None,
                    "response_tokens": None,
                    "total_tokens": None,
                    "details": None,
                },
                "kind": "response",
            }
        ]
    )

    decoded = decode_model_messages(raw)

    assert len(decoded) == 1
    assert isinstance(decoded[0], ModelResponse)
    assert decoded[0].usage == RequestUsage()


def test_model_message_reader_rejects_noncanonical_json() -> None:
    encoded = encode_model_messages(
        (ModelRequest(parts=[UserPromptPart(content="hello")]),)
    )

    with pytest.raises(AIError) as raised:
        decode_model_messages(b" " + encoded)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("raw", (b"{", b"[NaN]", b"[Infinity]", b"[-Infinity]"))
def test_model_message_reader_rejects_invalid_persistence(raw: bytes) -> None:
    with pytest.raises(AIError) as raised:
        decode_model_messages(raw)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
