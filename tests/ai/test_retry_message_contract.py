#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry messages preserve SDK meaning in the owned durable message contract."""

import json
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, NativeToolReturnPart, RetryPromptPart, ToolReturnPart

try:
    from pydantic_ai.messages import RetryFeedbackPart, RetryInfo
except ImportError:
    RetryFeedbackPart = RetryInfo = None

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import decode_model_messages, encode_model_messages, freeze_model_messages
from linktools.ai.persistence._views import project_session_history_message

_STAMP = datetime(2026, 10, 23, tzinfo=timezone.utc)
_RETRY = {"cause": "model_retry", "attempt": 1, "max_retries": 2}
_ERRORS = [{"type": "int_parsing", "loc": ("count",), "input": "bad", "msg": "expected integer"}]
_NEW_SDK = pytest.mark.skipif(RetryInfo is None, reason="SDK does not provide retry feedback types")


def _return_part(*, native: bool = False, outcome: str = "success") -> ToolReturnPart | NativeToolReturnPart:
    part_type = NativeToolReturnPart if native else ToolReturnPart
    return part_type("lookup", "done", tool_call_id="call-1", timestamp=_STAMP, outcome=outcome)


def _wire(part: dict[str, object], *, native: bool = False) -> bytes:
    message = ModelResponse(parts=[], timestamp=_STAMP) if native else ModelRequest(parts=[], timestamp=_STAMP)
    value = json.loads(encode_model_messages((message,)))
    value[0]["parts"] = [part]
    return canonical_json_bytes(value)


def _tool_wire(*, native: bool = False) -> dict[str, object]:
    return {
        "part_kind": "builtin-tool-return" if native else "tool-return",
        "tool_name": "lookup", "tool_call_id": "call-1", "tool_kind": None,
        "content": {"contract": "linktools.tool-return", "version": 1,
                    "value": {"type": "scalar", "value": "done"}},
        "timestamp": "2026-10-23T00:00:00Z", "outcome": "success", "metadata": None,
        **({"provider_name": None, "provider_details": None} if native else {}),
    }


def _feedback_wire() -> dict[str, object]:
    return {"part_kind": "retry-feedback", "content": "fix output", "retry": dict(_RETRY),
            "timestamp": "2026-10-23T00:00:00Z"}


@pytest.mark.parametrize("native", (False, True))
def test_existing_tool_return_bytes_are_unchanged(native: bool) -> None:
    part = _return_part(native=native)
    message = ModelResponse(parts=[part], timestamp=_STAMP) if native else ModelRequest(parts=[part], timestamp=_STAMP)
    encoded = encode_model_messages((message,))
    assert encoded == _wire(_tool_wire(native=native), native=native)
    assert decode_model_messages(encoded) == (message,)
    assert "retry" not in json.loads(encoded)[0]["parts"][0]


@pytest.mark.parametrize("content", ("fix output", _ERRORS))
def test_legacy_retry_prompt_remains_readable_without_reinterpretation(content: object) -> None:
    part = RetryPromptPart(content, tool_name="lookup", tool_call_id="call-1", timestamp=_STAMP)
    message = ModelRequest(parts=[part], timestamp=_STAMP)
    restored, = decode_model_messages(encode_model_messages((message,)))
    assert type(restored.parts[0]) is RetryPromptPart
    assert restored == message


@_NEW_SDK
@pytest.mark.parametrize("content,cause", [("fix output", "model_retry"), (_ERRORS, "validation_error"),
                                         ("Please return text.", "no_output")])
@pytest.mark.parametrize("attempt,max_retries", [(None, None), (1, 2), (3, 1), (None, 2)])
def test_retry_feedback_round_trip_preserves_content_and_budget(
    content: object, cause: str, attempt: int | None, max_retries: int | None,
) -> None:
    part = RetryFeedbackPart(content, retry=RetryInfo(cause=cause, attempt=attempt, max_retries=max_retries),
                             timestamp=_STAMP)
    message = ModelRequest(parts=[part], timestamp=_STAMP)
    encoded = encode_model_messages((message,))
    payload = json.loads(encoded)[0]
    assert payload["version"] == 1
    assert payload["parts"][0]["retry"] == {"cause": cause, "attempt": attempt, "max_retries": max_retries}
    assert decode_model_messages(encoded) == (message,)


@_NEW_SDK
@pytest.mark.parametrize("cause", ("model_retry", "validation_error", "unknown_tool", "unavailable_tool"))
def test_retried_tool_return_retains_retry_information(cause: str) -> None:
    part = ToolReturnPart("lookup", "try again", tool_call_id="call-1", timestamp=_STAMP, outcome="retried",
                          metadata={"caller": "test"}, retry=RetryInfo(cause=cause, attempt=1, max_retries=2))
    message = ModelRequest(parts=[part], timestamp=_STAMP)
    assert decode_model_messages(encode_model_messages((message,))) == (message,)


@_NEW_SDK
def test_output_feedback_keeps_its_payload_without_inventing_tool_coordinates() -> None:
    feedback = RetryFeedbackPart("fix output", retry=RetryInfo(**_RETRY), timestamp=_STAMP)
    frozen, = freeze_model_messages((ModelRequest(parts=[feedback], timestamp=_STAMP),))
    item, = project_session_history_message(frozen)
    assert item.item_kind == "retry-feedback"
    assert item.tool_name is None and item.tool_call_id is None
    assert item.content["content"] == "fix output"
    assert item.content["retry"] == _RETRY
    assert frozen.parts[0] == feedback and frozen.parts[0] is not feedback


@pytest.mark.parametrize("native", (False, True))
def test_retried_outcome_requires_sdk_support_even_without_retry_information(native: bool) -> None:
    part = _tool_wire(native=native)
    part["outcome"] = "retried"
    for nullable in ((False,) if native else (False, True)):
        if nullable:
            part["retry"] = None
        encoded = _wire(part, native=native)
        if RetryInfo is None:
            with pytest.raises(AIError) as caught:
                decode_model_messages(encoded)
            assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
            with pytest.raises(AIError) as caught:
                encode_model_messages((ModelResponse(parts=[_return_part(native=True, outcome="retried")])
                                       if native else ModelRequest(parts=[_return_part(outcome="retried")]),))
            assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
        else:
            restored, = decode_model_messages(encoded)
            assert restored.parts[0].outcome == "retried"
            assert "retry" not in json.loads(encode_model_messages((restored,)))[0]["parts"][0]


@pytest.mark.parametrize("feedback", (False, True))
def test_new_retry_payload_requires_supported_sdk(feedback: bool) -> None:
    part = _feedback_wire() if feedback else {**_tool_wire(), "retry": dict(_RETRY)}
    if RetryInfo is None:
        with pytest.raises(AIError) as caught:
            decode_model_messages(_wire(part))
        assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    else:
        restored, = decode_model_messages(_wire(part))
        assert restored.parts[0].retry.cause == "model_retry"


@_NEW_SDK
@pytest.mark.parametrize("feedback", (False, True))
@pytest.mark.parametrize("field,value", [("cause", "unknown"), ("cause", 1), ("attempt", True),
                                        ("attempt", "1"), ("max_retries", False), ("max_retries", 1.5)])
def test_retry_reader_rejects_invalid_fields(feedback: bool, field: str, value: object) -> None:
    retry = {**_RETRY, field: value}
    part = {**_feedback_wire(), "retry": retry} if feedback else {**_tool_wire(), "retry": retry}
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(part))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@_NEW_SDK
@pytest.mark.parametrize("feedback,cause", [(True, "unknown_tool"), (True, "unavailable_tool"), (False, "no_output")])
def test_retry_cause_keeps_feedback_and_tool_domains_separate(feedback: bool, cause: str) -> None:
    retry = RetryInfo(cause=cause)
    part = RetryFeedbackPart("retry", retry=retry) if feedback else ToolReturnPart("lookup", "retry", retry=retry)
    with pytest.raises(ValueError, match="retry cause"):
        encode_model_messages((ModelRequest(parts=[part]),))
    payload = {**_feedback_wire(), "retry": {**_RETRY, "cause": cause}} if feedback else {
        **_tool_wire(), "retry": {**_RETRY, "cause": cause},
    }
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(payload))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@_NEW_SDK
def test_feedback_retry_uses_declared_optional_defaults() -> None:
    restored, = decode_model_messages(_wire({**_feedback_wire(), "retry": {"cause": "model_retry"}}))
    expected = RetryFeedbackPart("fix output", retry=RetryInfo(cause="model_retry"), timestamp=_STAMP)
    assert restored == ModelRequest(parts=[expected], timestamp=_STAMP)


@_NEW_SDK
@pytest.mark.parametrize("retry", [None, "retry", {}, {**_RETRY, "extra": True}])
def test_feedback_requires_known_retry_structure(retry: object) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire({**_feedback_wire(), "retry": retry}))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("retry", [None, _RETRY])
def test_native_tool_return_cannot_carry_retry_fields(retry: object) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire({**_tool_wire(native=True), "retry": retry}, native=True))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@_NEW_SDK
@pytest.mark.parametrize("feedback", (False, True))
@pytest.mark.parametrize("invalid", ["unknown_cause", "boolean_budget", "wrong_type"])
def test_retry_writer_rejects_invalid_information_instead_of_omitting_it(feedback: bool, invalid: str) -> None:
    retry = ({"cause": "model_retry"} if invalid == "wrong_type" else
             RetryInfo(cause="unknown" if invalid == "unknown_cause" else "model_retry",
                       attempt=True if invalid == "boolean_budget" else None))
    part = RetryFeedbackPart("retry", retry=retry) if feedback else ToolReturnPart("lookup", "retry", retry=retry)
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((ModelRequest(parts=[part]),))
