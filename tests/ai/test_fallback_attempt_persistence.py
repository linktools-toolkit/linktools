#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fallback observations remain distinct from logical requests and SDK accounting."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic_ai import RequestUsage, messages as sdk_messages
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import _message as message_codec
from linktools.ai.runtime._message import (
    decode_model_messages,
    encode_model_messages,
    model_attempts_usage,
    model_response_usage,
    project_model_response_attempts,
)


def _attempt(*, error: str | None = None, usage: RequestUsage | None = None) -> object:
    attempt_type = getattr(sdk_messages, "ModelRequestAttempt", None)
    if attempt_type is None:
        pytest.skip("installed SDK does not expose fallback attempt observations")
    return attempt_type(
        model_name="decision-model",
        provider_name="decisions",
        outcome="error" if error is not None else "rejected",
        error=error,
        timestamp=datetime(2026, 10, 23, 12, 30, tzinfo=timezone.utc),
        duration=timedelta(seconds=2, microseconds=301),
        usage=usage,
    )


def test_attempts_round_trip_preserves_order_timing_and_unknown_usage() -> None:
    attempts = [
        _attempt(error="UnsureRoute: confidence below threshold"),
        _attempt(usage=RequestUsage(
            input_tokens=7, output_tokens=3, cost=Decimal("0.003"),
            details={"reasoning_tokens": 2}, future_tokens=4,
        )),
    ]
    response = ModelResponse(
        parts=[TextPart("accepted")], usage=RequestUsage(input_tokens=11, output_tokens=5),
        failed_attempts=attempts,
    )
    wire = encode_model_messages((response,))
    restored = decode_model_messages(wire)[0]

    assert restored == response
    assert restored.failed_attempts == attempts
    assert restored.failed_attempts[0].usage is None
    assert restored.failed_attempts[1].usage.__dict__["future_tokens"] == 4
    assert encode_model_messages((restored,)) == wire
    assert json.loads(wire)[0]["failed_attempts"][0]["duration_us"] == 2000301


def test_attempt_projection_redacts_secrets_without_discarding_ordinary_diagnostics() -> None:
    attempt = _attempt(error='ModelAPIError: password="private-value" detail=try-again')
    response = ModelResponse(parts=[], failed_attempts=[attempt])
    projected = project_model_response_attempts(response)

    assert projected is not response
    assert "private-value" not in projected.failed_attempts[0].error
    assert "detail=try-again" in projected.failed_attempts[0].error
    assert "private-value" in response.failed_attempts[0].error
    assert b"private-value" not in encode_model_messages((response,))
    assert b"detail=try-again" in encode_model_messages((response,))


def test_reference_omission_removes_only_attempt_error_content_without_mutation() -> None:
    attempt = _attempt(
        error="ProviderError: private referenced document content",
        usage=RequestUsage(input_tokens=9, cost=Decimal("0.002")),
    )
    response = ModelResponse(parts=[TextPart("accepted")], failed_attempts=[attempt])
    projected = project_model_response_attempts(
        response, content_omission_reason="referenced_content_lifecycle",
    )
    stored = decode_model_messages(encode_model_messages((projected,)))[0]

    assert projected is not response
    assert projected.failed_attempts[0] is not attempt
    assert response.failed_attempts[0].error == "ProviderError: private referenced document content"
    assert stored.failed_attempts[0].error is None
    assert stored.failed_attempts[0].model_name == attempt.model_name
    assert stored.failed_attempts[0].provider_name == attempt.provider_name
    assert stored.failed_attempts[0].timestamp == attempt.timestamp
    assert stored.failed_attempts[0].duration == attempt.duration
    assert stored.failed_attempts[0].usage == attempt.usage
    assert stored.parts == response.parts
    assert project_model_response_attempts(
        projected, content_omission_reason="referenced_content_lifecycle",
    ) == projected


def test_usage_projection_sums_known_attempts_once_without_mutating_sdk_usage() -> None:
    rejected_usage = RequestUsage(
        input_tokens=7, output_tokens=3, cost=Decimal("0.003"), details={"reasoning_tokens": 2},
    )
    accepted_usage = RequestUsage(
        input_tokens=11, output_tokens=5, cost=Decimal("0.005"), details={"reasoning_tokens": 4},
    )
    attempts = [_attempt(error="UnsureRoute: unknown usage"), _attempt(usage=rejected_usage)]
    response = ModelResponse(parts=[], usage=accepted_usage, failed_attempts=attempts)
    observed = model_response_usage(response)
    failed = model_attempts_usage(attempts)

    assert observed.input_tokens == 18
    assert observed.output_tokens == 8
    assert observed.cost == Decimal("0.008")
    assert observed.details == {"reasoning_tokens": 6}
    assert failed.input_tokens == 7
    assert failed.cost == Decimal("0.003")
    assert rejected_usage.details == {"reasoning_tokens": 2}
    assert accepted_usage.details == {"reasoning_tokens": 4}
    assert accepted_usage.input_tokens == 11
    assert model_response_usage(response) == observed
    assert model_attempts_usage(attempts[:1]) is None
    assert model_attempts_usage(None) is None


def test_unavailable_attempts_are_not_invented_by_the_message_writer() -> None:
    response = ModelResponse(parts=[], usage=RequestUsage(input_tokens=4))
    value = json.loads(encode_model_messages((response,)))[0]

    assert "failed_attempts" not in value
    assert model_response_usage(response).input_tokens == 4
    assert model_attempts_usage(None) is None
    assert decode_model_messages(canonical_json_bytes([value])) == (response,)


def test_explicit_empty_attempt_observation_is_distinct_from_unavailable() -> None:
    _attempt()
    response = ModelResponse(parts=[], failed_attempts=[])
    wire = encode_model_messages((response,))

    assert json.loads(wire)[0]["failed_attempts"] == []
    assert decode_model_messages(wire)[0].failed_attempts == []


@pytest.mark.parametrize(("field", "value"), [
    ("duration_us", True),
    ("duration_us", -1),
    ("duration_us", 10 ** 40),
    ("outcome", "success"),
    ("timestamp", "2026-10-23T12:30:00"),
    ("usage", {"input_tokens": 1}),
])
def test_malformed_attempt_fields_fail_closed(field: str, value: object) -> None:
    response = ModelResponse(parts=[], failed_attempts=[_attempt()])
    wire = json.loads(encode_model_messages((response,)))
    wire[0]["failed_attempts"][0][field] = value

    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_nonempty_attempts_cannot_be_restored_by_an_sdk_without_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    response = ModelResponse(parts=[], failed_attempts=[_attempt()])
    wire = encode_model_messages((response,))
    monkeypatch.setattr(message_codec, "_MODEL_REQUEST_ATTEMPT", None)

    with pytest.raises(AIError) as caught:
        decode_model_messages(wire)
    assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
