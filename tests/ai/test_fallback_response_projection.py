#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDK-facing responses are safe before history and interrupted-stream capture."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequestAttempt, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from linktools.ai.model.messages import model_response_usage, project_model_response_attempts
from linktools.ai.runtime._metric_capability import PreparedRequestModel


def _response(error: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("partial")], model_name="accepted", state="incomplete",
        usage=RequestUsage(input_tokens=11, output_tokens=5, cost=Decimal("0.005")),
        failed_attempts=[ModelRequestAttempt(
            model_name="rejected", provider_name="provider", outcome="error", error=error,
            timestamp=datetime(2026, 10, 23, 12, 30, tzinfo=timezone.utc),
            duration=timedelta(milliseconds=3),
            usage=RequestUsage(input_tokens=7, output_tokens=3, cost=Decimal("0.003")),
        )],
    )


class _SnapshotModel(Model):
    def __init__(self, response: ModelResponse) -> None:
        super().__init__()
        self.response = response
        self.stream: CompletedStreamedResponse | None = None

    @property
    def model_name(self) -> str:
        return "snapshot"

    @property
    def system(self) -> str:
        return "test"

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return self.response

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self.stream = CompletedStreamedResponse(self.response, model_request_parameters=model_request_parameters)
        try:
            yield self.stream
        finally:
            self.stream.response = replace(
                self.stream.response, state="suspended", metadata={"continuation": "late-pin"},
            )


def _prepared(source: _SnapshotModel, omission: str | None) -> PreparedRequestModel:
    async def prepare(
        messages: Sequence[ModelMessage], settings: ModelSettings | None,
        parameters: ModelRequestParameters, streaming: bool,
    ) -> Model:
        return source

    def failed(error: Exception) -> None:
        pytest.fail(f"response projection incorrectly reported provider failure: {type(error).__name__}")

    return PreparedRequestModel(
        source, prepare, failed,
        project_response=lambda response: project_model_response_attempts(
            response, content_omission_reason=omission,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("omission", [None, "referenced_content_lifecycle"])
async def test_sdk_boundary_projects_attempt_errors_without_mutating_source_or_usage(
    streaming: bool, omission: str | None,
) -> None:
    error = 'ProviderError: password="private-value" detail=try-again'
    original = _response(error)
    source = _SnapshotModel(original)
    model = _prepared(source, omission)
    if streaming:
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            projected = stream.get()
    else:
        projected = await model.request([], None, ModelRequestParameters())

    assert projected is not original
    assert original.failed_attempts[0].error == error
    attempt = projected.failed_attempts[0]
    assert attempt is not original.failed_attempts[0]
    if omission:
        assert attempt.error is None
    else:
        assert "private-value" not in attempt.error
        assert "detail=try-again" in attempt.error
    assert attempt.model_name == "rejected"
    assert attempt.timestamp == original.failed_attempts[0].timestamp
    assert attempt.duration == original.failed_attempts[0].duration
    assert attempt.usage == original.failed_attempts[0].usage
    assert projected.usage == original.usage
    assert model_response_usage(projected).input_tokens == 18
    assert model_response_usage(projected).cost == Decimal("0.008")
    assert original.usage.input_tokens == 11
    assert original.failed_attempts[0].usage.input_tokens == 7


@pytest.mark.asyncio
async def test_partial_stream_snapshots_remain_fresh_and_include_post_exit_continuation_metadata() -> None:
    original = _response("ProviderError: referenced document content")
    source = _SnapshotModel(original)
    model = _prepared(source, "referenced_content_lifecycle")
    async with model.request_stream([], None, ModelRequestParameters()) as stream:
        partial = stream.get()
        assert partial.state == "incomplete"
        assert partial.failed_attempts[0].error is None
        assert source.stream is not None
        source.stream.response = replace(
            original, parts=[TextPart("finished")],
            usage=RequestUsage(input_tokens=11, output_tokens=9),
        )
        newer = stream.get()
        assert newer.parts == [TextPart("finished")]
        assert newer.usage.output_tokens == 9
        assert newer.failed_attempts[0].error is None
        assert partial.parts == [TextPart("partial")]
        assert partial.usage.output_tokens == 5

    final = stream.get()
    assert final.state == "suspended"
    assert final.metadata == {"continuation": "late-pin"}
    assert final.failed_attempts[0].error is None
    assert source.stream.response.failed_attempts[0].error == "ProviderError: referenced document content"
    assert original.metadata is None
