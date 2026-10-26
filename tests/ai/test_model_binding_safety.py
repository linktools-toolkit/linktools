#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model binding must not expose credential material."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.exceptions import ModelHTTPError, RunCancelled, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.agent._model_resolver import _ModelResolver
from linktools.ai.model import ModelBinding, create_model, model_binding_error
from linktools.ai.model._binding import _RetryingModel


def test_model_binding_contract_excludes_secret_material() -> None:
    registry = create_model(api="openai", model="gpt-test", api_key="secret")
    binding = registry

    assert "secret" not in repr(binding)
    assert "secret" not in repr(dict(binding.contract))
    assert binding.model_identity == "openai:gpt-test"


def test_static_connection_values_are_normalized() -> None:
    binding = create_model(api="openai", model="gpt-test",
                           base_url="  https://EXAMPLE.com/v1/  ", api_key="   ")
    assert binding.base_url == "https://example.com/v1"
    assert binding.api_key is None


@pytest.mark.parametrize("field,value", [("timeout", 0), ("max_retries", -1), ("max_retries", None),
                                         ("retry_delay", -1), ("max_tokens", 0)])
def test_static_connection_and_request_settings_are_validated(field, value) -> None:
    with pytest.raises(ValueError):
        create_model(api="openai", model="gpt-test",
                     **({"model_settings": {field: value}} if field in {"max_tokens", "timeout"} else {field: value}))


def test_model_route_capture_keeps_shared_binding_when_source_map_changes() -> None:
    original = create_model(api="openai", model="gpt-old")
    source = {"target": original, "alias": original, "alias-2": original}
    snapshot = _ModelResolver(source)
    source["target"] = create_model(api="openai", model="gpt-new")
    assert snapshot.resolve("target") is original
    assert snapshot.resolve("alias") is original
    assert snapshot.resolve("alias-2") is original
    assert _ModelResolver(source).resolve("target").model_identity == "openai:gpt-new"
    with pytest.raises(AttributeError):
        snapshot.resolve("alias").model = "changed"


def test_model_resolver_requires_an_existing_route() -> None:
    with pytest.raises(AIError) as raised:
        _ModelResolver({}).resolve("missing")
    assert raised.value.code is ErrorCode.MODEL_CONNECTION_NOT_FOUND


@pytest.mark.parametrize(
    "error",
    (
        TimeoutError("execution deadline"),
        RunCancelled("cancelled"),
        UnexpectedModelBehavior("output"),
    ),
)
def test_model_binding_projection_leaves_execution_and_output_failures_to_callers(
    error: Exception,
) -> None:
    assert model_binding_error(error) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("after_stream_entry", (False, True))
async def test_model_transport_retry_stops_after_stream_entry(
    after_stream_entry: bool,
) -> None:
    error = ModelHTTPError(503, "model", body={"secret": "provider response"})

    class Provider(TestModel):
        attempts = 0

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[object] | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            self.attempts += 1
            if self.attempts == 1 and not after_stream_entry:
                raise error
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context,
            ) as response:
                yield response
                if after_stream_entry:
                    raise error

    provider = Provider()
    model = _RetryingModel(provider, 2, 0)

    async def request() -> None:
        async with model.request_stream([], None, ModelRequestParameters()):
            pass

    if after_stream_entry:
        with pytest.raises(ModelHTTPError) as captured:
            await request()
        assert captured.value is error
        assert provider.attempts == 1
    else:
        await request()
        assert provider.attempts == 2
