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
from linktools.ai.model import ModelResolver, ModelBinding, create_model, model_binding_error
from linktools.ai.model._openai import _OpenAIModelBinding
from linktools.ai.model._binding import _RetryingModel, _resolved_connection


def test_model_binding_contract_excludes_secret_material() -> None:
    registry = create_model(protocol="openai", model="gpt-test", api_key="secret")
    binding = registry

    assert "secret" not in repr(binding)
    assert "secret" not in repr(dict(binding.contract))
    assert binding.model_identity == "openai:gpt-test"


def test_openai_connection_resolver_is_lazy_and_overrides_defaults() -> None:
    calls: list[str] = []

    def resolve() -> dict[str, object]:
        calls.append("called")
        return {"api_key": "resolved", "max_retries": 4}

    binding = _OpenAIModelBinding(
        "gpt-test",
        api_key="default",
        connection_resolver=resolve,
    )

    assert calls == []
    values = _resolved_connection(binding)
    assert values["api_key"] == "resolved"
    assert values["max_retries"] == 4
    assert calls == ["called"]


def test_openai_connection_resolver_rejects_model_fields() -> None:
    binding = _OpenAIModelBinding(
        "gpt-test",
        connection_resolver=lambda: {"model": "other"},
    )

    with pytest.raises(AIError) as raised:
        _resolved_connection(binding)
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


def test_openai_connection_resolver_normalizes_connection_values() -> None:
    binding = _OpenAIModelBinding(
        "gpt-test",
        connection_resolver=lambda: {
            "base_url": "  https://EXAMPLE.com/v1/  ",
            "api_key": "   ",
        },
    )
    values = _resolved_connection(binding)
    assert values["base_url"] == "https://example.com/v1"
    assert values["api_key"] is None


def test_openai_connection_resolver_invalid_values_keep_a_cause() -> None:
    binding = _OpenAIModelBinding(
        "gpt-test",
        connection_resolver=lambda: {"max_retries": None},
    )

    with pytest.raises(AIError) as raised:
        _resolved_connection(binding)
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.__cause__ is not None


def test_model_route_capture_keeps_shared_binding_when_source_map_changes() -> None:
    original = create_model(protocol="openai", model="gpt-old")
    source = {"target": original, "alias": original, "alias-2": original}
    snapshot = ModelResolver(source)
    source["target"] = create_model(protocol="openai", model="gpt-new")
    assert snapshot.resolve("target") is original
    assert snapshot.resolve("alias") is original
    assert snapshot.resolve("alias-2") is original
    assert ModelResolver(source).resolve("target").model_identity == "openai:gpt-new"
    with pytest.raises(AttributeError):
        snapshot.resolve("alias").model = "changed"


def test_model_resolver_requires_an_existing_route() -> None:
    with pytest.raises(AIError) as raised:
        ModelResolver({}).resolve("missing")
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
