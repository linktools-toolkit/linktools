#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider stream failures retain their cause across SDK task cancellation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._metric_capability import _PreparedRequestModel

from ._runtime_test_helpers import _UsageFunctionModel
from .test_live_history_readback_integration import _Models


class ProviderFailure(ValueError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("entry", "iteration", "exit"))
async def test_provider_stream_failure_is_not_wrapper_cancellation(boundary: str) -> None:
    failures: list[ProviderFailure] = []

    def fail() -> None:
        error = ProviderFailure(boundary)
        failures.append(error)
        raise error

    async def stream(messages, info):
        del messages, info
        if boundary == "entry":
            fail()
        yield "provider text"
        if boundary == "iteration":
            fail()

    class ExitModel(FunctionModel):
        @asynccontextmanager
        async def request_stream(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters, run_context=None,
        ) -> AsyncIterator[StreamedResponse]:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context,
            ) as response:
                yield response
            if boundary == "exit":
                fail()

    group = CapabilityGroup("provider-stream-failure")
    group.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "provider-stream-failure", models=_Models(ExitModel(stream_function=stream)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("respond")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert len(failures) == 1
        interactions = (await execution.model_interactions(include_content=True)).items
        assert [value.status for value in interactions] == ["FAILED"]
        assert interactions[0].error_code == "INTERNAL_ERROR"
        assert [(value.payload["kind"], value.payload["status"]) for value in (await execution.trace()).items] == [
            ("MODEL_REQUEST", "STARTED"), ("MODEL_RESPONSE", "FAILED"),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", (False, True))
async def test_provider_observation_preserves_consumer_error_and_context_suppression(suppress: bool) -> None:
    class ConsumerFailure(ValueError):
        pass

    received: list[ConsumerFailure] = []
    provider_errors: list[Exception] = []
    response = ModelResponse(parts=[TextPart("answer")])

    class ConsumerModel(_UsageFunctionModel):
        @asynccontextmanager
        async def request_stream(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters, run_context=None,
        ) -> AsyncIterator[StreamedResponse]:
            try:
                yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)
            except ConsumerFailure as error:
                received.append(error)
                if not suppress:
                    raise

    async def prepare(messages, settings, parameters, streaming) -> None:
        pass

    model = _PreparedRequestModel(ConsumerModel(lambda messages, info: response), prepare, provider_errors.append)
    error = ConsumerFailure("consumer output failed")

    async def consume() -> None:
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            assert stream.get() == response
            raise error

    if suppress:
        await consume()
    else:
        with pytest.raises(ConsumerFailure) as caught:
            await consume()
        assert caught.value is error
    assert received == [error]
    assert provider_errors == []
