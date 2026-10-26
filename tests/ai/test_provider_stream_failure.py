#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Provider stream failures retain their cause across SDK task cancellation."""
from ._runtime_test_helpers import model_features

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import ModelClientRequestRecorder
from linktools.ai.model._http import capture_request
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._metric_capability import PreparedRequestModel
from linktools.ai.capability import TraceCapability

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

    group = RuntimeFeatures("provider-stream-failure")
    group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    async with Persistence.open('provider-stream-failure', storage=PersistenceStorage.in_memory()) as persistence_67, Runtime.open(persistence=persistence_67, features=model_features(_Models(ExitModel(stream_function=stream)), group)) as runtime:
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
@pytest.mark.parametrize("client_capture", (False, True))
async def test_provider_observation_preserves_consumer_error_and_context_suppression(suppress: bool, client_capture: bool) -> None:
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

    async def prepare(messages, settings, parameters, streaming):
        return model.wrapped

    recorder = ModelClientRequestRecorder(record_http_bodies=True)
    consumer_model = ConsumerModel(lambda messages, info: response)
    model = PreparedRequestModel(
        recorder.wrap_model(consumer_model) if client_capture else consumer_model, prepare, provider_errors.append,
    )
    error = ConsumerFailure("consumer output failed")

    async def consume() -> None:
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            assert stream.get() == response
            raise error

    with recorder:
        if suppress:
            await consume()
        else:
            with pytest.raises(ConsumerFailure) as caught:
                await consume()
            assert caught.value is error
    assert received == [error]
    assert provider_errors == []


@pytest.mark.asyncio
async def test_provider_cleanup_error_is_observed_without_reclassifying_consumer_failure() -> None:
    consumer_error = ValueError("consumer failed")
    cleanup_error = RuntimeError("provider cleanup failed")
    provider_errors: list[Exception] = []
    observations: list[tuple[str, BaseException | None]] = []

    class CleanupModel(_UsageFunctionModel):
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            try:
                yield CompletedStreamedResponse(ModelResponse(parts=[TextPart("answer")]),
                                                model_request_parameters=model_request_parameters)
            finally:
                raise cleanup_error

    async def prepare(messages, settings, parameters, streaming):
        return model.wrapped

    model = PreparedRequestModel(
        CleanupModel(lambda messages, info: ModelResponse(parts=[TextPart("answer")])),
        prepare, provider_errors.append, lambda event, error: observations.append((event, error)),
    )
    with pytest.raises(RuntimeError) as caught:
        async with model.request_stream([], None, ModelRequestParameters()):
            raise consumer_error
    assert caught.value is cleanup_error and cleanup_error.__context__ is consumer_error
    assert provider_errors == [cleanup_error]
    assert observations == [("entered", None), ("exit_error", cleanup_error)]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("entry", "iteration", "exit", "consumer_exit"))
async def test_task_cancellation_is_observed_at_sdk_boundary_without_provider_failure(phase: str) -> None:
    blocked = asyncio.Event()
    provider_errors: list[Exception] = []
    recorder = ModelClientRequestRecorder()

    async def block() -> None:
        blocked.set()
        await asyncio.Event().wait()

    class SDKResponse(CompletedStreamedResponse):
        async def _get_event_iterator(self):
            if phase == "iteration":
                await block()
            async for event in super()._get_event_iterator():
                yield event

    class SDKModel(_UsageFunctionModel):
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            await capture_request(httpx2.Request("POST", "https://model.invalid/responses", json={}))
            if phase == "entry":
                await block()
            try:
                yield SDKResponse(ModelResponse(parts=[TextPart("answer")]),
                                  model_request_parameters=model_request_parameters)
            finally:
                if phase in {"exit", "consumer_exit"}:
                    await block()

    async def prepare(messages, settings, parameters, streaming):
        return model.wrapped

    def observe(event: str, error: BaseException | None) -> None:
        recorder.observe_stream(event, attempt_index=len(recorder.requests) - 1, error=error)

    model = PreparedRequestModel(
        SDKModel(lambda messages, info: ModelResponse(parts=[])), prepare, provider_errors.append, observe,
    )

    async def consume() -> None:
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            if phase == "consumer_exit":
                raise ValueError("consumer failure before cleanup")
            async for _ in stream:
                pass

    with recorder:
        task = asyncio.create_task(consume())
        await asyncio.wait_for(blocked.wait(), 3)
        task.cancel("task cancelled")
        with pytest.raises(asyncio.CancelledError):
            await task
    assert provider_errors == []
    diagnostics = recorder.requests[0].diagnostics
    boundary = "exit" if phase == "consumer_exit" else phase
    assert diagnostics[f"sdk_{boundary}_error_type"] == "asyncio.exceptions.CancelledError"
    assert diagnostics[f"sdk_{boundary}_error_ns"] >= diagnostics.get("sdk_entered_ns", 0)
    assert "sdk_cancel_requested_ns" not in diagnostics and "read_outcome" not in diagnostics
    assert "sdk_entry_error_type" not in diagnostics or phase == "entry"
    assert "sdk_exit_error_type" not in diagnostics or boundary == "exit"
