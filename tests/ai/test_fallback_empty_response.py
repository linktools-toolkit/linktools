#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""An observed empty response is distinct from missing provider usage."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import FallbackBinding
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage

from .test_runtime_fallback_failures import _Binding


class _Response(CompletedStreamedResponse):
    def get(self) -> ModelResponse:
        return replace(super().get(), metadata=self.metadata)


class _EmptyThenFailure(FunctionModel):
    def __init__(self, *, initial_response: bool, input_tokens: int = 0) -> None:
        super().__init__(stream_function=self._unused_stream, model_name="empty")
        self._initial_response = initial_response
        self._input_tokens = input_tokens

    async def _unused_stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError("the completed response adapter owns streaming")
        yield ""  # pragma: no cover

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del messages, model_settings, run_context
        if not self._initial_response:
            raise ModelAPIError("empty", "unmeasured provider failure")
        self._initial_response = False
        response = ModelResponse(
            [], model_name=None, provider_response_id="actual-response",
            state="suspended", usage=RequestUsage(input_tokens=self._input_tokens),
        )
        yield _Response(response, model_request_parameters=model_request_parameters, replay_events=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_response", [False, True])
@pytest.mark.parametrize("input_tokens", [0, 7])
async def test_empty_zero_usage_response_is_not_confused_with_no_response(initial_response: bool, input_tokens: int) -> None:
    features = RuntimeFeatures("empty-fallback-response")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", FallbackBinding(_Binding(_EmptyThenFailure(initial_response=initial_response, input_tokens=input_tokens))))
    features.capability(TraceCapability(), scope="runtime")

    async with Persistence.open('empty-fallback-response', storage=PersistenceStorage.in_memory()) as persistence_64, Runtime.attach(persistence=persistence_64, features=features) as runtime:
        execution = await runtime.agents.get().start("respond")
        result = (await execution.wait(timeout_seconds=10)).result
        interactions = (await execution.model_interactions(include_content=True)).items

    assert result.status is ExecutionStatus.FAILED
    assert len(interactions) == 1
    assert interactions[0].status == "FAILED"
    usage = interactions[0].usage
    if initial_response:
        assert usage is not None
        assert usage.input_tokens == input_tokens
        assert result.usage.input_tokens == input_tokens
        assert result.usage.output_tokens == 0
        assert usage.output_tokens == 0
        assert usage.model_requests == 1
    else:
        assert usage is None
