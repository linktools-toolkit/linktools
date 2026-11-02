#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native fallback freezes the serving model identity before durable publication."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from tests.ai.test_decision_model_binding import _Binding


class _IdentityModel(Model):
    def __init__(self, name: str, calls: list[str], *, fail: bool = False, response_id: str | None = None) -> None:
        super().__init__()
        self.name, self.calls, self.fail, self.response_id = name, calls, fail, response_id

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def system(self) -> str:
        return "identity-test"

    async def request(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                      model_request_parameters: ModelRequestParameters) -> ModelResponse:
        self.calls.append(self.name)
        if self.fail:
            raise ModelAPIError(self.name, "synthetic failure")
        return ModelResponse([TextPart("answer")], model_name=self.name, model_id=self.response_id,
                             usage=RequestUsage(input_tokens=7, output_tokens=2))

    @asynccontextmanager
    async def request_stream(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                             model_request_parameters: ModelRequestParameters,
                             run_context: RunContext[object] | None = None) -> AsyncIterator[StreamedResponse]:
        response = await self.request(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)


class _ReplaceResponse(AbstractCapability):
    async def after_model_request(self, ctx, *, request_context, response) -> ModelResponse:
        return replace(response, model_id=None)


class _SelectionAlias(AbstractCapability):
    def __init__(self, alias: str | None) -> None:
        self.alias = alias

    async def prepare_model_request(self, ctx, request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.model_id = self.alias
        return request_context


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback,response_id,alias,replaced", [
    (False, None, None, False), (True, None, None, False), (True, "provider-explicit", None, False),
    (True, None, "selection-alias", False), (True, "provider-explicit", "selection-alias", False),
    (True, None, None, True), (True, None, "selection-alias", True),
])
async def test_native_fallback_identity_matches_sdk_and_survives_reopen(tmp_path, fallback, response_id, alias, replaced):
    calls: list[str] = []
    first = _IdentityModel("first", calls, fail=fallback, response_id=response_id)
    second = _IdentityModel("second", calls, response_id=response_id)
    capabilities = [ModelFallback(second), _SelectionAlias(alias)]
    if replaced:
        capabilities.append(_ReplaceResponse())
    native = await Agent(first, capabilities=capabilities).run("inspect")
    expected = [message for message in native.all_messages() if isinstance(message, ModelResponse)][-1]
    calls.clear()

    def features() -> RuntimeFeatures:
        result = RuntimeFeatures("fallback-identity")
        result.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
        result.model("default", _Binding(first))
        result.capability(ModelFallback(second), id="native-fallback")
        result.capability(_SelectionAlias(alias), id="selection-alias")
        if replaced:
            result.capability(_ReplaceResponse(), id="replace-response")
        result.capability(ExecutionCapability(retain_history=True))
        result.capability(TraceCapability())
        return result

    async def verify(runtime, execution_id, principal):
        interactions = (await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True)).items
        response = interactions[-1].response
        assert response["model_id"] == expected.model_id
        assert response["model_name"] == expected.model_name
        assert response["usage"]["input_tokens"] == 7
        assert len(response["failed_attempts"] or ()) == int(fallback)
        if fallback:
            assert response["failed_attempts"][0]["model_name"] == "first"
        messages = (await runtime.history.messages("session", principal=principal)).items
        final = [message for message in messages if isinstance(message, ModelResponse)][-1]
        assert final.model_id == expected.model_id
        assert final.model_name == expected.model_name
        assert final.usage.input_tokens == 7

    storage = PersistenceStorage.sqlite(tmp_path / "identity.sqlite")
    async with Persistence.open("fallback-identity", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features(),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("inspect")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        execution_id, principal = execution.execution_id, runtime.default_principal
        await verify(runtime, execution_id, principal)
        assert calls == (["first", "second"] if fallback else ["first"])
    before = list(calls)
    async with Persistence.open("fallback-identity", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features(),
    ) as runtime:
        await verify(runtime, execution_id, principal)
        assert calls == before
