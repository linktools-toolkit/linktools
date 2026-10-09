#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Runtime tests and persistence fixtures."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import (
    CompletedStreamedResponse,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from linktools.ai.capability import tool_metadata
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import StoredPayload


_ReadT = TypeVar("_ReadT")


async def _wait_for_committed(
    read: Callable[[], Awaitable[_ReadT]],
    ready: Callable[[_ReadT], bool],
    *,
    timeout: float = 5.0,
) -> _ReadT:
    """Wait for a canonical read condition without advancing the producer."""
    async def wait() -> _ReadT:
        while True:
            value = await read()
            if ready(value):
                return value
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(wait(), timeout=timeout)


def tool_run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


def execution_owner_fields(prompt: str = "prompt") -> dict[str, object]:
    return {
        "principal_id": "principal",
        "principal_kind": "service",
        "stored_user_input": StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
        ),
    }


def tool_with_metadata(
    function: Callable[..., Any],
    descriptor: ManagedToolDescriptor,
) -> Tool[Any]:
    path_fields = list(descriptor.workspace_path_fields) or None
    return Tool(
        function,
        metadata=tool_metadata(
            effect_policy=descriptor.effect_policy,
            tool_class=descriptor.tool_class,
            path_fields=path_fields,
        ),
    )


async def _runtime_usage_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del messages, info
    return ModelResponse(
        parts=[TextPart("done")],
        usage=RequestUsage(
            input_tokens=101,
            output_tokens=202,
            cache_read_tokens=303,
            cache_write_tokens=404,
        ),
    )


class _UsageFunctionModel(FunctionModel):
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        response = await self.request(
            messages,
            model_settings,
            model_request_parameters,
        )
        yield CompletedStreamedResponse(
            response,
            model_request_parameters=model_request_parameters,
            replay_events=True,
        )


class _RuntimeUsageModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:usage"
    vision = False
    model_digest = "u" * 64
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "usage",
    }

    def materialize(self) -> FunctionModel:
        return _UsageFunctionModel(_runtime_usage_model)


class RuntimeUsageModels:
    def capture(self) -> "RuntimeUsageModels":
        return self

    def resolve(self, route_id: str) -> _RuntimeUsageModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _RuntimeUsageModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _RuntimeUsageModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _RuntimeUsageModelBinding.contract
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _RuntimeUsageModelBinding()
