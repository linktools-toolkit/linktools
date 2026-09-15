#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Runtime tests and persistence fixtures."""

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import (
    CompletedStreamedResponse,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import tool_semantic_metadata
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.storage import StoredPayload
from linktools.ai.workspace import Workspace


def execution_owner_fields(prompt: str = "prompt") -> dict[str, object]:
    return {
        "principal_id": "principal",
        "principal_kind": "service",
        "stored_user_input": StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
        ),
    }


def semantic_tool(
    function: Callable[..., Any],
    descriptor: ManagedToolDescriptor,
) -> Tool[Any]:
    path_fields = list(descriptor.workspace_path_fields) or None
    return Tool(
        function,
        metadata=tool_semantic_metadata(
            effect=descriptor.effect,
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
    fingerprint = "u" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "usage",
    }

    def materialize(self) -> FunctionModel:
        return _UsageFunctionModel(_runtime_usage_model)


class RuntimeUsageModels:
    def snapshot(self) -> "RuntimeUsageModels":
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
            or dict(payload) != _RuntimeUsageModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _RuntimeUsageModelBinding()


def runtime_usage_workspace(path: Path) -> Workspace:
    path.mkdir(parents=True)
    agent_path = path / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    return Workspace.load(path, workspace_id="workspace")
