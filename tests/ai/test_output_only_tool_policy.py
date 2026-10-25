#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in providers limit parallel calls only for a single required output tool."""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import httpx2
import pytest
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from linktools.ai.core import JsonValue
from linktools.ai.model import create_model
from linktools.ai.model import _anthropic, _openai
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage


_Protocol = Literal["openai", "openai-responses", "anthropic"]
_PROTOCOLS: tuple[_Protocol, ...] = ("openai", "openai-responses", "anthropic")
_SCHEMA = {
    "type": "object", "properties": {"value": {"type": "string"}},
    "required": ["value"], "additionalProperties": False,
}


def _parameters() -> ModelRequestParameters:
    return ModelRequestParameters(
        output_mode="auto", allow_text_output=True,
        output_tools=[ToolDefinition(name="finish", parameters_json_schema=deepcopy(_SCHEMA))],
        output_object=OutputObjectDefinition(name="Result", json_schema=deepcopy(_SCHEMA)),
    )


def _sse(events: list[dict[str, object]]) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"},
        text="".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ),
    )


def _response(protocol: _Protocol, request: httpx2.Request) -> httpx2.Response:
    streaming = json.loads(request.content).get("stream", False)
    if protocol == "openai":
        common = {"id": "completion-fixture", "created": 1, "model": "fixture-model"}
        if not streaming:
            return httpx2.Response(200, json={
                **common, "object": "chat.completion", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            })
        chunk = {**common, "object": "chat.completion.chunk", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )
    if protocol == "openai-responses":
        item = {"id": "message-fixture", "type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": "done",
                                                       "annotations": []}]}
        body = {"id": "response-fixture", "object": "response", "created_at": 1,
                "model": "fixture-model", "status": "completed", "output": [item]}
        if not streaming:
            return httpx2.Response(200, json=body)
        return _sse([
            {"type": "response.created", "sequence_number": 0,
             "response": {**body, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
             "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0,
             "item_id": item["id"], "content_index": 0, "delta": "done", "logprobs": []},
            {"type": "response.completed", "sequence_number": 3, "response": body},
        ])
    body = {"id": "message-fixture", "type": "message", "role": "assistant",
            "model": "fixture-model", "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}
    if not streaming:
        return httpx2.Response(200, json=body)
    return _sse([
        {"type": "message_start", "message": {**body, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "done"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _model(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol, *, native_default: bool = False,
    unsupported_parallel: bool = False, max_tokens: int | None = None,
) -> tuple[Model, list[dict[str, object]]]:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _response(protocol, request)

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(_anthropic if protocol == "anthropic" else _openai,
                        "DefaultAsyncHttpxClient", client)
    profile = {"default_structured_output_mode": "native" if native_default else "tool",
               "supports_json_schema_output": True}
    if protocol != "anthropic":
        profile["openai_chat_supports_web_search"] = True
        if unsupported_parallel:
            profile["openai_unsupported_model_settings"] = ["parallel_tool_calls"]
    model = create_model(
        protocol=protocol, model="fixture-model", api_key="synthetic", profile=profile,
        base_url="https://offline.invalid/v1", max_retries=0, max_tokens=max_tokens,
    ).materialize()
    return model, bodies


async def _dispatch(
    model: Model, parameters: ModelRequestParameters, *, streaming: bool = False,
    settings: ModelSettings | None = None,
) -> None:
    original_parameters = deepcopy(parameters)
    original_settings = deepcopy(settings)
    original_defaults = deepcopy(model.settings)
    messages = [ModelRequest(parts=[UserPromptPart("synthetic")])]
    async with model:
        if streaming:
            async with model.request_stream(messages, settings, parameters) as response:
                async for _event in response:
                    pass
                result = response.get()
        else:
            result = await model.request(messages, settings, parameters)
    assert [part.content for part in result.parts if isinstance(part, TextPart)] == ["done"]
    assert parameters == original_parameters
    assert settings == original_settings
    assert model.settings == original_defaults


def _assert_parallel(body: dict[str, object], protocol: _Protocol, expected: bool | None) -> None:
    if protocol == "anthropic":
        choice = body.get("tool_choice", {})
        assert isinstance(choice, dict)
        if expected is None:
            assert "disable_parallel_tool_use" not in choice
        else:
            assert choice["disable_parallel_tool_use"] is (not expected)
    elif expected is None:
        assert "parallel_tool_calls" not in body
    else:
        assert body["parallel_tool_calls"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize(("mode", "max_tokens"), (("auto", None), ("tool", 64)))
async def test_single_required_output_tool_disables_parallel_calls_on_provider_wire(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol, streaming: bool, mode: str,
    max_tokens: int | None,
) -> None:
    model, bodies = _model(monkeypatch, protocol, max_tokens=max_tokens)
    parameters = _parameters()
    if mode == "tool":
        parameters = replace(parameters, output_mode="tool", allow_text_output=False)
    await _dispatch(model, parameters, streaming=streaming)
    assert len(bodies) == 1
    _assert_parallel(bodies[0], protocol, False)
    if max_tokens is not None:
        assert bodies[0]["max_tokens" if protocol == "anthropic" else "max_completion_tokens"
                         if protocol == "openai" else "max_output_tokens"] == max_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize(("default", "override", "expected"), (
    (None, True, True), (None, False, False), (True, None, True), (False, None, False),
    (True, False, False), (False, True, True),
))
async def test_explicit_parallel_settings_override_the_output_only_default(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol,
    default: bool | None, override: bool | None, expected: bool,
) -> None:
    model, bodies = _model(monkeypatch, protocol, max_tokens=64)
    assert model.settings is not None
    if default is not None:
        model.settings["parallel_tool_calls"] = default
    settings = {"parallel_tool_calls": override} if override is not None else None
    await _dispatch(model, _parameters(), settings=settings)
    _assert_parallel(bodies[0], protocol, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("protocol", "case"), (
    ("openai", "worker"), ("openai-responses", "multiple_outputs"),
    ("anthropic", "native_tool"), ("openai", "text_allowed"),
    ("openai-responses", "no_output_tool"), ("anthropic", "text_mode"),
))
async def test_other_request_shapes_keep_provider_parallel_behavior(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol, case: str,
) -> None:
    model, bodies = _model(monkeypatch, protocol)
    parameters = replace(_parameters(), output_mode="tool", allow_text_output=False)
    if case == "worker":
        parameters = replace(parameters, function_tools=[ToolDefinition(name="lookup")])
    elif case == "multiple_outputs":
        parameters = replace(parameters, output_tools=[*parameters.output_tools,
                                                       ToolDefinition(name="alternative")])
    elif case == "native_tool":
        parameters = replace(parameters, native_tools=[WebSearchTool()])
    elif case == "text_allowed":
        parameters = replace(parameters, allow_text_output=True)
    elif case == "no_output_tool":
        parameters = replace(parameters, output_tools=[])
    else:
        parameters = replace(parameters, output_mode="text", allow_text_output=True)
    prepared_settings, _ = model.prepare_request(None, parameters)
    assert "parallel_tool_calls" not in (prepared_settings or {})
    await _dispatch(model, parameters)
    _assert_parallel(bodies[0], protocol, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_auto_native_output_keeps_provider_parallel_behavior(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol,
) -> None:
    model, bodies = _model(monkeypatch, protocol, native_default=True)
    parameters = _parameters()
    prepared_settings, prepared_parameters = model.prepare_request(None, parameters)
    assert prepared_parameters.output_mode == "native"
    assert "parallel_tool_calls" not in (prepared_settings or {})
    await _dispatch(model, parameters)
    _assert_parallel(bodies[0], protocol, None)
    assert not any(tool.get("name") == "finish" for tool in bodies[0].get("tools", []))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("openai", "openai-responses"))
async def test_unsupported_parallel_profile_does_not_gain_an_output_only_setting(
    monkeypatch: pytest.MonkeyPatch, protocol: _Protocol,
) -> None:
    model, bodies = _model(monkeypatch, protocol, unsupported_parallel=True)
    parameters = _parameters()
    prepared_settings, _ = model.prepare_request(None, parameters)
    assert "parallel_tool_calls" not in (prepared_settings or {})
    await _dispatch(model, parameters)
    _assert_parallel(bodies[0], protocol, None)


class _CustomBinding:
    provider = "openai"
    model_identity = "openai:fixture-model"
    contract: dict[str, JsonValue] = {"provider": provider, "model_identity": model_identity}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


@pytest.mark.asyncio
async def test_runtime_keeps_caller_supplied_sdk_model_parallel_behavior() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert len(body["tools"]) == 1
        call = {"id": "call-fixture", "type": "function", "function": {
            "name": body["tools"][0]["function"]["name"], "arguments": '{"value":"done"}',
        }}
        message = {"role": "assistant", "tool_calls": [call]}
        response = {"id": "completion-fixture", "created": 1, "model": "fixture-model",
                    "object": "chat.completion", "choices": [{"index": 0, "message": message,
                                                                "finish_reason": "tool_calls"}]}
        if not body.get("stream"):
            return httpx2.Response(200, json=response)
        response["object"] = "chat.completion.chunk"
        response["choices"] = [{"index": 0, "finish_reason": "tool_calls", "delta": {
            "role": "assistant", "tool_calls": [{"index": 0, **call}],
        }}]
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = OpenAIChatModel("fixture-model", provider=OpenAIProvider(
            base_url="https://offline.invalid/v1", api_key="synthetic", http_client=client,
        ))
        features = RuntimeFeatures("custom-output")
        features.agent("default", output_schema=_SCHEMA, allow_tools=(), allow_skills=(),
                       allow_subagents=())
        features.model("default", _CustomBinding(model))
        async with Runtime.open(
            "custom-output", storage=RuntimeStorage.in_memory(), features=features,
        ) as runtime:
            result = (await runtime.agents.get().run("synthetic", timeout_seconds=10)).result
            assert result.status == "SUCCEEDED"
            assert result.output == {"value": "done"}
    assert len(bodies) == 1
    assert bodies[0]["tool_choice"] == "required"
    _assert_parallel(bodies[0], "openai", None)
