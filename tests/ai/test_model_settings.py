#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native request extensions and SDK thinking retain their actual wire semantics."""

import json
from collections.abc import Mapping

import httpx2
import pytest
import yaml

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.model import ModelBinding, ModelClientRequestRecorder, create_model
from linktools.ai.model import _anthropic, _openai
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from .test_model_protocols import response, stream_events


def _binding(monkeypatch: pytest.MonkeyPatch, protocol: str, model: str,
             sent: list[dict], *, model_settings: Mapping[str, JsonValue] | None = None) -> ModelBinding:
    def capture(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        for key, value in (model_settings or {}).get("extra_headers", {}).items():
            assert request.headers[key] == value
        if (model_settings or {}).get("timeout") is not None:
            assert request.extensions["timeout"]["read"] == model_settings["timeout"]
        if sent[-1].get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                  content=b"".join(stream_events(protocol)))
        return httpx2.Response(200, json=response(protocol))

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(capture), **kwargs)

    monkeypatch.setattr(_anthropic if protocol == "anthropic" else _openai,
                        "DefaultAsyncHttpxClient", client)
    settings = dict(model_settings or {})
    limit = {"openai": "max_completion_tokens", "openai-responses": "max_output_tokens", "anthropic": "max_tokens"}[protocol]
    if limit not in settings.get("extra_body", {}):
        settings.setdefault("max_tokens", 12000)
    return create_model(protocol=protocol, model=model, api_key="mock-only",
                        base_url="https://model.invalid/v1", max_retries=0, model_settings=settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol,limit", [
    ("openai", "max_completion_tokens"), ("openai-responses", "max_output_tokens"),
    ("anthropic", "max_tokens"),
])
async def test_unknown_native_fields_and_raw_only_output_limit_reach_wire(monkeypatch, protocol, limit):
    sent = []
    original = {"vendor_option": {"tokens": 3072, "labels": ["custom"]}, limit: 9000}
    binding = _binding(monkeypatch, protocol, "unregistered-model-alias", sent, model_settings={"extra_body": original})
    original["vendor_option"]["tokens"] = 1
    saved = json.loads(json.dumps(dict(binding.contract)))
    saved["settings"]["extra_body"]["vendor_option"]["labels"].append("mutated")
    assert binding.contract["settings"]["extra_body"]["vendor_option"] == {
        "tokens": 3072, "labels": ["custom"],
    }
    for _ in range(2):
        model = binding.materialize()
        async with model:
            await model.request([ModelRequest(parts=[UserPromptPart("hello")])],
                                {"thinking": False}, ModelRequestParameters())
        model.settings["extra_body"]["vendor_option"]["labels"].append("local mutation")
    assert len(sent) == 2
    for body in sent:
        assert body["vendor_option"] == {"tokens": 3072, "labels": ["custom"]}
        assert body[limit] == 9000
    assert "max_tokens" not in binding.contract["settings"]
    assert "mock-only" not in repr(dict(binding.contract))


@pytest.mark.asyncio
async def test_native_manual_budget_overrides_unified_false(monkeypatch):
    sent = []
    binding = _binding(monkeypatch, "anthropic", "claude-sonnet-4-5", sent,
                       model_settings={"anthropic_thinking": {"type": "enabled", "budget_tokens": 4096}})
    model = binding.materialize()
    async with model:
        await model.request([ModelRequest(parts=[UserPromptPart("hello")])],
                            {"thinking": False}, ModelRequestParameters())
    assert sent[0]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert sent[0]["max_tokens"] == 12000


@pytest.mark.parametrize("value", [[1], "body", {"value": object()}, {"value": float("nan")}])
def test_model_settings_requires_json_object(value):
    with pytest.raises((TypeError, ValueError)):
        create_model(protocol="openai", model="alias", model_settings=value)


@pytest.mark.parametrize("protocol", ["openai", "openai-responses", "anthropic"])
@pytest.mark.parametrize("field", ["max_tokens", "timeout"])
def test_request_settings_have_no_duplicate_factory_argument(protocol, field):
    with pytest.raises(TypeError):
        create_model(protocol=protocol, model="fixture", **{field: 1000})


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol,limit", [
    ("openai", "max_completion_tokens"), ("openai-responses", "max_output_tokens"),
    ("anthropic", "max_tokens"),
])
async def test_explicit_null_limit_keeps_native_semantics(monkeypatch, protocol, limit):
    sent = []
    binding = _binding(monkeypatch, protocol, "fixture", sent,
                       model_settings={"max_tokens": None, "timeout": None})
    async with binding.materialize() as model:
        await model.request([ModelRequest(parts=[UserPromptPart("hello")])], None, ModelRequestParameters())
    assert binding.model_settings["max_tokens"] is None
    assert binding.model_settings["timeout"] is None
    assert limit in sent[0] and sent[0][limit] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking,expected", [(False, "minimal"), ("minimal", "minimal"), ("max", "high")])
async def test_runtime_uses_sdk_thinking_mapping_without_local_model_rejection(monkeypatch, thinking, expected):
    sent = []
    binding = _binding(monkeypatch, "openai-responses", "gpt-5", sent)
    features = RuntimeFeatures()
    features.model("default", binding)
    features.agent("default", model="default")
    features.capability(Thinking(effort="high"))
    async with Runtime.open("native-thinking", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run("hello", thinking=thinking, timeout_seconds=10)
    assert result.result.status is ExecutionStatus.SUCCEEDED
    assert sent[-1]["reasoning"]["effort"] == expected


@pytest.mark.asyncio
async def test_runtime_unknown_model_keeps_native_extension_without_profile(monkeypatch):
    sent = []
    binding = _binding(monkeypatch, "openai", "unregistered-model-alias", sent,
                       model_settings={"extra_body": {"vendor_thinking": {"budget": 2048}}})
    features = RuntimeFeatures()
    features.model("default", binding)
    features.agent("default", model="default")
    async with Runtime.open("unknown-model", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run("hello", thinking=True, timeout_seconds=10)
    assert result.result.status is ExecutionStatus.SUCCEEDED
    assert sent[-1]["vendor_thinking"] == {"budget": 2048}
    assert "reasoning_effort" not in sent[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol,model,limit", [
    ("openai", "gpt-4o", "max_completion_tokens"),
    ("openai-responses", "gpt-4o", "max_output_tokens"),
    ("anthropic", "claude-opus-4-5", "max_tokens"),
])
async def test_yaml_sdk_settings_override_defaults_and_keep_transport_out_of_contract(
    monkeypatch, protocol, model, limit,
):
    sent = []
    configuration = yaml.safe_load("""
model_settings:
  temperature: 0.25
  max_tokens: 6000
  timeout: 13
  request_timeout: 20
  max_retries: 0
  extra_headers:
    x-fixture-secret: transport-only
  extra_body:
    vendor_feature:
      enabled: true
""")
    binding = _binding(monkeypatch, protocol, model, sent, **configuration)
    assert binding.model_settings["max_tokens"] == 6000 and binding.model_settings["timeout"] == 13
    assert binding.contract["settings"] == {
        "temperature": 0.25, "max_tokens": 6000,
        "extra_body": {"vendor_feature": {"enabled": True}},
    }
    assert "transport-only" not in repr(binding)
    assert "transport-only" not in repr(binding.contract)
    from linktools.ai.runtime._model_interaction import request_envelope
    envelope, encoded = request_envelope(model_settings=dict(binding.model_settings),
                                         parameters=ModelRequestParameters(), streaming=False)
    assert "transport-only" not in str(envelope)
    assert b"transport-only" not in encoded
    sdk_model = binding.materialize()
    assert sdk_model.settings["timeout"] == 13
    assert sdk_model.settings["extra_headers"] == {"x-fixture-secret": "transport-only"}
    async with sdk_model:
        await sdk_model.request([ModelRequest(parts=[UserPromptPart("hello")])],
                                None, ModelRequestParameters())
    assert sent[-1]["temperature"] == 0.25
    assert sent[-1][limit] == 6000
    assert sent[-1]["vendor_feature"] == {"enabled": True}
    assert not {"timeout", "request_timeout", "max_retries", "extra_headers"} & sent[-1].keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol", ["openai", "openai-responses", "anthropic"])
@pytest.mark.parametrize("location", ["missing", "binding", "request"])
async def test_sdk_zero_does_not_disable_binding_retry_owner(monkeypatch, streaming, protocol, location):
    from pydantic_ai.exceptions import ModelHTTPError
    attempts = []
    clients = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request)
        return httpx2.Response(429, headers={"retry-after": "0.001"},
                               json={"error": {"type": "rate_limit_error", "message": "mock failure"}})

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(_anthropic if protocol == "anthropic" else _openai,
                        "DefaultAsyncHttpxClient", client)
    settings = {"max_tokens": 1000}
    if location == "binding":
        settings["max_retries"] = 0
    binding = create_model(protocol=protocol, model="fixture", api_key="mock-only",
                           base_url="https://model.invalid", max_retries=2, retry_delay=0,
                           model_settings=settings)
    request_settings = {"max_retries": 0} if location == "request" else None
    model = binding.materialize()
    with ModelClientRequestRecorder() as recorder:
        async with model:
            with pytest.raises(ModelHTTPError):
                if streaming:
                    async with model.request_stream([ModelRequest(parts=[UserPromptPart("hello")])],
                                                     request_settings, ModelRequestParameters()) as stream:
                        async for _ in stream:
                            pass
                else:
                    await model.request([ModelRequest(parts=[UserPromptPart("hello")])],
                                        request_settings, ModelRequestParameters())
    assert len(attempts) == 3
    assert len(recorder.requests) == 3
    assert clients and all(client.is_closed for client in clients)


@pytest.mark.parametrize("protocol", ["openai", "openai-responses", "anthropic"])
@pytest.mark.parametrize("value", [None, 1, 2, False])
def test_sdk_retry_override_rejected_at_binding(protocol, value):
    with pytest.raises(AIError) as raised:
        create_model(protocol=protocol, model="fixture", model_settings={"max_retries": value})
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol", ["openai", "openai-responses", "anthropic"])
@pytest.mark.parametrize("value", [None, 1, 2, False])
async def test_sdk_retry_override_rejected_before_request(monkeypatch, streaming, protocol, value):
    sent = []
    binding = _binding(monkeypatch, protocol, "fixture", sent)
    async with binding.materialize() as model:
        with pytest.raises(AIError):
            if streaming:
                async with model.request_stream([ModelRequest(parts=[UserPromptPart("hello")])],
                                                 {"max_retries": value}, ModelRequestParameters()) as stream:
                    async for _ in stream:
                        pass
            else:
                await model.request([ModelRequest(parts=[UserPromptPart("hello")])],
                                    {"max_retries": value}, ModelRequestParameters())
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_thinking,run_thinking,expected", [
    (None, None, "high"), (False, None, "minimal"), ("low", None, "low"),
    ("low", False, "minimal"), (False, "max", "high"),
])
async def test_yaml_thinking_is_inherited_until_spec_or_run_explicitly_overrides(
    monkeypatch, spec_thinking, run_thinking, expected,
):
    sent = []
    configuration = yaml.safe_load("model_settings:\n  thinking: high\n")
    binding = _binding(monkeypatch, "openai-responses", "gpt-5", sent, **configuration)
    features = RuntimeFeatures()
    features.model("default", binding)
    features.agent("default", model="default",
                   **({} if spec_thinking is None else {"thinking": spec_thinking}))
    async with Runtime.open("inherited-thinking", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run(
            "hello", timeout_seconds=10,
            **({} if run_thinking is None else {"thinking": run_thinking}),
        )
    assert result.result.status is ExecutionStatus.SUCCEEDED
    assert sent[-1]["reasoning"]["effort"] == expected


@pytest.mark.asyncio
async def test_omitted_thinking_preserves_sdk_model_default(monkeypatch):
    sent = []
    binding = _binding(monkeypatch, "openai-responses", "gpt-5", sent)
    features = RuntimeFeatures()
    features.model("default", binding)
    features.agent("default", model="default")
    async with Runtime.open("default-thinking", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get().run("hello", timeout_seconds=10)
    assert result.result.status is ExecutionStatus.SUCCEEDED
    assert "reasoning" not in sent[-1]
