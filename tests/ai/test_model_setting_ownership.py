#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One explicit owner per wire object without narrowing native extensions."""

import json

import httpx2
import pytest
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import CompactionPart, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import create_model
from linktools.ai.model import _anthropic, _openai
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from .test_model_protocols import response, stream_events


def _capture(monkeypatch, protocol, sent):
    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                  content=b"".join(stream_events(protocol)))
        return httpx2.Response(200, json=response(protocol))

    def client(**kwargs):
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(_anthropic if protocol == "anthropic" else _openai,
                        "DefaultAsyncHttpxClient", client)


async def _request(model, settings, streaming, parameters=None):
    messages = [ModelRequest(parts=[UserPromptPart("hello")])]
    parameters = parameters or ModelRequestParameters()
    if streaming:
        async with model.request_stream(messages, settings, parameters) as stream:
            async for _ in stream:
                pass
    else:
        await model.request(messages, settings, parameters)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("per_request", [False, True])
@pytest.mark.parametrize("protocol,limit", [
    ("openai", "max_completion_tokens"),
    pytest.param("openai", "max_tokens", id="chat-legacy-token-alias"),
    ("openai-responses", "max_output_tokens"),
    ("anthropic", "max_tokens"),
])
async def test_explicit_token_limit_and_raw_limit_cannot_compete(
    monkeypatch: pytest.MonkeyPatch, protocol: str, limit: str, streaming: bool, per_request: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    raw = {"extra_body": {limit: 4096}}
    binding = create_model(protocol=protocol, model="unregistered-model-alias", api_key="mock-only",
                           max_retries=0, model_settings={"max_tokens": 8192, **({} if per_request else raw)})
    async with binding.materialize() as model:
        with pytest.raises(AIError) as raised:
            await _request(model, raw if per_request else None, streaming)
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.retryable is False
    assert raised.value.safe_details == {"field": limit, "owner": "max_tokens"}
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol,settings", [
    ("anthropic", {"anthropic_effort": "low", "extra_body": {"output_config": {"format": {"type": "text"}}}}),
    ("anthropic", {"anthropic_effort": None, "extra_body": {"output_config": {"effort": "low"}}}),
    ("anthropic", {"anthropic_thinking": {"type": "disabled"}, "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 1024}}}),
    ("openai", {"openai_reasoning_effort": "low", "extra_body": {"reasoning_effort": "high"}}),
    ("openai-responses", {"openai_reasoning_effort": "low", "extra_body": {"reasoning": {"summary": "auto"}}}),
    ("openai-responses", {"openai_text_verbosity": "low", "extra_body": {"text": {"format": {"type": "text"}}}}),
])
async def test_provider_native_objects_are_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch, protocol: str, settings: dict, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    binding = create_model(protocol=protocol, model="unregistered-model-alias", api_key="mock-only",
                           max_retries=0, model_settings={"max_tokens": 8192, **settings})
    async with binding.materialize() as model:
        with pytest.raises(AIError):
            await _request(model, None, streaming)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol,limit", [
    ("openai", "max_completion_tokens"),
    ("openai-responses", "max_output_tokens"),
    ("anthropic", "max_tokens"),
])
async def test_raw_only_limits_and_unknown_objects_remain_open(
    monkeypatch: pytest.MonkeyPatch, protocol: str, limit: str, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    raw = {limit: 8192, "vendor_feature": {"enabled": False, "values": [None, 1, {"nested": "value"}]}}
    binding = create_model(protocol=protocol, model="unregistered-model-alias", api_key="mock-only",
                           max_retries=0, model_settings={"extra_body": raw})
    async with binding.materialize() as model:
        await _request(model, None, streaming)
    assert sent[-1][limit] == 8192
    assert sent[-1]["vendor_feature"] == raw["vendor_feature"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    {"thinking": {"type": "disabled"}},
    {"thinking": {"type": "enabled", "budget_tokens": 1024}},
    {"output_config": {"effort": "low", "vendor_option": False}},
])
async def test_runtime_thinking_is_fallback_not_a_second_native_owner(
    monkeypatch: pytest.MonkeyPatch, raw: dict,
) -> None:
    sent = []
    _capture(monkeypatch, "anthropic", sent)
    features = RuntimeFeatures()
    features.model("default", create_model(
        protocol="anthropic", model="unregistered-model-alias", api_key="mock-only", max_retries=0,
        model_settings={"max_tokens": 32768, "extra_body": raw},
    ))
    features.agent("default", model="default")
    features.capability(Thinking(effort="high"))
    async with (
        Persistence.open("native-owner", storage=PersistenceStorage.in_memory()) as persistence,
        Runtime.open(persistence=persistence, features=features) as runtime,
    ):
        result = await runtime.agents.get().run("hello", timeout_seconds=10)
    assert result.result.status is ExecutionStatus.SUCCEEDED
    for name, value in raw.items():
        assert sent[-1][name] == value


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol,input_field", [
    ("openai", "messages"), ("openai-responses", "input"), ("anthropic", "messages"),
])
@pytest.mark.parametrize("field,value", [
    ("model", "other-model"), ("tools", []), ("tool_choice", "none"), ("stream", False),
    ("input_field", []),
])
async def test_raw_cannot_replace_model_request_owned_fields(
    monkeypatch: pytest.MonkeyPatch, protocol: str, input_field: str, field: str, value: object, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    field = input_field if field == "input_field" else field
    binding = create_model(protocol=protocol, model="unregistered-model-alias", api_key="mock-only",
                           max_retries=0, model_settings={"extra_body": {field: value}})
    async with binding.materialize() as model:
        with pytest.raises(AIError):
            await _request(model, None, streaming)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol,model_name,wire_field", [
    ("openai", "gpt-4o", "response_format"),
    ("openai-responses", "gpt-5", "text"),
    ("anthropic", "claude-sonnet-4-5", "output_config"),
])
async def test_raw_cannot_replace_native_output_contract(
    monkeypatch: pytest.MonkeyPatch, protocol: str, model_name: str, wire_field: str, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    binding = create_model(protocol=protocol, model=model_name, api_key="mock-only", max_retries=0,
                           model_settings={"extra_body": {wire_field: {}}})
    parameters = ModelRequestParameters(
        output_mode="native",
        output_object=OutputObjectDefinition(name="answer", json_schema={"type": "object", "properties": {}}),
    )
    async with binding.materialize() as model:
        with pytest.raises(AIError):
            await _request(model, None, streaming, parameters)
    assert sent == []


@pytest.mark.parametrize("protocol,setting,wire_field", [
    ('anthropic', 'max_tokens', 'max_tokens'),
    ('anthropic', 'temperature', 'temperature'),
    ('anthropic', 'top_p', 'top_p'),
    ('anthropic', 'top_k', 'top_k'),
    ('anthropic', 'stop_sequences', 'stop_sequences'),
    ('anthropic', 'service_tier', 'service_tier'),
    ('anthropic', 'anthropic_service_tier', 'service_tier'),
    ('anthropic', 'anthropic_metadata', 'metadata'),
    ('anthropic', 'anthropic_thinking', 'thinking'),
    ('anthropic', 'anthropic_effort', 'output_config'),
    ('anthropic', 'anthropic_task_budget', 'output_config'),
    ('anthropic', 'anthropic_speed', 'speed'),
    ('anthropic', 'anthropic_cache', 'cache_control'),
    ('anthropic', 'anthropic_cache_diagnostics', 'diagnostics'),
    ('openai', 'temperature', 'temperature'),
    ('openai', 'top_p', 'top_p'),
    ('openai', 'parallel_tool_calls', 'parallel_tool_calls'),
    ('openai', 'service_tier', 'service_tier'),
    ('openai', 'openai_service_tier', 'service_tier'),
    ('openai', 'openai_user', 'user'),
    ('openai', 'openai_top_logprobs', 'top_logprobs'),
    ('openai', 'openai_store', 'store'),
    ('openai', 'openai_moderation', 'moderation'),
    ('openai', 'openai_prompt_cache_key', 'prompt_cache_key'),
    ('openai', 'openai_prompt_cache_retention', 'prompt_cache_retention'),
    ('openai', 'openai_prompt_cache_options', 'prompt_cache_options'),
    ('openai', 'stop_sequences', 'stop'),
    ('openai', 'seed', 'seed'),
    ('openai', 'presence_penalty', 'presence_penalty'),
    ('openai', 'frequency_penalty', 'frequency_penalty'),
    ('openai', 'logit_bias', 'logit_bias'),
    ('openai', 'openai_reasoning_effort', 'reasoning_effort'),
    ('openai', 'openai_logprobs', 'logprobs'),
    ('openai', 'openai_prediction', 'prediction'),
    ('openai', 'openai_text_verbosity', 'verbosity'),
    ('openai', 'max_tokens', 'max_completion_tokens'),
    ('openai-responses', 'temperature', 'temperature'),
    ('openai-responses', 'top_p', 'top_p'),
    ('openai-responses', 'parallel_tool_calls', 'parallel_tool_calls'),
    ('openai-responses', 'service_tier', 'service_tier'),
    ('openai-responses', 'openai_service_tier', 'service_tier'),
    ('openai-responses', 'openai_user', 'user'),
    ('openai-responses', 'openai_top_logprobs', 'top_logprobs'),
    ('openai-responses', 'openai_store', 'store'),
    ('openai-responses', 'openai_moderation', 'moderation'),
    ('openai-responses', 'openai_prompt_cache_key', 'prompt_cache_key'),
    ('openai-responses', 'openai_prompt_cache_retention', 'prompt_cache_retention'),
    ('openai-responses', 'openai_prompt_cache_options', 'prompt_cache_options'),
    ('openai-responses', 'max_tokens', 'max_output_tokens'),
    ('openai-responses', 'openai_reasoning_effort', 'reasoning'),
    ('openai-responses', 'openai_reasoning_mode', 'reasoning'),
    ('openai-responses', 'openai_reasoning_context', 'reasoning'),
    ('openai-responses', 'openai_reasoning_summary', 'reasoning'),
    ('openai-responses', 'openai_text_verbosity', 'text'),
    ('openai-responses', 'openai_truncation', 'truncation'),
    ('openai-responses', 'openai_context_management', 'context_management'),
    ('openai-responses', 'openai_background', 'background'),
    ('openai-responses', 'openai_logprobs', 'include'),
    ('openai-responses', 'openai_include_code_execution_outputs', 'include'),
    ('openai-responses', 'openai_include_web_search_sources', 'include'),
    ('openai-responses', 'openai_include_file_search_results', 'include'),
    ('openai-responses', 'openai_prompt_cache_diagnostics', 'prompt_cache_options'),

])
@pytest.mark.asyncio
async def test_each_native_setting_rejects_a_second_raw_object_owner(
    protocol: str, setting: str, wire_field: str,
) -> None:
    binding = create_model(protocol=protocol, model="unregistered-model-alias", api_key="mock-only",
                           model_settings={setting: None, "extra_body": {wire_field: {}}})
    async with binding.materialize() as model:
        with pytest.raises(AIError, match="same request object"):
            model.wrapped.prepare_request(None, ModelRequestParameters())


def test_sdk_setting_owners_have_explicit_protocol_classification() -> None:
    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIChatModelSettings, OpenAIResponsesModelSettings
    from linktools.ai.model._anthropic import _REQUEST_SETTING_FIELDS
    from linktools.ai.model._openai import _CHAT_REQUEST_SETTING_FIELDS, _RESPONSES_REQUEST_SETTING_FIELDS

    operational = {"timeout", "request_timeout", "max_retries", "extra_headers", "extra_body"}
    fallback = {"thinking", "cache"}
    request_owned = {"tool_choice", "citations"}
    cases = (
        (AnthropicModelSettings, _REQUEST_SETTING_FIELDS, {
            # These native settings change existing input/tool objects, which already have an owner.
            "parallel_tool_calls", "anthropic_cache_tool_definitions", "anthropic_cache_instructions",
            "anthropic_cache_messages", "anthropic_code_execution_tool_version", "anthropic_eager_input_streaming",
            "anthropic_container", "anthropic_context_management",  # Can also be derived from history.
            "anthropic_betas",  # Native HTTP header, not a JSON body field.
            "frequency_penalty", "logit_bias", "presence_penalty", "seed",  # Unsupported by this adapter.
        }),
        (OpenAIChatModelSettings, _CHAT_REQUEST_SETTING_FIELDS, {
            "max_tokens",  # Both native token aliases are checked explicitly.
            "openai_cache_instructions",  # Changes request messages.
            "openai_include_raw_annotations",  # Response parsing only.
            "openai_continuous_usage_stats",  # Streaming request owns stream_options.
            "top_k",  # Unsupported by this adapter.
        }),
        (OpenAIResponsesModelSettings, _RESPONSES_REQUEST_SETTING_FIELDS, {
            "openai_cache_instructions", "openai_native_tools", "openai_send_reasoning_ids",
            "openai_previous_response_id", "openai_conversation_id",
            "openai_include_raw_annotations",  # Response parsing only.
            # Unsupported inherited Chat settings do not own a Responses wire object.
            "openai_continuous_usage_stats", "openai_prediction",
            "frequency_penalty", "logit_bias", "presence_penalty", "seed", "stop_sequences", "top_k",
        }),
    )
    for sdk_settings, mapped, other in cases:
        assert set(sdk_settings.__annotations__) == set(mapped) | operational | fallback | request_owned | other


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "openai-responses"])
@pytest.mark.parametrize("settings", [
    {"max_retries": 1}, {"extra_body": {"model": "different-model"}},
])
async def test_token_counting_keeps_request_and_retry_owners(
    monkeypatch: pytest.MonkeyPatch, protocol: str, settings: dict,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    binding = create_model(protocol=protocol, model="fixture", api_key="mock-only")
    async with binding.materialize() as model:
        with pytest.raises(AIError) as raised:
            await model.count_tokens([ModelRequest(parts=[UserPromptPart("hello")])],
                                     settings, ModelRequestParameters())
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert sent == []


@pytest.mark.asyncio
async def test_runtime_reports_owner_conflict_as_nonretryable_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = []
    _capture(monkeypatch, "anthropic", sent)
    features = RuntimeFeatures()
    features.model("default", create_model(
        protocol="anthropic", model="fixture", api_key="mock-only", max_retries=3,
        model_settings={"anthropic_effort": "low", "extra_body": {"output_config": {"custom": "private-value"}}},
    ))
    features.agent("default", model="default")
    async with (
        Persistence.open("owner-conflict", storage=PersistenceStorage.in_memory()) as persistence,
        Runtime.open(persistence=persistence, features=features) as runtime,
    ):
        result = await runtime.agents.get().run("hello", timeout_seconds=10)
    assert result.result.status is ExecutionStatus.FAILED
    assert result.result.error_code == ErrorCode.MODEL_CONFIG_INVALID.value
    assert "private-value" not in str(result.result.safe_error_details)
    assert "private-value" not in str(result.result.error_diagnostics)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("supports_json_object", [False, True])
@pytest.mark.parametrize("protocol,wire_field,raw", [
    ("openai", "response_format", {"type": "text"}),
    ("openai-responses", "text", {"format": {"type": "text"}, "vendor_extension": False}),
])
async def test_prompted_output_owns_wire_format_only_when_profile_generates_it(
    monkeypatch: pytest.MonkeyPatch, protocol: str, wire_field: str, raw: dict,
    supports_json_object: bool, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    binding = create_model(protocol=protocol, model="gpt-4o", api_key="mock-only", max_retries=0,
                           model_settings={"extra_body": {wire_field: raw, "vendor_flag": False}})
    monkeypatch.setattr(type(binding), "_sdk_profile", lambda self: {
        "supports_json_object_output": supports_json_object,
    })
    parameters = ModelRequestParameters(
        output_mode="prompted",
        output_object=OutputObjectDefinition(name="answer", json_schema={"type": "object", "properties": {}}),
    )
    async with binding.materialize() as model:
        if supports_json_object:
            with pytest.raises(AIError) as raised:
                await _request(model, None, streaming, parameters)
            assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
            assert raised.value.safe_details == {"field": wire_field, "owner": "model_request"}
            assert sent == []
        else:
            await _request(model, None, streaming, parameters)
            assert sent[-1][wire_field] == raw
            assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("has_native_tool", [False, True])
async def test_chat_native_search_owns_options_only_when_selected(
    monkeypatch: pytest.MonkeyPatch, has_native_tool: bool, streaming: bool,
) -> None:
    from pydantic_ai.native_tools import WebSearchTool

    sent = []
    _capture(monkeypatch, "openai", sent)
    raw = {"search_context_size": "low", "vendor_option": False}
    binding = create_model(protocol="openai", model="gpt-4o-search-preview", api_key="mock-only",
                           max_retries=0, model_settings={"extra_body": {"web_search_options": raw}})
    parameters = ModelRequestParameters(native_tools=[WebSearchTool()] if has_native_tool else [])
    async with binding.materialize() as model:
        if has_native_tool:
            with pytest.raises(AIError) as raised:
                await _request(model, None, streaming, parameters)
            assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
            assert raised.value.safe_details == {"field": "web_search_options", "owner": "model_request"}
            assert sent == []
        else:
            await _request(model, None, streaming, parameters)
            assert sent[-1]["web_search_options"] == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("protocol,field", [
    ("anthropic", "container"), ("openai-responses", "previous_response_id"),
    ("openai-responses", "conversation"),
])
async def test_raw_cannot_replace_sdk_session_state(
    monkeypatch: pytest.MonkeyPatch, protocol: str, field: str, with_history: bool, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, protocol, sent)
    binding = create_model(protocol=protocol, model="fixture", api_key="mock-only", max_retries=0,
                           model_settings={"extra_body": {field: "raw-session"}})
    messages = [ModelRequest(parts=[UserPromptPart("next")])]
    if with_history:
        messages = [ModelRequest(parts=[UserPromptPart("first")]), ModelResponse(
            parts=[TextPart("answer")], provider_name="anthropic" if protocol == "anthropic" else "openai",
            provider_response_id="history-response", provider_details={
                "container_id": "history-container", "conversation_id": "history-conversation",
            },
        ), *messages]
    async with binding.materialize() as model:
        with pytest.raises(AIError) as raised:
            if streaming:
                async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                    async for _ in stream:
                        pass
            else:
                await model.request(messages, None, ModelRequestParameters())
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"field": field, "owner": "model_request"}
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("selection,expected", [(None, None), ("chosen-container", "chosen-container"), (False, None)])
async def test_anthropic_native_container_selection_retains_sdk_history_semantics(
    monkeypatch: pytest.MonkeyPatch, with_history: bool, selection: str | bool | None, expected: str | None,
) -> None:
    sent = []
    _capture(monkeypatch, "anthropic", sent)
    settings = {"extra_body": {"vendor_flag": False}}
    if selection is not None:
        settings["anthropic_container"] = selection
    binding = create_model(protocol="anthropic", model="fixture", api_key="mock-only", max_retries=0,
                           model_settings=settings)
    messages = [ModelRequest(parts=[UserPromptPart("next")])]
    if with_history:
        messages = [ModelRequest(parts=[UserPromptPart("first")]), ModelResponse(
            parts=[TextPart("answer")], provider_name="anthropic",
            provider_details={"container_id": "history-container"},
        ), *messages]
    async with binding.materialize() as model:
        await model.request(messages, None, ModelRequestParameters())
    assert sent[-1].get("container") == ("history-container" if with_history and selection is None else expected)
    assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,field", [
    ("openai_previous_response_id", "previous_response_id"), ("openai_conversation_id", "conversation"),
])
async def test_responses_native_session_selection_and_auto_history_remain_supported(
    monkeypatch: pytest.MonkeyPatch, setting: str, field: str,
) -> None:
    sent = []
    _capture(monkeypatch, "openai-responses", sent)
    binding = create_model(protocol="openai-responses", model="gpt-4o", api_key="mock-only", max_retries=0,
                           model_settings={setting: "selected-session", "extra_body": {"vendor_flag": False}})
    first_request = ModelRequest(parts=[UserPromptPart("first")])
    async with binding.materialize() as model:
        first_response = await model.request([first_request], None, ModelRequestParameters())
        assert sent[-1][field] == "selected-session"
        if field == "conversation":
            first_response.provider_details = {**(first_response.provider_details or {}), "conversation_id": "history-conversation"}
        await model.request([first_request, first_response, ModelRequest(parts=[UserPromptPart("next")])],
                            {setting: "auto"}, ModelRequestParameters())
    assert sent[-1][field] == (first_response.provider_response_id if field == "previous_response_id" else "history-conversation")
    assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("configuration", ["raw", "native", "default"])
async def test_anthropic_compaction_history_keeps_native_context_management_owner(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, with_history: bool, configuration: str,
) -> None:
    sent = []
    _capture(monkeypatch, "anthropic", sent)
    context_management = {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 100000}}]}
    settings = {"extra_body": {"vendor_flag": False}}
    if configuration == "raw":
        settings["extra_body"]["context_management"] = context_management
    elif configuration == "native":
        settings["anthropic_context_management"] = context_management
    binding = create_model(protocol="anthropic", model="claude-sonnet-4-6", api_key="mock-only", max_retries=0,
                           model_settings=settings)
    messages = [ModelRequest(parts=[UserPromptPart("next")])]
    if with_history:
        messages = [ModelRequest(parts=[UserPromptPart("first")]), ModelResponse(
            parts=[CompactionPart(content="summary", provider_name="anthropic")], provider_name="anthropic",
        ), *messages]
    async with binding.materialize() as model:
        async def send() -> None:
            if streaming:
                async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                    async for _ in stream:
                        pass
            else:
                await model.request(messages, None, ModelRequestParameters())
        if configuration == "raw":
            with pytest.raises(AIError) as raised:
                await send()
            assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
            assert raised.value.safe_details == {"field": "context_management", "owner": "model_request"}
            assert sent == []
        else:
            await send()
            expected = context_management if configuration == "native" else (
                {"edits": [{"type": "compact_20260112"}]} if with_history else None
            )
            assert sent[-1].get("context_management") == expected
            assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model_name,logprobs,conflict", [
    ("unregistered-model-alias", False, False),
    ("unregistered-model-alias", True, True),
    ("gpt-5", False, True),
])
async def test_responses_include_keeps_explicit_and_profile_derived_owners(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, model_name: str, logprobs: bool, conflict: bool,
) -> None:
    sent = []
    _capture(monkeypatch, "openai-responses", sent)
    settings = {"extra_body": {"include": ["vendor.extension"], "vendor_flag": False}}
    if logprobs:
        settings["openai_logprobs"] = True
    binding = create_model(protocol="openai-responses", model=model_name, api_key="mock-only", max_retries=0,
                           model_settings=settings)
    async with binding.materialize() as model:
        if conflict:
            with pytest.raises(AIError) as raised:
                await _request(model, None, streaming)
            assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
            assert raised.value.safe_details["field"] == "include"
            assert sent == []
        else:
            await _request(model, None, streaming)
            assert sent[-1]["include"] == ["vendor.extension"]
            assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_native_logprobs_preserves_encrypted_reasoning_include(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    sent = []
    _capture(monkeypatch, "openai-responses", sent)
    binding = create_model(protocol="openai-responses", model="gpt-5", api_key="mock-only", max_retries=0,
                           model_settings={"openai_logprobs": True, "extra_body": {"vendor_flag": False}})
    async with binding.materialize() as model:
        await _request(model, None, streaming)
    assert set(sent[-1]["include"]) >= {"reasoning.encrypted_content", "message.output_text.logprobs"}
    assert sent[-1]["vendor_flag"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("per_request", [False, True])
async def test_chat_stream_options_are_owned_only_during_streaming(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, per_request: bool,
) -> None:
    sent = []
    _capture(monkeypatch, "openai", sent)
    raw = {"extra_body": {"stream_options": {"include_usage": False, "vendor_flag": True}}}
    binding = create_model(protocol="openai", model="fixture", api_key="mock-only", max_retries=0,
                           model_settings={} if per_request else raw)
    async with binding.materialize() as model:
        if streaming:
            with pytest.raises(AIError) as raised:
                await _request(model, raw if per_request else None, streaming)
            assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
            assert raised.value.safe_details == {"field": "stream_options", "owner": "model_request"}
            assert sent == []
        else:
            await _request(model, raw if per_request else None, streaming)
            assert sent[-1]["stream_options"] == raw["extra_body"]["stream_options"]


@pytest.mark.asyncio
@pytest.mark.parametrize("continuous", [False, True])
async def test_chat_native_stream_usage_options_remain_supported(
    monkeypatch: pytest.MonkeyPatch, continuous: bool,
) -> None:
    sent = []
    _capture(monkeypatch, "openai", sent)
    binding = create_model(protocol="openai", model="fixture", api_key="mock-only", max_retries=0,
                           model_settings={"openai_continuous_usage_stats": continuous})
    async with binding.materialize() as model:
        await _request(model, None, True)
    assert sent[-1]["stream_options"] == {"include_usage": True, **({"continuous_usage_stats": True} if continuous else {})}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_raw_parallel_calls_may_override_output_only_default(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    from pydantic_ai.tools import ToolDefinition

    sent = []
    _capture(monkeypatch, "openai", sent)
    binding = create_model(protocol="openai", model="fixture", api_key="mock-only", max_retries=0,
                           model_settings={"extra_body": {"parallel_tool_calls": True}})
    parameters = ModelRequestParameters(output_mode="tool", allow_text_output=False,
                                        output_tools=[ToolDefinition(name="finish", parameters_json_schema={"type": "object", "properties": {}})])
    async with binding.materialize() as model:
        await _request(model, None, streaming, parameters)
    assert sent[-1]["parallel_tool_calls"] is True
