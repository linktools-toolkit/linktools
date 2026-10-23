#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SDK protocols retain their own request and recovery semantics."""

import json
from collections.abc import Callable

import httpx2
import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ThinkingPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from linktools.ai.errors import AIError
from linktools.ai.agent._model_resolver import _ModelResolver
from linktools.ai.model import ModelClientRequestRecorder, ModelBinding, create_model, validate_model_input
from linktools.ai.model import _anthropic, _openai
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages


PROTOCOLS = [('openai', 'openai', 'gpt-4o'),
             ('openai_responses', 'openai-responses', 'gpt-5'),
             ('anthropic', 'anthropic', 'claude-sonnet-4-5')]


def model_binding(monkeypatch: pytest.MonkeyPatch, kind: str, model: str, handler: Callable,
             **options: object) -> tuple[ModelBinding, list[httpx2.AsyncClient]]:
    module = _anthropic if kind == 'anthropic' else _openai
    clients = []

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(module, 'DefaultAsyncHttpxClient', client)
    return create_model(protocol=kind.replace('_', '-'), model=model, api_key='fixture-secret',
        base_url='https://model.invalid/v1', max_retries=options.pop('max_retries', 0),
        retry_delay=0, max_tokens=128, **options), clients


def response(api: str, *, tool: bool = False) -> dict:
    if api == 'openai':
        message = {'role': 'assistant', 'content': None if tool else 'answer'}
        if tool:
            message.update(reasoning_content='returned thinking', tool_calls=[{
                'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":1}'}}])
        return {'id': 'chat-1', 'object': 'chat.completion', 'created': 1, 'model': 'gpt-4o',
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if tool else 'stop'}]}
    if api == 'openai-responses':
        output = [{'type': 'reasoning', 'id': 'rs-1', 'summary': [{'type': 'summary_text', 'text': 'brief summary'}],
                   'encrypted_content': 'opaque-reasoning'},
                  {'type': 'function_call', 'id': 'fc-1', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{"key":1}', 'status': 'completed'}] if tool else [
                  {'type': 'message', 'id': 'msg-1', 'role': 'assistant', 'status': 'completed',
                   'content': [{'type': 'output_text', 'text': 'answer', 'annotations': []}]}]
        return {'id': 'resp-1', 'object': 'response', 'created_at': 1, 'model': 'gpt-5', 'status': 'completed',
                'output': output, 'parallel_tool_calls': True, 'tool_choice': 'auto', 'tools': []}
    content = [{'type': 'thinking', 'thinking': 'returned thinking', 'signature': 'signed'},
               {'type': 'redacted_thinking', 'data': 'opaque'},
               {'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {'key': 1}}] if tool else [{'type': 'text', 'text': 'answer'}]
    return {'id': 'msg-1', 'type': 'message', 'role': 'assistant', 'model': 'claude-sonnet-4-5',
            'content': content, 'stop_reason': 'tool_use' if tool else 'end_turn', 'stop_sequence': None,
            'usage': {'input_tokens': 5, 'output_tokens': 3}}


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
async def test_native_tool_turn_survives_durable_restore_and_client_reentry(monkeypatch, kind, api, model_name):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=response(api, tool=len(sent) == 1))

    profile = {'openai_chat_thinking_field': 'reasoning_content', 'openai_chat_send_back_thinking_parts': 'field'} if kind == 'openai' else None
    models, clients = model_binding(monkeypatch, kind, model_name, handler, profile=profile)
    resolver = _ModelResolver({"default": models})
    binding = resolver.resolve('default')
    contract = json.loads(json.dumps(dict(binding.contract)))
    restored = resolver.resolve('default').materialize()
    params = ModelRequestParameters(function_tools=[ToolDefinition(name='lookup', parameters_json_schema={'type': 'object', 'properties': {'key': {'type': 'integer'}}})])
    initial = [ModelRequest(parts=[UserPromptPart('find')])]
    with ModelClientRequestRecorder(record_http_bodies=True) as capture:
        async with restored:
            first = await restored.request(initial, None, params)
        assert clients[-1].is_closed
        assert any(isinstance(part, ThinkingPart) for part in first.parts)
        call = next(part for part in first.parts if isinstance(part, ToolCallPart))
        history = decode_model_messages(encode_model_messages(tuple(initial + [first,
            ModelRequest(parts=[ToolReturnPart(call.tool_name, 'found', tool_call_id=call.tool_call_id)])])))
        async with restored:
            final = await restored.request(list(history), None, params)
    assert final.parts[0].content == 'answer'
    assert all(client.is_closed for client in clients) and len(clients) == 2
    assert [attempt.api for attempt in capture.requests] == [api, api]
    assert all(attempt.response.complete for attempt in capture.requests)
    assert 'fixture-secret' not in repr(capture.requests)
    outgoing = json.dumps(sent[1])
    assert 'found' in outgoing
    if kind == 'openai_responses':
        assert 'opaque-reasoning' in outgoing and 'brief summary' in outgoing
        assert 'input' in sent[1] and 'messages' not in sent[1]
    elif kind == 'anthropic':
        assert 'signed' in outgoing and 'opaque' in outgoing
    else:
        assert 'returned thinking' in outgoing


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
async def test_public_registration_matches_model_type_and_restores(kind, api, model_name):
    from linktools.ai.runtime import RuntimeFeatures
    binding = create_model(protocol=api, model=model_name)
    features = RuntimeFeatures()
    features.model("selected", binding)
    sources = (await features.capture()).models
    resolver = _ModelResolver(sources)
    actual = sources["selected"].contract
    assert actual == binding.contract
    assert actual.get("api", "openai") == api
    assert resolver.resolve("selected").contract == actual
    if api == "openai":
        assert "api" not in actual
    elif api == "anthropic":
        assert resolver.resolve("selected").contract == actual


def test_profile_overrides_are_explicit_frozen_and_contract_owned():
    expected = {'provider': 'openai', 'model_identity': 'openai:same', 'settings': {}, 'input_mime_types': []}
    assert create_model(protocol="openai", model='same').contract == expected
    assert create_model(protocol="openai", model='same', profile={}).contract == expected
    declaration = {'supports_thinking': True, 'openai_chat_thinking_field': 'reasoning_content'}
    models = create_model(protocol="openai", model='same', profile=declaration)
    binding = models
    declaration['supports_thinking'] = False
    assert binding.contract['profile']['supports_thinking'] is True
    exposed = binding.contract
    exposed['profile']['supports_thinking'] = False
    assert binding.contract['profile']['supports_thinking'] is True
    assert create_model(protocol="openai-responses", model='same').contract != expected
    models = create_model(protocol='openai', model='same', profile={'supports_thinking': False})
    assert _ModelResolver({'default': models}).resolve('default') is models
    for invalid in ({'supports_thinking': 'yes'}, {'unknown_capability': True}, {'json_schema_transformer': object}, {'supports_thinking': float('nan')}):
        with pytest.raises((TypeError, ValueError)):
            create_model(protocol="openai", model='same', profile=invalid)


@pytest.mark.parametrize('kind,accepted,rejected', [
    ('openai', 'audio/wav', 'video/mp4'),
    ('openai_responses', 'application/pdf', 'audio/wav'),
    ('anthropic', 'text/plain', 'audio/wav'),
])
def test_native_protocol_input_encoding_is_distinct(kind, accepted, rejected):
    binding = create_model(protocol=kind.replace('_', '-'), model='unknown', input_types=['audio', 'video', 'document'])
    validate_model_input(binding, accepted)
    with pytest.raises(AIError):
        validate_model_input(binding, rejected)
    chat = create_model(protocol="openai", model='unknown', input_types=['document'], profile={'openai_chat_supports_document_input': False})
    with pytest.raises(AIError):
        validate_model_input(chat, 'application/pdf')


def stream_events(api: str) -> list[bytes]:
    if api == 'openai':
        from .test_model_client_response_capture import _EVENT, _DONE
        return [_EVENT, _DONE]
    if api == 'openai-responses':
        final = response(api)
        events = [
            {'type': 'response.created', 'response': {**final, 'status': 'in_progress', 'output': []}},
            {'type': 'response.output_item.added', 'output_index': 0, 'item': {**final['output'][0], 'content': []}},
            {'type': 'response.content_part.added', 'output_index': 0, 'content_index': 0, 'item_id': 'msg-1', 'part': {'type': 'output_text', 'text': '', 'annotations': []}},
            {'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 0, 'item_id': 'msg-1', 'delta': 'answer'},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': final['output'][0]},
            {'type': 'response.completed', 'response': final},
        ]
    else:
        final = response(api)
        events = [
            {'type': 'message_start', 'message': {**final, 'content': [], 'stop_reason': None}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'answer'}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 3}},
            {'type': 'message_stop'},
        ]
    return [('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode() for event in events]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
async def test_native_stream_observation_and_http_retry_are_not_multiplied(monkeypatch, kind, api, model_name):
    from .test_model_client_response_capture import _Stream, _invoke
    source = _Stream(stream_events(api))
    sent = []

    def handler(request):
        sent.append(request.content)
        if len(sent) == 1:
            return httpx2.Response(429, json={'error': {'type': 'rate_limit_error', 'message': 'retry'}})
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=source)

    models, clients = model_binding(monkeypatch, kind, model_name, handler, max_retries=1)
    model = models.materialize()
    with ModelClientRequestRecorder(record_http_bodies=True) as capture:
        async with model:
            await _invoke(model, True)
    assert len(sent) == 2 and len(capture.requests) == 2
    assert capture.requests[0].response.status_code == 429
    assert capture.requests[1].response.complete
    assert capture.requests[1].response.body == b''.join(source.received)
    assert source.closes == 1 and all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
async def test_native_partial_stream_is_not_replayed(monkeypatch, kind, api, model_name):
    from .test_model_client_response_capture import _Stream, _invoke
    from pydantic_ai.exceptions import ModelAPIError
    source = _Stream(stream_events(api)[:-1] + [httpx2.ReadError('stream interrupted')])
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=source)

    models, _ = model_binding(monkeypatch, kind, model_name, handler, max_retries=1)
    model = models.materialize()
    with ModelClientRequestRecorder(record_http_bodies=True) as capture:
        async with model:
            with pytest.raises((ModelAPIError, httpx2.ReadError)):
                await _invoke(model, True)
    assert len(sent) == 1
    assert capture.requests[0].response.body == b''.join(source.received)
    assert not capture.requests[0].response.complete


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
async def test_native_attachments_are_encoded_and_bodies_stay_off_by_default(monkeypatch, kind, api, model_name):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=response(api))

    models, _ = model_binding(monkeypatch, kind, model_name, handler, input_types=['image'])
    model = models.materialize()
    with ModelClientRequestRecorder() as capture:
        async with model:
            await model.request([ModelRequest(parts=[UserPromptPart(['image', BinaryContent(b'fixture-image', media_type='image/png')])])], None, ModelRequestParameters())
    assert 'Zml4dHVyZS1pbWFnZQ==' in json.dumps(sent[0])
    assert capture.requests[0].api == api
    assert capture.requests[0].body is None and capture.requests[0].response.body is None

@pytest.mark.asyncio
@pytest.mark.parametrize("factory,api,name", PROTOCOLS)
async def test_native_structured_output_uses_sdk_adapter(monkeypatch, factory, api, name):
    from pydantic import BaseModel
    from pydantic_ai import Agent, NativeOutput
    class Answer(BaseModel):
        value: int
    def handler(request):
        body = response(api, tool=False)
        if api == "openai": body["choices"][0]["message"]["content"] = '{"value":7}'
        elif api == "openai-responses": body["output"][0]["content"][0]["text"] = '{"value":7}'
        else: body["content"][0]["text"] = '{"value":7}'
        return httpx2.Response(200, json=body)
    models, clients = model_binding(monkeypatch, factory, name, handler)
    async with Agent(models.materialize(), output_type=NativeOutput(Answer)) as agent:
        result = await agent.run("Return seven")
    assert result.output.value == 7
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory,api,name", PROTOCOLS)
async def test_native_cancelled_stream_preserves_partial_observation(monkeypatch, factory, api, name):
    import asyncio
    ready = asyncio.Event()
    class Paused(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield stream_events(api)[0]
            ready.set()
            await asyncio.Event().wait()
    models, clients = model_binding(monkeypatch, factory, name,
        lambda request: httpx2.Response(200, headers={"content-type":"text/event-stream"}, stream=Paused()))
    async def consume(model):
        async with model.request_stream([ModelRequest(parts=[UserPromptPart("Hello")])], None, ModelRequestParameters()) as stream:
            async for _ in stream: pass
    async with models.materialize() as model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            task = asyncio.create_task(consume(model))
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        assert not recorder.requests[0].response.complete
    assert len(recorder.requests) == 1 and all(client.is_closed for client in clients)


def test_thinking_control_uses_frozen_effective_profile_without_connections_is_pure(monkeypatch):
    from linktools.ai.model import model_thinking_control
    binding = create_model(protocol="openai", model="fixture", api_key="secret",
                           profile={"supports_thinking": True})
    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", lambda **kwargs: pytest.fail("capability read created client"))
    assert model_thinking_control(binding)["mode"] == "controllable"
    resolver = _ModelResolver({"default": binding, "alias": binding})
    assert model_thinking_control(resolver.resolve("alias")) == model_thinking_control(binding)
    always = create_model(protocol="anthropic", model="fixture", profile={"thinking_always_enabled":True,"thinking_enabled_by_default":True})
    assert model_thinking_control(always)["mode"] == "always_on"
    assert model_thinking_control(create_model(protocol="openai", model="unknown"))["mode"] == "undeclared"


@pytest.mark.asyncio
@pytest.mark.parametrize("factory,api,name", PROTOCOLS)
async def test_runtime_web_profile_and_durable_model_content(monkeypatch, tmp_path, factory, api, name):
    from linktools.ai.capability import TraceCapability, ExecutionCapability
    from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeFeatures
    from linktools.ai.web import create_app
    from .test_web_console_viewers import _client
    from .test_model_client_response_capture import _Stream
    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type":"text/event-stream"}, stream=_Stream(stream_events(api)))
        return httpx2.Response(200, json=response(api))
    models, clients = model_binding(monkeypatch, factory, name, handler)
    features = RuntimeFeatures("app")
    features.model("default", models)
    features.agent("old", model="default", allow_tools=())
    features.capability(TraceCapability(), scope="runtime")
    features.capability(ExecutionCapability(), scope="runtime")
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    async with Runtime.open("default", features=features, storage=storage) as runtime:
        session = await runtime.agents.get("old").create_session("session")
        async with _client(create_app(runtime=runtime)) as http:
            meta = await http.get("/api/session", params={"session_id":"session","include_timeline":"false"})
            assert meta.status_code == 200, meta.text
            control = meta.json()["thinking_control"]
            assert control["mode"] in {"controllable", "always_on", "undeclared"}
            execution = await session.start("Hello", thinking=control["mode"] == "always_on")
            result = await execution.wait()
            assert result.result.status.value == "SUCCEEDED", result
            content = await http.get(f"/api/executions/{execution.execution_id}/models", params={"include_content":"true","include_client_requests":"true"})
            assert content.status_code == 200, content.text
            items = content.json()["items"]
            assert items and items[0]["content_included"] and items[0]["request"]
            assert items[0]["client_requests"][0]["api"] == api
            assert items[0]["client_requests"][0]["body"] is None
    assert all(client.is_closed for client in clients)
    # Historical sessions remain readable when their agent is no longer registered.
    replacement = RuntimeFeatures("app")
    replacement.model("default", models)
    replacement.agent("replacement", model="default", allow_tools=())
    replacement.capability(TraceCapability(), scope="runtime")
    replacement.capability(ExecutionCapability(), scope="runtime")
    async with Runtime.open("default", features=replacement, storage=storage) as runtime:
        async with _client(create_app(runtime=runtime)) as http:
            meta = await http.get("/api/session", params={"session_id":"session","include_timeline":"false"})
            assert meta.status_code == 200, meta.text
            assert meta.json()["thinking_control"]["mode"] == "unknown"


@pytest.mark.parametrize('kind,api,model_name', PROTOCOLS)
def test_generic_factory_forwards_all_options(monkeypatch, kind, api, model_name):
    from linktools.ai.model import _factory
    module = _anthropic if api == "anthropic" else _openai
    name = {"openai": "_OpenAIModelBinding", "openai-responses": "_OpenAIResponsesModelBinding", "anthropic": "_AnthropicModelBinding"}[api]
    sentinel, received = object(), {}
    def construct(**kwargs):
        received.update(kwargs)
        return sentinel
    monkeypatch.setattr(module, name, construct)
    options = dict(model=model_name, input_types=("image",), profile={"supports_thinking": True},
                   base_url="https://model.invalid/v1", api_key="fixture-secret", timeout=9,
                   max_retries=3, retry_delay=0.2, max_tokens=99)
    assert create_model(protocol=api, **options) is sentinel
    assert received == options


@pytest.mark.parametrize('protocol', ['', 'openai-chat', 'anthropic-messages', 'unknown-secret', None, {}])
def test_generic_factory_rejects_invalid_protocol_without_echoing_values(protocol):
    from linktools.ai.errors import ErrorCode
    with pytest.raises(AIError) as caught:
        create_model(protocol=protocol, model='fixture-model', api_key='fixture-secret')
    assert caught.value.code == ErrorCode.MODEL_CONFIG_INVALID
    assert caught.value.safe_details == {"reason": "unsupported_model_protocol"}
    assert 'fixture-secret' not in str(caught.value)
    assert 'unknown-secret' not in str(caught.value)
