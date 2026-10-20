#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP body recording is opt-in at capture and remains separate from attempt metadata."""

from pathlib import Path

import httpx2
import pytest
from pydantic_ai.capabilities import CombinedCapability, WrapperCapability

from linktools.ai.capability import CheckpointCapability, RuntimeFeatures, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from linktools.ai.model import _openai
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime._trace_capture import record_http_bodies_enabled
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import _CURRENT_CODEC, _decode_domain, _encode_persisted_domain
from linktools.ai.runtime.state._contracts import ModelClientRequestRef
from linktools.ai.runtime.state._model_client import prepare_client_requests, resolve_client_requests

from .test_model_client_request_capture import _model, _registry, _response
from .test_model_client_response_capture import _invoke


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
async def test_http_policy_gates_body_observation_before_stream_capture(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, enabled: bool,
) -> None:
    if not enabled:
        def forbidden_buffer(*args: object, **kwargs: object) -> object:
            pytest.fail('disabled policy must not construct a response-body capture buffer')
        monkeypatch.setattr(_openai, '_CapturedChatResponseStream', forbidden_buffer)
    model, _clients = _model(monkeypatch, _response)
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=enabled) as capture:
            await _invoke(model, streaming)
    assert len(capture.requests) == 1
    attempt = capture.requests[0]
    assert attempt.record_http_bodies is enabled
    assert (attempt.body is not None) is enabled
    assert attempt.capture_error is None
    assert attempt.response is not None and attempt.response.status_code == 200
    assert (attempt.response.body is not None) is enabled
    assert attempt.response.capture_error is None
    assert attempt.response.format == ('sse' if streaming else 'json')


@pytest.mark.asyncio
async def test_disabled_capture_does_not_read_request_body() -> None:
    request = httpx2.Request('POST', 'https://capture.invalid/chat/completions', content=iter([b'body']))
    with ModelClientRequestRecorder() as capture:
        await _openai._capture_chat_request(request)
    assert capture.requests == (ModelClientRequest('openai-chat', None, record_http_bodies=False),)
    with pytest.raises(httpx2.RequestNotRead):
        _ = request.content


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('domain', [RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT])
async def test_attempt_policy_and_metadata_roundtrip_without_copying_bodies_to_checkpoint(
    enabled: bool, domain: RuntimeDomain,
) -> None:
    source = ModelClientRequest(
        'openai-chat', b'request' if enabled else None, record_http_bodies=enabled,
        response=ModelClientResponse(b'response' if enabled else None, 200, 'json', None, enabled),
    )
    refs = await prepare_client_requests((source,), domain)
    assert refs is not None
    encoded = _encode_persisted_domain(refs[0])
    restored = _decode_domain(encoded, ModelClientRequestRef, _CURRENT_CODEC, persisted=True)
    recovered = await resolve_client_requests((restored,), domain)
    assert recovered is not None
    value = recovered[0]
    assert value.record_http_bodies is enabled
    assert value.response is not None and value.response.status_code == 200
    assert value.capture_error is value.response.capture_error is None
    if domain is RuntimeDomain.TRACE:
        assert value == source
    else:
        assert value.body is value.response.body is None


def test_trace_policy_is_read_through_native_wrappers_without_defaulting_on() -> None:
    assert TraceCapability().record_http_bodies is False
    assert not record_http_bodies_enabled([TraceCapability()])
    enabled = TraceCapability(record_http_bodies=True)
    assert record_http_bodies_enabled([WrapperCapability(enabled)])
    assert record_http_bodies_enabled([WrapperCapability(CombinedCapability([enabled]))])


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
async def test_runtime_policy_survives_checkpoint_and_cold_trace_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    models, _clients = _registry(monkeypatch, _response)
    features = RuntimeFeatures('body-policy')
    features.agent('default', model='default')
    features.capability(WrapperCapability(TraceCapability(record_http_bodies=enabled), id='wrapped-trace'))
    features.capability(CheckpointCapability())
    root = tmp_path / 'state'
    async with Runtime.open('body-policy', models=models, storage=RuntimeStorage.filesystem(root),
                            features=features) as runtime:
        execution = await runtime.agents.get().start('question')
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = execution.execution_id
        principal = runtime.default_principal
    async with RuntimeHistory.open('body-policy', storage=RuntimeStorage.filesystem(root), features=features) as history:
        items = await history.model_interactions(execution_id, principal=principal, include_client_requests=True)
        attempt = items.items[0].client_requests[0]
        assert attempt.record_http_bodies is enabled
        assert (attempt.body is not None) is enabled
        assert attempt.response is not None
        assert (attempt.response.body is not None) is enabled
        assert attempt.capture_error is attempt.response.capture_error is None
        assert items.items[0].usage is not None
