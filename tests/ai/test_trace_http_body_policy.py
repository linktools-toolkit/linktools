#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""HTTP body recording is opt-in at capture and remains separate from attempt metadata."""
from ._runtime_test_helpers import model_features

from pathlib import Path

import httpx2
import pytest
from pydantic_ai.capabilities import CombinedCapability, WrapperCapability

from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from linktools.ai.model import _http, _openai
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._trace_capture import record_http_bodies_enabled
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import _CURRENT_CODEC, _decode_domain, _encode_persisted_domain
from linktools.ai.persistence.contracts import ModelClientRequestRef
from linktools.ai.persistence._model_client import prepare_client_requests, resolve_client_requests

from .test_model_client_request_capture import _model, _binding, _response
from .test_model_client_response_capture import _invoke


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
async def test_http_policy_gates_body_observation_before_stream_capture(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, enabled: bool,
) -> None:
    if not enabled:
        def forbidden_parse(*args: object, **kwargs: object) -> object:
            pytest.fail('disabled body policy must not parse response content')
        monkeypatch.setattr(_http, 'EventSource', forbidden_parse)
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
        await _http.capture_request(request)
    attempt, = capture.requests
    assert attempt.api == 'openai' and attempt.body is None and not attempt.record_http_bodies
    assert attempt.diagnostics is not None and set(attempt.diagnostics) == {'started_at'}
    with pytest.raises(httpx2.RequestNotRead):
        _ = request.content


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('domain', [PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT])
async def test_attempt_policy_and_metadata_roundtrip_without_copying_bodies_to_checkpoint(
    enabled: bool, domain: PersistenceDomain,
) -> None:
    source = ModelClientRequest(
        'openai', b'request' if enabled else None, record_http_bodies=enabled,
        response=ModelClientResponse(b'response' if enabled else None, 200, 'json', None, enabled),
        diagnostics={'started_at': '2026-01-01T00:00:00+00:00', 'received_bytes': 8,
                     'read_outcome': 'error', 'read_error_type': 'httpx2.RemoteProtocolError'},
    )
    refs = await prepare_client_requests((source,), domain)
    assert refs is not None
    encoded = _encode_persisted_domain(refs[0])
    restored = _decode_domain(encoded, ModelClientRequestRef, _CURRENT_CODEC, persisted=True)
    recovered = await resolve_client_requests((restored,), domain)
    assert recovered is not None
    value = recovered[0]
    assert value.record_http_bodies is enabled
    assert value.diagnostics == source.diagnostics
    assert value.response is not None and value.response.status_code == 200
    assert value.capture_error is value.response.capture_error is None
    if domain is PersistenceDomain.TRACE:
        assert value == source
    else:
        assert value.body is value.response.body is None


@pytest.mark.parametrize("missing", (False, True))
def test_missing_attempt_diagnostics_remain_unknown_on_existing_wire(missing: bool) -> None:
    encoded = _encode_persisted_domain(ModelClientRequestRef(
        "openai", None, record_http_bodies=False, diagnostics={"received_bytes": 7},
    ))
    if missing:
        del encoded["fields"]["diagnostics"]
    else:
        encoded["fields"]["diagnostics"] = None
    restored = _decode_domain(encoded, ModelClientRequestRef, _CURRENT_CODEC, persisted=True)
    assert restored.diagnostics is None


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
    models, _clients = _binding(monkeypatch, _response)
    features = RuntimeFeatures('body-policy')
    features.agent('default', model='default')
    features.capability(WrapperCapability(TraceCapability(record_http_bodies=enabled), id='wrapped-trace'))
    features.capability(CheckpointCapability())
    root = tmp_path / 'state'
    async with Persistence.open('body-policy', storage=PersistenceStorage.filesystem(root)) as persistence_126, Runtime.open(persistence=persistence_126, features=model_features(models, features)) as runtime:
        execution = await runtime.agents.get().start('question')
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = execution.execution_id
        principal = runtime.default_principal
    async with Persistence.open('body-policy', storage=PersistenceStorage.filesystem(root), tenant_id="default", read_only=True) as history_persistence_134, History.open(persistence=history_persistence_134, authorization=features.authorization) as history:
        items = await history.model_interactions(execution_id, principal=principal, include_client_requests=True)
        attempt = items.items[0].client_requests[0]
        assert attempt.record_http_bodies is enabled
        assert (attempt.body is not None) is enabled
        assert attempt.response is not None
        assert (attempt.response.body is not None) is enabled
        assert attempt.capture_error is attempt.response.capture_error is None
        assert attempt.diagnostics is not None and "headers_received_ns" in attempt.diagnostics
        assert items.items[0].usage is not None
