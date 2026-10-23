#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical request body omission preserves dispatches and passive diagnostics."""

import asyncio
import json

import httpx2
import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from linktools.ai.model import ModelClientRequestRecorder
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ModelClientRequestRef
from linktools.ai.runtime.state._model_client import prepare_client_requests, resolve_client_requests

from .test_model_client_request_capture import _model, _request, _response
from .test_model_client_response_capture import _DONE, _EVENT, _JSON, _Stream, _invoke


_REASON = "referenced_content_lifecycle"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("configured_capture", (False, True))
async def test_body_omission_covers_sdk_retries_without_changing_transport_observations(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, configured_capture: bool,
) -> None:
    sent: list[bytes] = []
    sources: list[_Stream] = []
    failure = b'{"error":{"message":"retry"}}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        retry = len(sent) == 1
        source = _Stream([failure if retry else _EVENT + _DONE if streaming else _JSON])
        sources.append(source)
        return httpx2.Response(
            429 if retry else 200, stream=source,
            headers={"content-type": "text/event-stream" if streaming and not retry else "application/json"},
        )

    model, _clients = _model(monkeypatch, handler, retries=1)
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=configured_capture) as recorder:
            ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            await _invoke(model, streaming)

    assert recorder.record_http_bodies is configured_capture
    assert recorder.body_omission_reason == _REASON
    assert len(sent) == len(recorder.requests) == 2
    assert all(json.loads(body)["messages"][0]["content"] == "prompt" for body in sent)
    assert [attempt.response.status_code for attempt in recorder.requests] == [429, 200]
    for attempt, source in zip(recorder.requests, sources):
        assert attempt.record_http_bodies is False
        assert attempt.body is attempt.response.body is None
        assert attempt.capture_error is attempt.response.capture_error is None
        assert attempt.response.complete is False
        diagnostics = attempt.diagnostics
        assert diagnostics["body_omission_reason"] == _REASON
        assert diagnostics["received_bytes"] == sum(map(len, source.received))
        assert diagnostics["received_chunks"] == len(source.received)
        assert diagnostics["headers_received_ns"] <= diagnostics["first_chunk_ns"]
        assert diagnostics["close_started_ns"] <= diagnostics["close_finished_ns"]
        assert "protocol_terminal" not in diagnostics
        assert source.closes == 1

    prepared = await prepare_client_requests(recorder.requests, RuntimeDomain.TRACE)
    restored = decode_domain(encode_domain(prepared), tuple[ModelClientRequestRef, ...])
    assert await resolve_client_requests(restored, RuntimeDomain.TRACE) == recorder.requests
    assert all(attempt.body is attempt.response.body is None for attempt in restored)


@pytest.mark.asyncio
async def test_body_omission_is_owned_by_nested_and_concurrent_logical_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)

    async def child(label: str, omit: bool) -> ModelClientRequestRecorder:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            if omit:
                ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            await asyncio.sleep(0)
            await _request(model, label)
        return recorder

    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as outer:
            await _request(model, "before")
            with ModelClientRequestRecorder(record_http_bodies=True) as inner:
                ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
                await asyncio.create_task(_request(model, "inherited"))
                children = await asyncio.gather(child("ordinary", False), child("omitted", True))
                with ModelClientRequestRecorder(record_http_bodies=True) as nested:
                    await _request(model, "nested")
                await _request(model, "inner")
            await _request(model, "after")
        with ModelClientRequestRecorder(record_http_bodies=True) as following:
            await _request(model, "following")

    for recorder, labels in ((outer, ["before", "after"]), (nested, ["nested"]),
                             (children[0], ["ordinary"]), (following, ["following"])):
        assert [json.loads(attempt.body)["messages"][0]["content"] for attempt in recorder.requests] == labels
        assert all(attempt.response.body is not None for attempt in recorder.requests)
        assert all("body_omission_reason" not in attempt.diagnostics for attempt in recorder.requests)
    assert len(inner.requests) == 2 and len(children[1].requests) == 1
    for recorder in (inner, children[1]):
        assert recorder.record_http_bodies is True
        assert all(attempt.body is attempt.response.body is None for attempt in recorder.requests)
        assert all(attempt.diagnostics["body_omission_reason"] == _REASON for attempt in recorder.requests)


@pytest.mark.asyncio
async def test_body_omission_rejects_calls_after_dispatch_without_relabeling_retained_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            await _request(model, "already dispatched")
            before = recorder.requests
            with pytest.raises(RuntimeError, match="before the first dispatch"):
                ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            assert recorder.requests == before
            await _request(model, "still configured")
    assert all(attempt.record_http_bodies for attempt in recorder.requests)
    assert all(attempt.body is not None and attempt.response.body is not None for attempt in recorder.requests)
    assert all("body_omission_reason" not in attempt.diagnostics for attempt in recorder.requests)


@pytest.mark.asyncio
async def test_body_omission_is_idempotent_across_dispatches_and_cannot_change_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            assert recorder.body_omission_reason is None
            ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            with pytest.raises(RuntimeError, match="reason cannot change"):
                ModelClientRequestRecorder.omit_current_http_bodies("different_lifecycle")
            await _request(model, "token counting")
            ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            await _request(model, "actual request")
            with pytest.raises(RuntimeError, match="reason cannot change"):
                ModelClientRequestRecorder.omit_current_http_bodies("different_lifecycle")
            with pytest.raises(AttributeError):
                recorder.body_omission_reason = "different_lifecycle"
    assert recorder.body_omission_reason == _REASON
    assert len(recorder.requests) == 2
    assert all(attempt.body is attempt.response.body is None for attempt in recorder.requests)
    assert all(attempt.diagnostics["body_omission_reason"] == _REASON for attempt in recorder.requests)


@pytest.mark.asyncio
async def test_body_omission_without_an_active_recorder_does_not_affect_later_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)
    ready = asyncio.Event()

    async def late_omission() -> None:
        await ready.wait()
        ModelClientRequestRecorder.omit_current_http_bodies(_REASON)

    ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
    with ModelClientRequestRecorder(record_http_bodies=True) as closed:
        task = asyncio.create_task(late_omission())
    ready.set()
    await task
    assert closed.requests == ()
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as following:
            await _request(model, "ordinary")
    attempt, = following.requests
    assert attempt.record_http_bodies and attempt.body is not None and attempt.response.body is not None
    assert "body_omission_reason" not in attempt.diagnostics


@pytest.mark.asyncio
async def test_body_omission_survives_bound_stream_entry_and_close_in_different_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Stream([_EVENT + _DONE])
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=source,
    ))
    recorder = ModelClientRequestRecorder(record_http_bodies=True)
    wrapped = recorder.wrap_model(model)
    stream = wrapped.request_stream(
        [ModelRequest(parts=[UserPromptPart("referenced input")])], None, ModelRequestParameters(),
    )
    ready = asyncio.Event()

    async def enter() -> StreamedResponse:
        await ready.wait()
        return await stream.__aenter__()

    async with model:
        task = asyncio.create_task(enter())
        with recorder:
            ModelClientRequestRecorder.omit_current_http_bodies(_REASON)
            ready.set()
            response = await task
            try:
                async for _event in response:
                    pass
            finally:
                await asyncio.create_task(stream.__aexit__(None, None, None))

    attempt, = recorder.requests
    assert attempt.body is attempt.response.body is None
    assert attempt.capture_error is attempt.response.capture_error is None
    assert attempt.diagnostics["body_omission_reason"] == _REASON
    assert "close_finished_ns" in attempt.diagnostics
    assert source.closes == 1


@pytest.mark.parametrize("reason", ("", " \t", None, 1))
def test_body_omission_requires_a_nonempty_diagnostic_label(reason: object) -> None:
    with ModelClientRequestRecorder(record_http_bodies=True):
        with pytest.raises(ValueError, match="nonempty string"):
            ModelClientRequestRecorder.omit_current_http_bodies(reason)
