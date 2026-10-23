#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempt diagnostics remain useful without retaining HTTP bodies."""

import asyncio
from datetime import datetime

import httpx2
import pytest

from linktools.core import environ
from linktools.ai.model import ModelClientRequest, ModelClientRequestRecorder
from linktools.ai.model._http import capture_request, capture_response

from .test_model_client_response_capture import _Stream


@pytest.mark.asyncio
async def test_cancelled_stream_keeps_timing_without_buffering_response_bytes() -> None:
    body = b'data: {"private":"unrecorded-response"}\n\n'
    source = _Stream([body], block=True)
    response = httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
    recorder = ModelClientRequestRecorder()

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: response),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        async def invoke() -> None:
            with recorder:
                await client.post("https://model.invalid/chat/completions", json={"private": "unrecorded-input"})

        task = asyncio.create_task(invoke())
        await asyncio.wait_for(source.blocked.wait(), 3)
        # No public API exposes retained in-flight bytes; this checks the privacy boundary.
        assert response.stream._parts is None
        headers_view = recorder.requests
        assert headers_view[0].response.status_code == 200
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    attempt, = recorder.requests
    assert attempt.body is attempt.response.body is None
    diagnostics = attempt.diagnostics
    assert datetime.fromisoformat(diagnostics["started_at"]).utcoffset().total_seconds() == 0
    assert diagnostics["received_bytes"] == len(body) and diagnostics["received_chunks"] == 1
    assert diagnostics["headers_received_ns"] <= diagnostics["first_chunk_ns"] == diagnostics["last_chunk_ns"]
    assert diagnostics["last_chunk_ns"] <= diagnostics["read_finished_ns"] <= diagnostics["close_started_ns"]
    assert diagnostics["close_started_ns"] <= diagnostics["close_finished_ns"]
    assert diagnostics["read_outcome"] == "cancelled"
    assert diagnostics["read_error_type"] == "asyncio.exceptions.CancelledError"
    assert "sdk_cancel_requested_ns" not in diagnostics and "protocol_terminal" not in diagnostics
    assert "unrecorded" not in str(dict(diagnostics))
    assert "received_bytes" not in headers_view[0].diagnostics
    assert source.closes == 1


@pytest.mark.asyncio
async def test_read_and_close_failures_remain_separate_observations() -> None:
    read_error = httpx2.RemoteProtocolError("private read details")
    close_error = RuntimeError("private close details")
    source = _Stream([b"partial", read_error], close_error=close_error)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, stream=source)),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder() as recorder:
            with pytest.raises(RuntimeError) as caught:
                await client.post("https://model.invalid/responses", json={"input": "hello"})
    assert caught.value is close_error and close_error.__context__ is read_error
    diagnostics = recorder.requests[0].diagnostics
    assert diagnostics["read_outcome"] == "error"
    assert diagnostics["read_error_type"] == "httpx2.RemoteProtocolError"
    assert diagnostics["close_error_type"] == "builtins.RuntimeError"
    assert diagnostics["received_bytes"] == len(b"partial")
    assert "private" not in str(dict(diagnostics))
    assert source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_error", [ValueError("consumer failed"), asyncio.CancelledError()])
async def test_consumer_injected_error_is_not_a_transport_read_failure(consumer_error: BaseException) -> None:
    request = httpx2.Request("POST", "https://model.invalid/responses", json={"input": "hello"})
    source = _Stream([b"first", b"second"])
    response = httpx2.Response(200, stream=source, request=request)
    with ModelClientRequestRecorder() as recorder:
        await capture_request(request)
        await capture_response(response)
        iterator = response.stream.__aiter__()
        assert await iterator.__anext__() == b"first"
        with pytest.raises(type(consumer_error)) as caught:
            await iterator.athrow(consumer_error)
        assert caught.value is consumer_error
        await response.aclose()
    diagnostics = recorder.requests[0].diagnostics
    assert diagnostics["received_bytes"] == len(b"first")
    assert "read_outcome" not in diagnostics and "read_error_type" not in diagnostics
    assert "close_finished_ns" in diagnostics
    assert source.received == [b"first"] and source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path,header", [
    ("chat/completions", "x-request-id"), ("responses", "x-request-id"), ("messages", "request-id"),
])
@pytest.mark.parametrize("request_id", ["req_123-abc", "bad value", "x" * 129, "known-private-secret"])
async def test_request_id_is_protocol_scoped_and_bounded_without_buffered_chunk_facts(
    monkeypatch: pytest.MonkeyPatch, path: str, header: str, request_id: str,
) -> None:
    redact = environ.logging.redact
    monkeypatch.setattr(environ.logging, "redact", lambda text: redact(text).replace("known-private-secret", "***"))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, json={"private": "unrecorded"}, headers={header: request_id, "x-secret": "not-retained"},
            extensions={"http_version": b"HTTP/2"},
        )),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder() as recorder:
            await client.post(f"https://model.invalid/{path}", json={"input": "hello"})
    diagnostics = recorder.requests[0].diagnostics
    assert diagnostics.get("request_id") == (request_id if request_id == "req_123-abc" else None)
    assert diagnostics["http_version"] == "HTTP/2"
    assert "first_chunk_ns" not in diagnostics and "received_bytes" not in diagnostics
    assert "protocol_terminal" not in diagnostics and "not-retained" not in str(dict(diagnostics))


@pytest.mark.asyncio
async def test_sdk_observations_remain_bound_to_their_attempt_and_closed_scope() -> None:
    recorder = ModelClientRequestRecorder()
    with recorder:
        recorder.observe_stream("entered", attempt_index=-1)
        assert recorder.requests == ()
        for index in range(2):
            await capture_request(httpx2.Request("POST", "https://model.invalid/responses", json={"n": index}))
        recorder.observe_stream("entered", attempt_index=0)
        recorder.observe_stream("exhausted", attempt_index=0)
        recorder.observe_stream("entry_error", attempt_index=1, error=RuntimeError("private provider details"))
        first, second = recorder.requests
        assert first.diagnostics["sdk_entered_ns"] <= first.diagnostics["sdk_exhausted_ns"]
        assert "sdk_entry_error_type" not in first.diagnostics
        assert second.diagnostics["sdk_entry_error_type"] == "builtins.RuntimeError"
        assert "sdk_entered_ns" not in second.diagnostics
    before = recorder.requests
    recorder.observe_stream("cancel_requested", attempt_index=0)
    assert recorder.requests == before


def test_attempt_diagnostics_are_detached_immutable_json() -> None:
    source = {"received_bytes": 7, "nested": {"value": "before"}}
    attempt = ModelClientRequest("openai", None, diagnostics=source)
    source["nested"]["value"] = "after"
    assert attempt.diagnostics["nested"] == {"value": "before"}
    with pytest.raises(TypeError):
        attempt.diagnostics["received_bytes"] = 8
