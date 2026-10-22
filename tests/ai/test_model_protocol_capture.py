#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-native terminal observations do not imply model success."""

import json

import httpx2
import pytest

from linktools.ai.model import ModelClientRequestRecorder
from linktools.ai.model._http import capture_request, capture_response

from .test_model_client_response_capture import _Stream


@pytest.mark.asyncio
@pytest.mark.parametrize("path,api,terminal", [
    ("chat/completions", "openai-chat", "[DONE]"),
    ("responses", "openai-responses", '{"type":"response.completed"}'),
    ("responses", "openai-responses", '{"type":"response.failed"}'),
    ("responses", "openai-responses", '{"type":"response.incomplete"}'),
    ("responses", "openai-responses", '{"type":"error","message":"failed"}'),
    ("messages", "anthropic-messages", '{"type":"message_stop"}'),
    ("messages", "anthropic-messages", '{"type":"error","error":{"type":"overloaded_error"}}'),
])
async def test_protocol_terminal_observation_keeps_original_bytes(
    path: str, api: str, terminal: str,
) -> None:
    body = b'data: ' + terminal.encode() + b'\n\n'
    source = _Stream([body[:8], body[8:]])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source,
        )),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"},
                              headers={"Authorization": "Bearer not-recorded"})
    captured = recorder.requests[0]
    assert captured.api == api
    assert json.loads(captured.body) == {"input": "hello"}
    assert captured.response.body == body and captured.response.complete
    assert "not-recorded" not in repr(captured)
    assert source.received == [body[:8], body[8:]] and source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["responses", "messages"])
async def test_protocol_disconnect_and_disabled_capture_do_not_invent_completion(path: str) -> None:
    body = b'data: {"type":"unknown.delta","value":"partial"}\n\n'
    for enabled in (False, True):
        source = _Stream([body, httpx2.ReadError("disconnected")])
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=source,
            )),
            event_hooks={"request": [capture_request], "response": [capture_response]},
        ) as client:
            with ModelClientRequestRecorder(record_http_bodies=enabled) as recorder:
                with pytest.raises(httpx2.ReadError):
                    await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"})
        captured = recorder.requests[0]
        assert captured.response.body == (body if enabled else None)
        assert not captured.response.complete
        assert (captured.body is not None) == enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("path,terminal", [("responses", "response.completed"), ("messages", "message_stop")])
@pytest.mark.parametrize("finished", [False, True])
async def test_malformed_event_type_preserves_observed_bytes(path: str, terminal: str, finished: bool) -> None:
    malformed = b'data: {"type":{}}\n\ndata: {"type":[]}\n\n'
    ending = b'data: ' + json.dumps({"type": terminal}).encode() + b'\n\n'
    chunks = [malformed, ending] if finished else [malformed, httpx2.ReadError("disconnected")]
    source = _Stream(chunks)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source,
        )),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            if finished:
                response = await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"})
                assert response.content == malformed + ending
            else:
                with pytest.raises(httpx2.ReadError):
                    await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"})
    observed = recorder.requests[0].response
    assert observed.body == malformed + (ending if finished else b'')
    assert observed.complete is finished
    assert observed.capture_error is None
    assert source.received == ([malformed, ending] if finished else [malformed])
    assert source.closes == 1
