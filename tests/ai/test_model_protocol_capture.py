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
    ("chat/completions", "openai", "[DONE]"),
    ("responses", "openai-responses", '{"type":"response.completed"}'),
    ("responses", "openai-responses", '{"type":"response.failed"}'),
    ("responses", "openai-responses", '{"type":"response.incomplete"}'),
    ("responses", "openai-responses", '{"type":"error","message":"failed"}'),
    ("messages", "anthropic", '{"type":"message_stop"}'),
    ("messages", "anthropic", '{"type":"error","error":{"type":"overloaded_error"}}'),
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
    assert captured.diagnostics["protocol_terminal"] == ("[DONE]" if terminal == "[DONE]" else json.loads(terminal)["type"])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("path,terminal", [("responses", "response.completed"), ("messages", "message_stop")])
@pytest.mark.parametrize("ending", ["eof", "terminal", "read_error", "close_error"])
async def test_deep_sse_json_retains_bytes_without_changing_transport_outcome(path: str, terminal: str, ending: str) -> None:
    # Native JSON nesting limits vary independently of the Python recursion limit.
    depth = 10000
    body = b'data: {"type":"error","details":' + b'[' * depth + b'0' + b']' * depth + b'}\n\n'
    try:
        json.loads(body.removeprefix(b"data: "))
    except RecursionError:
        decode_failed = True
    else:
        decode_failed = False
    if ending == "terminal":
        body += b'data: ' + json.dumps({"type": terminal}).encode() + b'\n\n'
    failure = httpx2.ReadError("read failed") if ending == "read_error" else RuntimeError("close failed")
    source = _Stream([body, failure] if ending == "read_error" else [body],
                     close_error=failure if ending == "close_error" else None)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source,
        )),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            if ending in {"read_error", "close_error"}:
                with pytest.raises(type(failure)) as caught:
                    await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"})
                assert caught.value is failure
            else:
                response = await client.post(f"https://model.invalid/v1/{path}", json={"input": "hello"})
                assert response.content == body
    observed = recorder.requests[0].response
    assert observed.body == body and observed.format == "sse" and observed.status_code == 200
    assert observed.content_encoding is None
    assert observed.complete is not decode_failed
    assert observed.capture_error == ("response_decode_failed" if decode_failed else None)
    assert source.received == [body] and source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["Response", "EventSource", "_terminal_event"])
async def test_diagnostic_replica_failure_retains_known_response_metadata(monkeypatch: pytest.MonkeyPatch, failure_at: str) -> None:
    from linktools.ai.model import _http

    def fail(*args: object, **kwargs: object) -> None:
        if failure_at == "_terminal_event":
            raise RecursionError("private decoder details")
        raise RuntimeError("private decoder details")

    monkeypatch.setattr(_http, failure_at, fail)
    body = b'data: [DONE]\n\n'
    source = _Stream([body])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(
            201, headers={"content-type": "text/event-stream", "content-encoding": "identity"}, stream=source,
        )),
        event_hooks={"request": [capture_request], "response": [capture_response]},
    ) as client:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            response = await client.post("https://model.invalid/v1/chat/completions", json={"input": "hello"})
            assert response.content == body
    observed = recorder.requests[0].response
    assert observed.body == body and observed.format == "sse" and observed.status_code == 201
    assert observed.content_encoding == "identity"
    assert not observed.complete and observed.capture_error == "response_decode_failed"
    assert "private decoder details" not in repr(observed)
    assert source.closes == 1
