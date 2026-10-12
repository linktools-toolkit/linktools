#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Client response capture preserves bytes without reading ahead of the SDK."""

import asyncio
import gc
import gzip
import importlib.util
import json
import zlib
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import httpx2
import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters

from linktools.ai.model import ModelClientRequestRecorder
from linktools.ai.model import _client_request, _openai
from .test_model_client_request_capture import _model, _request


_JSON = (
    b'{ "id":"chatcmpl-test", "object":"chat.completion", "created":null,'
    b' "model":"gpt-4o", "unknown":{"keep":true}, "choices":[{"index":0,'
    b'"message":{"role":"assistant","content":"answer","unknown":null},'
    b'"finish_reason":null}] }'
)
_EVENT = b'data: ' + json.dumps({
    "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
    "model": "gpt-4o", "unknown": None,
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer"},
                 "finish_reason": "stop"}],
}).encode() + b'\r\n\r\n'
_DONE = b'data: [DONE]\n\n'


class _Stream(httpx2.AsyncByteStream):
    def __init__(
        self, chunks: list[bytes | BaseException], *, block: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self.chunks = chunks
        self.block = block
        self.close_error = close_error
        self.blocked = asyncio.Event()
        self.pulls = 0
        self.closes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.pulls += 1
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk
        if self.block:
            self.blocked.set()
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


async def _invoke(model: Model, streaming: bool) -> None:
    if not streaming:
        await _request(model, "prompt")
        return
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("prompt")])], None, ModelRequestParameters(),
    ) as response:
        async for _event in response:
            pass


def _encoded(body: bytes, encoding: str) -> bytes:
    if encoding == "gzip":
        return gzip.compress(body)
    if encoding == "deflate":
        return zlib.compress(body)
    if encoding == "raw-deflate":
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        return compressor.compress(body) + compressor.flush()
    if encoding == "br":
        if importlib.util.find_spec("brotli"):
            import brotli
            return brotli.compress(body)
        if importlib.util.find_spec("brotlicffi"):
            import brotlicffi
            return brotlicffi.compress(body)
        pytest.skip("Brotli decoding extra is not installed")
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("encoding", ("", "gzip", "deflate", "raw-deflate", "br"))
async def test_model_client_response_keeps_received_bytes_and_encoding(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, encoding: str,
) -> None:
    body = _encoded(_EVENT + _DONE if streaming else _JSON, encoding)
    source = _Stream([body[:13], body[13:]])
    response = httpx2.Response(200, stream=source, headers={
        "content-type": "text/event-stream" if streaming else "application/json; charset=utf-8",
        "content-encoding": "deflate" if encoding == "raw-deflate" else encoding,
        "x-secret": "do not retain",
    })
    model, _clients = _model(monkeypatch, lambda request: response)
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await _invoke(model, streaming)
    captured = recorder.requests[0].response
    assert captured is not None
    assert captured.body == body
    assert captured.content_encoding == ("deflate" if encoding == "raw-deflate" else encoding or None)
    assert captured.format == ("sse" if streaming else "json")
    assert captured.status_code == 200 and captured.complete
    assert captured.capture_error is None
    assert source.pulls == 2 and source.closes == 1
    assert response.elapsed.total_seconds() >= 0
    assert "do not retain" not in repr(captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending,complete", [(b"", False), (b"data: [DONE]", False), (_DONE, True)])
async def test_model_client_response_completion_requires_observed_sse_terminator(
    monkeypatch: pytest.MonkeyPatch, ending: bytes, complete: bool,
) -> None:
    source = _Stream([_EVENT, ending])
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=source,
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await _invoke(model, True)
    captured = recorder.requests[0].response
    assert captured is not None and captured.body == _EVENT + ending
    assert captured.complete is complete


@pytest.mark.asyncio
async def test_model_client_response_does_not_pull_after_done_or_claim_tail_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _EVENT + _DONE + b": received in same chunk but not processed by SDK\n\n"
    source = _Stream([body, b"not received"])
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=source,
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await _invoke(model, True)
    captured = recorder.requests[0].response
    assert captured is not None and captured.body == body and captured.complete
    assert source.pulls == 1 and source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type,body,expected", [
    ("text/html", b"<html>gateway failure</html>", "unknown"),
    ("", b"", "unknown"),
    ("application/json", b"{bad json", "json"),
])
async def test_model_client_response_format_is_only_a_content_type_hint(
    monkeypatch: pytest.MonkeyPatch, content_type: str, body: bytes, expected: str,
) -> None:
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        400, headers={"content-type": content_type}, stream=_Stream([body]),
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            with pytest.raises(ModelHTTPError):
                await _invoke(model, False)
    captured = recorder.requests[0].response
    assert captured is not None and captured.body == body
    assert captured.format == expected and captured.status_code == 400
    assert captured.complete and captured.capture_error is None


@pytest.mark.asyncio
async def test_model_client_response_replaces_frozen_snapshots_and_binds_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    failure = b'{"error":{"message":"retry"}}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        snapshots.append(recorder.requests)
        if len(snapshots) == 1:
            return httpx2.Response(429, headers={"content-type": "application/json"}, stream=_Stream([failure]))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Stream([_EVENT + _DONE]))

    model, _clients = _model(monkeypatch, handler, retries=1)
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await _invoke(model, True)
    first, second = recorder.requests
    assert first.response.status_code == 429 and first.response.body == failure
    assert second.response.status_code == 200 and second.response.body == _EVENT + _DONE
    assert snapshots[0][0].response is None and snapshots[1][1].response is None
    assert snapshots[1][0].response == first.response
    with pytest.raises(FrozenInstanceError):
        first.response = None
    with pytest.raises(FrozenInstanceError):
        second.response.complete = False


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ("", "gzip"))
async def test_model_client_response_cancellation_retains_partial_body(
    monkeypatch: pytest.MonkeyPatch, encoding: str,
) -> None:
    body = _encoded(_EVENT, encoding)
    if encoding:
        body = body[:-4]
    source = _Stream([body], block=True)
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream", "content-encoding": encoding}, stream=source,
    ))
    recorder = ModelClientRequestRecorder()

    async def invoke() -> None:
        with recorder:
            await _invoke(model, True)

    async with model:
        task = asyncio.create_task(invoke())
        await asyncio.wait_for(source.blocked.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    captured = recorder.requests[0].response
    assert captured is not None and captured.body == body and not captured.complete
    assert source.closes == 1


@pytest.mark.asyncio
async def test_model_client_response_complete_body_does_not_imply_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"broken":'
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "application/json"}, stream=_Stream([body]),
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            with pytest.raises(ModelAPIError):
                await _invoke(model, False)
    captured = recorder.requests[0].response
    assert captured.body == body and captured.complete and captured.format == "json"


@pytest.mark.asyncio
async def test_model_client_response_preserves_read_and_close_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for error_on_close in (False, True):
        failure = RuntimeError("close failed") if error_on_close else httpx2.ReadError("read failed")
        source = _Stream([_EVENT] if error_on_close else [_EVENT, failure], close_error=failure if error_on_close else None)
        model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source,
        ))
        async with model:
            with ModelClientRequestRecorder() as recorder:
                with pytest.raises(RuntimeError if error_on_close else ModelAPIError) as caught:
                    await _invoke(model, True)
        if error_on_close:
            assert caught.value is failure
        captured = recorder.requests[0].response
        assert captured is not None and captured.body == _EVENT and not captured.complete
        assert source.closes == 1


@pytest.mark.asyncio
async def test_model_client_response_buffered_transport_content_is_already_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, content=gzip.compress(_JSON),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await _invoke(model, False)
    captured = recorder.requests[0].response
    assert captured is not None and captured.body == _JSON
    assert captured.content_encoding is None and captured.complete


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", (False, True))
async def test_model_client_response_binding_survives_overlapping_requests_and_scopes(
    monkeypatch: pytest.MonkeyPatch, nested: bool,
) -> None:
    first_dispatched = asyncio.Event()
    finish_first = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        if prompt == "first":
            first_dispatched.set()
            await finish_first.wait()
        return httpx2.Response(
            200, headers={"content-type": "application/json"},
            stream=_Stream([_JSON.replace(b"answer", prompt.encode())]),
        )

    model, _clients = _model(monkeypatch, handler)
    async with model:
        with ModelClientRequestRecorder() as outer:
            first = asyncio.create_task(_request(model, "first"))
            await asyncio.wait_for(first_dispatched.wait(), 3)
            if nested:
                with ModelClientRequestRecorder() as inner:
                    await _request(model, "second")
            else:
                await _request(model, "second")
            finish_first.set()
            await first
    requests = outer.requests + inner.requests if nested else outer.requests
    assert len(requests) == 2
    assert requests[0].response.body == _JSON.replace(b"answer", b"first")
    assert requests[1].response.body == _JSON.replace(b"answer", b"second")


@pytest.mark.asyncio
async def test_model_client_response_does_not_change_a_closed_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        dispatched.set()
        await release.wait()
        return httpx2.Response(200, headers={"content-type": "application/json"}, stream=_Stream([_JSON]))

    model, _clients = _model(monkeypatch, handler)
    async with model:
        with ModelClientRequestRecorder() as recorder:
            task = asyncio.create_task(_request(model, "prompt"))
            await asyncio.wait_for(dispatched.wait(), 3)
        before = recorder.requests
        release.set()
        await task
    assert recorder.requests == before and before[0].response is None


@pytest.mark.asyncio
async def test_model_client_response_decoding_failure_retains_bytes_and_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"not a gzip stream"
    source = _Stream([body])
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream", "content-encoding": "gzip"}, stream=source,
    ))
    async with model:
        with ModelClientRequestRecorder() as recorder:
            with pytest.raises(ModelAPIError):
                await _invoke(model, True)
    captured = recorder.requests[0].response
    assert captured.body == body and captured.content_encoding == "gzip"
    assert not captured.complete and captured.capture_error == "response_decode_failed"
    assert source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at,outcome", [
    ("append", "read_error"), ("snapshot", "close_error"),
    ("record", "success"), ("snapshot", "cancel"),
])
async def test_model_client_response_capture_failures_preserve_sdk_outcome(
    monkeypatch: pytest.MonkeyPatch, failure_at: str, outcome: str,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unsafe capture exception details")

    if failure_at == "append":
        class FailedParts(list):
            append = fail

        initialize = _openai._CapturedChatResponseStream.__init__

        def initialize_failed_parts(self, response, entry):
            initialize(self, response, entry)
            self._parts = FailedParts()

        monkeypatch.setattr(_openai._CapturedChatResponseStream, "__init__", initialize_failed_parts)
    elif failure_at == "snapshot":
        monkeypatch.setattr(_openai, "_chat_response_snapshot", fail)
    else:
        # The fallback encounters the same unavailable recorder.
        monkeypatch.setattr(_client_request._ClientRequestEntry, "record_response", fail)

    close_error = RuntimeError("original close failure")
    chunks = [_EVENT, httpx2.ReadError("original read failure")] if outcome == "read_error" else [_EVENT]
    if outcome == "success":
        chunks.append(_DONE)
    source = _Stream(
        chunks, block=outcome == "cancel", close_error=close_error if outcome == "close_error" else None,
    )
    model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=source,
    ))
    recorder = ModelClientRequestRecorder()

    async def invoke() -> None:
        with recorder:
            await _invoke(model, True)

    async with model:
        task = asyncio.create_task(invoke())
        if outcome == "cancel":
            await asyncio.wait_for(source.blocked.wait(), 3)
            task.cancel()
        if outcome == "success":
            await task
        else:
            expected_error = {
                "read_error": ModelAPIError, "close_error": RuntimeError, "cancel": asyncio.CancelledError,
            }[outcome]
            with pytest.raises(expected_error) as caught:
                await task
            if outcome == "close_error":
                assert caught.value is close_error
    captured = recorder.requests[0].response
    if failure_at == "record":
        assert captured is None
    else:
        assert captured.body is None and not captured.complete
        assert captured.capture_error == "response_capture_failed"
        assert "unsafe capture exception details" not in repr(captured)
    assert source.closes == 1


@pytest.mark.asyncio
async def test_model_client_response_closed_scope_stops_accumulating_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_consumed = asyncio.Event()
    resume = asyncio.Event()
    finished = asyncio.Event()

    class PausedStream(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _EVENT
            first_consumed.set()
            await resume.wait()
            for _ in range(8):
                yield _EVENT
            finished.set()
            await asyncio.Event().wait()

    response = httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=PausedStream())
    model, _clients = _model(monkeypatch, lambda request: response)
    async with model:
        with ModelClientRequestRecorder() as recorder:
            task = asyncio.create_task(_invoke(model, True))
            await asyncio.wait_for(first_consumed.wait(), 3)
            capture_stream = response.stream
            assert capture_stream._parts == [_EVENT]
        resume.set()
        await asyncio.wait_for(finished.wait(), 3)
        assert capture_stream._parts is None
        assert recorder.requests[0].response is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_model_client_response_iterator_finalization_matches_plain_client(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder, cancel: bool,
) -> None:
    original = RuntimeError("underlying iterator finalization failed")

    class FinallyFailStream(httpx2.AsyncByteStream):
        def __init__(self) -> None:
            self.blocked = asyncio.Event()
            self.finalized = asyncio.Event()
            self.pulls = 0
            self.closes = 0
            self.finalizations = 0

        async def __aiter__(self) -> AsyncIterator[bytes]:
            try:
                self.pulls += 1
                yield _EVENT if cancel else _EVENT + _DONE
                if cancel:
                    self.blocked.set()
                    await asyncio.Event().wait()
                self.pulls += 1
                yield b"unobserved"
            finally:
                self.finalizations += 1
                self.finalized.set()
                raise original

        async def aclose(self) -> None:
            self.closes += 1

    async def run(capture: bool):
        source = FinallyFailStream()
        unhandled = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        try:
            model, _clients = _model(monkeypatch, lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=source,
            ))

            async def invoke() -> None:
                with ModelClientRequestRecorder() if capture else nullcontext():
                    await _invoke(model, True)

            error = None
            async with model:
                task = asyncio.create_task(invoke())
                if cancel:
                    await asyncio.wait_for(source.blocked.wait(), 3)
                    task.cancel()
                try:
                    await task
                except BaseException as caught:
                    error = caught
            gc.collect()
            await asyncio.wait_for(source.finalized.wait(), 3)
            # Async generator finalizers may report through the event loop.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return error, source, unhandled
        finally:
            loop.set_exception_handler(previous_handler)

    plain_error, plain_source, plain_unhandled = await run(False)
    captured_error, captured_source, captured_unhandled = await run(True)
    assert type(captured_error) is type(plain_error)

    def caused_by_original(error: BaseException | None) -> bool:
        while error is not None:
            if error is original:
                return True
            error = error.__cause__
        return False

    assert caused_by_original(captured_error) == caused_by_original(plain_error)
    assert captured_source.pulls == plain_source.pulls == 1
    assert captured_source.closes == plain_source.closes == 1
    assert captured_source.finalizations == plain_source.finalizations == 1
    assert [item.get("exception") for item in captured_unhandled] == [item.get("exception") for item in plain_unhandled]
    assert not [warning for warning in recwarn if issubclass(warning.category, pytest.PytestUnraisableExceptionWarning)]
