#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Passive HTTP observation shared by supported model protocols."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable

from httpx2 import AsyncClient, AsyncByteStream, ByteStream, DecodingError, EventSource, Request, RequestNotRead, Response, SSEError

from ._client_request import ModelClientResponse, _ClientRequestEntry, _record_client_request, _record_http_bodies

_CLIENT_REQUEST_ENTRY = "linktools.model.client_request_entry"


async def capture_request(request: Request) -> None:
    api = next((api for suffix, api in (
        ("/chat/completions", "openai-chat"),
        ("/responses", "openai-responses"),
        ("/messages", "anthropic-messages"),
    ) if request.url.path.endswith(suffix)), None)
    if request.method != "POST" or api is None:
        return
    if not _record_http_bodies():
        entry = _record_client_request(api, None)
    else:
        try:
            body = request.content
        except RequestNotRead:
            entry = _record_client_request(api, None, capture_error="body_unavailable")
        else:
            entry = _record_client_request(api, body)
    if entry is not None:
        request.extensions[_CLIENT_REQUEST_ENTRY] = entry


def response_snapshot(
    response: Response, body: bytes | None, *, content_encoding: str | None, eof: bool,
    record_http_bodies: bool = True, api: str = "openai-chat",
) -> ModelClientResponse:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    format = "sse" if content_type == "text/event-stream" else (
        "json" if content_type == "application/json" or content_type.endswith("+json") else "unknown"
    )
    if not record_http_bodies:
        return ModelClientResponse(
            body=None, status_code=response.status_code, format=format,
            content_encoding=content_encoding, complete=False,
        )
    complete = eof and body is not None
    capture_error = "response_capture_failed" if body is None else None
    if format == "sse" and body is not None:
        replica = Response(
            200, headers={"content-type": "text/event-stream", "content-encoding": content_encoding or ""},
            stream=ByteStream(body), request=Request("GET", "https://model-response.invalid"),
        )
        try:
            complete = any(_terminal_event(api, event.data, event.event) for event in EventSource(replica))
        except (DecodingError, SSEError, ImportError):
            complete = False
            capture_error = "response_decode_failed"
    return ModelClientResponse(
        body=body, status_code=response.status_code, format=format,
        content_encoding=content_encoding, complete=complete, capture_error=capture_error,
    )


class CapturedResponseStream(AsyncByteStream):
    def __init__(self, response: Response, entry: _ClientRequestEntry) -> None:
        self._response = response
        self._inner = response.stream
        self._entry = entry
        self._parts: list[bytes] | None = []
        self._eof = False

    def _capture(self, part: bytes | None = None, *, buffered: bool = False) -> None:
        if not self._entry.active:
            self._parts = None
            return
        try:
            if part is not None:
                if self._parts is not None:
                    self._parts.append(part)
                return
            body = self._response.content if buffered else (
                b"".join(self._parts) if self._parts is not None else None
            )
            self._entry.record_response(response_snapshot(
                self._response, body,
                content_encoding=None if buffered else self._response.headers.get("content-encoding") or None,
                eof=buffered or self._eof, api=self._entry.api,
            ))
        except Exception:
            # Observation failures must not change SDK reads or cleanup errors.
            self._parts = None
            if part is None:
                try:
                    self._entry.record_response(ModelClientResponse(
                        body=None, status_code=self._response.status_code, format="unknown",
                        content_encoding=None, complete=False, capture_error="response_capture_failed",
                    ))
                except Exception:
                    # An unavailable recorder cannot provide a replacement view.
                    pass
        finally:
            if part is None:
                self._parts = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self._inner.__aiter__()
        try:
            async for part in iterator:
                self._capture(part)
                yield part
            self._eof = True
        finally:
            if isinstance(iterator, AsyncGenerator):
                await iterator.aclose()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            # HTTPX retains elapsed timing on its own stream wrapper.
            self._response.stream = self._inner
            self._capture()


async def capture_response(response: Response) -> None:
    entry = response.request.extensions.pop(_CLIENT_REQUEST_ENTRY, None)
    if entry is None or not entry.active:
        return
    if not entry.record_http_bodies:
        entry.record_response(response_snapshot(
            response, None, content_encoding=response.headers.get("content-encoding") or None,
            eof=False, record_http_bodies=False, api=entry.api,
        ))
        return
    capture = CapturedResponseStream(response, entry)
    if response.is_stream_consumed:
        # A custom transport may return a response it already read and decoded.
        capture._capture(buffered=True)
    else:
        response.stream = capture


def _terminal_event(api: str, data: str, event: str | None) -> bool:
    if api == "openai-chat":
        return data == "[DONE]"
    try:
        payload = json.loads(data)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type", event)
    if api == "openai-responses":
        return kind in {"response.completed", "response.failed", "response.incomplete", "error"}
    if api == "anthropic-messages":
        return kind in {"message_stop", "error"}
    return False


class HttpClientOwner:
    """Close one injected client after the last overlapping model context."""

    def __init__(self, factory: Callable[[], AsyncClient]) -> None:
        self._factory = factory
        self.client = factory()
        self._lock = asyncio.Lock()
        self._entries = 0

    async def acquire(self, replace: Callable[[AsyncClient], None]) -> None:
        async with self._lock:
            if self.client.is_closed:
                client = self._factory()
                try:
                    replace(client)
                except BaseException:
                    await client.aclose()
                    raise
                self.client = client
            self._entries += 1

    async def release(self) -> None:
        async with self._lock:
            if self._entries:
                self._entries -= 1
                if not self._entries:
                    await self.client.aclose()
