#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Passive HTTP observation shared by supported model protocols."""

import asyncio
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable

from httpx2 import AsyncClient, AsyncByteStream, ByteStream, EventSource, Request, RequestNotRead, Response
from linktools.core import environ

from ._client_request import ModelClientResponse, _ClientRequestEntry, _record_client_request, _record_http_bodies
from ..core import JsonValue

_CLIENT_REQUEST_ENTRY = "linktools.model.client_request_entry"


async def capture_request(request: Request) -> None:
    api = next((api for suffix, api in (
        ("/chat/completions", "openai"),
        ("/responses", "openai-responses"),
        ("/messages", "anthropic"),
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
    record_http_bodies: bool = True, api: str = "openai",
) -> tuple[ModelClientResponse, str | None]:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    format = "sse" if content_type == "text/event-stream" else (
        "json" if content_type == "application/json" or content_type.endswith("+json") else "unknown"
    )
    if not record_http_bodies:
        return ModelClientResponse(
            body=None, status_code=response.status_code, format=format,
            content_encoding=content_encoding, complete=False,
        ), None
    complete = eof and body is not None
    terminal = None
    capture_error = "response_capture_failed" if body is None else None
    if format == "sse" and body is not None:
        try:
            replica = Response(
                200, headers={"content-type": "text/event-stream", "content-encoding": content_encoding or ""},
                stream=ByteStream(body), request=Request("GET", "https://model-response.invalid"),
            )
            terminal = next((kind for event in EventSource(replica)
                             if (kind := _terminal_event(api, event.data, event.event)) is not None), None)
            complete = terminal is not None
        except Exception:
            # Derived metadata must not discard bytes already observed from the SDK stream.
            complete = False
            capture_error = "response_decode_failed"
    return ModelClientResponse(
        body=body, status_code=response.status_code, format=format,
        content_encoding=content_encoding, complete=complete, capture_error=capture_error,
    ), terminal


class CapturedResponseStream(AsyncByteStream):
    def __init__(self, response: Response, entry: _ClientRequestEntry) -> None:
        self._response = response
        self._inner = response.stream
        self._entry = entry
        self._parts: list[bytes] | None = [] if entry.record_http_bodies else None
        self._eof = False
        self._received_bytes = 0
        self._received_chunks = 0
        self._first_chunk_ns: int | None = None
        self._last_chunk_ns: int | None = None

    def _observe(self, **values: JsonValue) -> None:
        self._entry.record_observation(
            received_bytes=self._received_bytes, received_chunks=self._received_chunks,
            **({"first_chunk_ns": self._first_chunk_ns, "last_chunk_ns": self._last_chunk_ns}
               if self._first_chunk_ns is not None else {}),
            **values,
        )

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
            snapshot, terminal = response_snapshot(
                self._response, body,
                content_encoding=None if buffered else self._response.headers.get("content-encoding") or None,
                eof=buffered or self._eof, api=self._entry.api,
                record_http_bodies=self._entry.record_http_bodies,
            )
            self._entry.record_response(snapshot)
            if snapshot.format == "sse" and body is not None and snapshot.capture_error is None:
                self._entry.record_observation(protocol_terminal=terminal)
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
            while True:
                try:
                    part = await iterator.__anext__()
                except StopAsyncIteration:
                    self._eof = True
                    self._observe(read_outcome="eof", read_finished_ns=self._entry.elapsed_ns)
                    return
                except asyncio.CancelledError as error:
                    self._observe(read_outcome="cancelled", read_finished_ns=self._entry.elapsed_ns,
                                  read_error_type=self._entry.error_type(error))
                    raise
                except Exception as error:
                    self._observe(read_outcome="error", read_finished_ns=self._entry.elapsed_ns,
                                  read_error_type=self._entry.error_type(error))
                    raise
                observed_ns = self._entry.elapsed_ns
                if self._first_chunk_ns is None:
                    self._first_chunk_ns = observed_ns
                self._last_chunk_ns = observed_ns
                self._received_bytes += len(part)
                self._received_chunks += 1
                self._capture(part)
                yield part
        finally:
            if isinstance(iterator, AsyncGenerator):
                try:
                    await iterator.aclose()
                except BaseException as error:
                    self._observe(iterator_close_error_type=self._entry.error_type(error),
                                  iterator_close_error_ns=self._entry.elapsed_ns)
                    raise

    async def aclose(self) -> None:
        self._observe(close_started_ns=self._entry.elapsed_ns)
        try:
            await self._inner.aclose()
        except BaseException as error:
            self._observe(close_error_type=self._entry.error_type(error))
            raise
        finally:
            self._observe(close_finished_ns=self._entry.elapsed_ns)
            # HTTPX retains elapsed timing on its own stream wrapper.
            self._response.stream = self._inner
            self._capture()


async def capture_response(response: Response) -> None:
    entry = response.request.extensions.pop(_CLIENT_REQUEST_ENTRY, None)
    if entry is None or not entry.active:
        return
    observations: dict[str, JsonValue] = {"headers_received_ns": entry.elapsed_ns}
    if response.http_version in {"HTTP/1.0", "HTTP/1.1", "HTTP/2", "HTTP/3"}:
        observations["http_version"] = response.http_version
    request_id = response.headers.get("request-id" if entry.api == "anthropic" else "x-request-id")
    if (request_id is not None and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id)
            and environ.logging.redact(request_id) == request_id):
        observations["request_id"] = request_id
    entry.record_observation(**observations)
    try:
        snapshot, _ = response_snapshot(
            response, None, content_encoding=response.headers.get("content-encoding") or None,
            eof=False, record_http_bodies=False, api=entry.api,
        )
        entry.record_response(snapshot)
    except Exception:
        pass
    capture = CapturedResponseStream(response, entry)
    if response.is_stream_consumed:
        # A custom transport may return a response it already read and decoded.
        if entry.record_http_bodies:
            capture._capture(buffered=True)
    else:
        response.stream = capture


def _terminal_event(api: str, data: str, event: str | None) -> str | None:
    if api == "openai":
        return "[DONE]" if data == "[DONE]" else None
    try:
        payload = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    kind = payload.get("type", event)
    if not isinstance(kind, str):
        return None
    if api == "openai-responses":
        return kind if kind in {"response.completed", "response.failed", "response.incomplete", "error"} else None
    if api == "anthropic":
        return kind if kind in {"message_stop", "error"} else None
    return None


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
