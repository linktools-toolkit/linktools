#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-scoped observation of requests prepared by supported SDK clients."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelClientResponse:
    """Bytes observed by the client, not a reconstructed assistant message.

    ``content_encoding`` describes the stored bytes. Streaming transports retain
    their encoded bytes; already-buffered HTTP client responses provide decoded
    bytes. ``format`` is only a Content-Type hint, not validation of the body.
    ``complete`` means an observed Chat SSE terminator or a non-SSE transport EOF,
    not SDK success. Bytes after a terminator in the same received chunk may be
    present even though the SDK did not process them. No response headers other
    than the format/encoding hints are retained.
    A missing body means capture failed; an empty body is retained as ``b""``.
    """

    body: bytes | None
    status_code: int
    format: Literal["json", "sse", "unknown"]
    content_encoding: str | None
    complete: bool
    capture_error: str | None = None

    def __post_init__(self) -> None:
        if self.body is None and self.capture_error is None:
            raise ValueError("missing client response body requires a capture error")
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("client response body must be bytes")
        if self.format not in ("json", "sse", "unknown"):
            raise ValueError("client response format is invalid")
        if self.complete and self.capture_error is not None:
            raise ValueError("failed response capture cannot be complete")


@dataclass(frozen=True, slots=True)
class ModelClientRequest:
    """One client dispatch, not proof of receipt or provider-side processing.

    The body is the client's original JSON bytes. No headers, URL, or credentials
    from the connection are retained. The body is unredacted, including values
    supplied through extra_body. A failed capture keeps its place among attempts
    without inventing a request body. ``response=None`` means no response view
    was recorded, not proof that the provider produced no response.
    """

    api: str
    body: bytes | None
    capture_error: str | None = None
    response: ModelClientResponse | None = None

    def __post_init__(self) -> None:
        if not self.api or (self.body is None) != (self.capture_error is not None):
            raise ValueError("client request capture is invalid")
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("client request body must be bytes")


_current_capture: ContextVar[ModelClientRequestRecorder | None] = ContextVar(
    "linktools_model_client_request_capture", default=None,
)


class ModelClientRequestRecorder:
    """Collect supported SDK dispatches in one scope without changing requests.

    Each instance is used once. SDK child tasks inherit their request's scope;
    separate logical requests enter their own recorder. Nested requests belong
    to their innermost scope. Requests are retained only in memory. Persistence
    and authorization remain the caller's responsibility.
    """

    def __init__(self) -> None:
        self._requests: list[ModelClientRequest] = []
        self._token: Token[ModelClientRequestRecorder | None] | None = None
        self._entered = False

    @property
    def requests(self) -> tuple[ModelClientRequest, ...]:
        """Immutable views; response closure replaces values in later views."""
        return tuple(self._requests)

    def __enter__(self) -> ModelClientRequestRecorder:
        if self._entered:
            raise RuntimeError("client request capture cannot be entered again")
        self._entered = True
        self._token = _current_capture.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        token = self._token
        if token is not None:
            _current_capture.reset(token)
            self._token = None


@dataclass(frozen=True, slots=True)
class _ClientRequestEntry:
    recorder: ModelClientRequestRecorder
    index: int

    @property
    def active(self) -> bool:
        return self.recorder._token is not None

    def record_response(self, response: ModelClientResponse) -> None:
        if self.active:
            requests = self.recorder._requests
            requests[self.index] = replace(requests[self.index], response=response)


def _record_client_request(
    api: str, body: bytes | None, *, capture_error: str | None = None,
) -> _ClientRequestEntry | None:
    capture = _current_capture.get()
    if capture is not None and capture._token is not None:
        capture._requests.append(ModelClientRequest(api, body, capture_error))
        return _ClientRequestEntry(capture, len(capture._requests) - 1)
    return None


__all__ = ["ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse"]
