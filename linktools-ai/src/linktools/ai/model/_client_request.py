#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-scoped observation of requests prepared by supported SDK clients."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType


@dataclass(frozen=True, slots=True)
class ModelClientRequest:
    """One client dispatch, not proof of receipt or provider-side processing.

    The body is the client's original JSON bytes. No headers, URL, or credentials
    from the connection are retained. The body is unredacted, including values
    supplied through extra_body. A failed capture keeps its place among attempts
    without inventing a request body.
    """

    api: str
    body: bytes | None
    capture_error: str | None = None

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


def _record_client_request(
    api: str, body: bytes | None, *, capture_error: str | None = None,
) -> None:
    capture = _current_capture.get()
    if capture is not None and capture._token is not None:
        capture._requests.append(ModelClientRequest(api, body, capture_error))


__all__ = ["ModelClientRequest", "ModelClientRequestRecorder"]
