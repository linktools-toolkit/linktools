#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-scoped observation of requests prepared by supported SDK clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic_ns
from types import TracebackType
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage
from linktools.core import environ

from ..core import ImmutableJsonMapping, JsonValue
from ._stream_attempt import ModelStreamAttempt


@dataclass(frozen=True, slots=True)
class ModelClientResponse:
    """Bytes observed by the client, not a reconstructed assistant message.

    ``content_encoding`` describes the stored bytes. Streaming transports retain
    their encoded bytes; already-buffered HTTP client responses provide decoded
    bytes. ``format`` is only a Content-Type hint, not validation of the body.
    ``complete`` means an observed protocol terminal SSE event or a non-SSE transport EOF,
    not SDK success. Bytes after a terminator in the same received chunk may be
    present even though the SDK did not process them. No response headers other
    than the format/encoding hints are retained in this response view.
    A missing body may be intentionally unrecorded or omitted by a metadata-only
    archive. capture_error is set only for an actual capture failure. An empty
    captured body is retained as ``b""``.
    """

    body: bytes | None
    status_code: int
    format: Literal["json", "sse", "unknown"]
    content_encoding: str | None
    complete: bool
    capture_error: str | None = None

    def __post_init__(self) -> None:
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("client response body must be bytes")
        if self.format not in ("json", "sse", "unknown"):
            raise ValueError("client response format is invalid")
        if self.complete and self.capture_error is not None:
            raise ValueError("failed response capture cannot be complete")


@dataclass(frozen=True, slots=True)
class ModelClientRequest:
    """One client dispatch, not proof of receipt or provider-side processing.

    The body is the client's original JSON bytes. No URL, credentials or full
    headers from the connection are retained. The body is unredacted, including values
    supplied through extra_body. A failed capture keeps its place among attempts
    without inventing a request body. ``response=None`` means no response view
    was recorded, not proof that the provider produced no response.
    record_http_bodies preserves the dispatch's capture policy. Metadata-only
    archives can omit recorded bytes without changing that policy or inventing errors.
    An explicit recorder-level omission sets this policy to false and retains
    its diagnostic label as ``diagnostics.body_omission_reason``.

    ``diagnostics`` contains passive observations of this HTTP attempt. Missing
    diagnostics or keys mean unobserved, including older records. ``started_at``
    is a UTC ISO timestamp; all ``*_ns`` values are monotonic elapsed nanoseconds
    from that attempt's start. ``headers_received_ns``, first/last chunk timing,
    ``received_bytes`` and ``received_chunks`` describe raw transport observation,
    not SDK events or decoded text. Buffered responses do not invent chunk facts.
    ``read_outcome`` is eof, error or cancelled; ``read_finished_ns`` and
    ``read_error_type`` describe that boundary. ``close_started_ns`` and
    ``close_finished_ns`` bracket the close attempt, including failures retained
    as ``close_error_type``. Iterator-finalization failures use
    ``iterator_close_error_ns`` and ``iterator_close_error_type`` instead.
    ``sdk_*`` records stream entry, exhaustion, cancellation requests and closure,
    not provider success. ``protocol_terminal`` names an observed SSE terminal,
    is null when recorded SSE bytes parse without a terminal, and is absent
    when not observed or decoding failed. HTTP version and a bounded
    protocol request ID (``http_version`` and ``request_id``) are the only
    additional response-header observations. Exception values contain only
    redacted qualified types, never messages or request/response data.
    """

    api: str
    body: bytes | None
    capture_error: str | None = None
    response: ModelClientResponse | None = None
    record_http_bodies: bool = True
    diagnostics: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.api or not isinstance(self.record_http_bodies, bool):
            raise ValueError("client request capture is invalid")
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("client request body must be bytes")
        if not self.record_http_bodies and (
            self.body is not None or self.response is not None and self.response.body is not None
        ):
            raise ValueError("disabled HTTP body capture cannot contain bodies")
        if self.diagnostics is not None:
            object.__setattr__(self, "diagnostics", ImmutableJsonMapping(self.diagnostics))


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

    def __init__(
        self, *, record_http_bodies: bool = False,
        stream_attempt_observer: Callable[[ModelStreamAttempt], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(record_http_bodies, bool):
            raise ValueError("record_http_bodies must be a boolean")
        self.record_http_bodies = record_http_bodies
        self._body_omission_reason: str | None = None
        self._entries: list[_ClientRequestEntry] = []
        self._token: Token[ModelClientRequestRecorder | None] | None = None
        self._entered = False
        self._stream_attempt_observer = stream_attempt_observer
        self._stream_attempts: list[ModelStreamAttempt] = []
        self._stream_attempt_seq = 0
        self._unreturned_stream_usage: RequestUsage | None = None

    @classmethod
    def current(cls) -> ModelClientRequestRecorder | None:
        """The open logical request scope, if one is bound in this task."""
        capture = _current_capture.get()
        return capture if capture is not None and capture._token is not None else None

    @property
    def unreturned_stream_usage(self) -> RequestUsage | None:
        """Known usage billed before any response reached the SDK response ledger."""
        return self._unreturned_stream_usage

    def record_unreturned_stream_usage(self, usage: RequestUsage) -> None:
        if self._unreturned_stream_usage is None:
            self._unreturned_stream_usage = RequestUsage()
        self._unreturned_stream_usage.incr(usage)

    @property
    def supports_stream_replacement(self) -> bool:
        """Whether the host has an awaited attempt lifecycle consumer."""
        return self._stream_attempt_observer is not None

    @property
    def stream_attempts(self) -> tuple[ModelStreamAttempt, ...] | None:
        """Ended stream attempts; None means no stream scope was observed."""
        return tuple(self._stream_attempts) if self._stream_attempt_seq else None

    def next_stream_attempt_seq(self) -> int:
        self._stream_attempt_seq += 1
        return self._stream_attempt_seq

    async def observe_stream_attempt(self, attempt: ModelStreamAttempt) -> None:
        """Publish control boundaries in order; observer failures are not swallowed."""
        if attempt.status != "RUNNING":
            self._stream_attempts.append(attempt)
        if self._stream_attempt_observer is not None:
            await self._stream_attempt_observer(attempt)

    @property
    def requests(self) -> tuple[ModelClientRequest, ...]:
        """Immutable views; response closure replaces values in later views."""
        return tuple(entry.request for entry in self._entries)

    @property
    def body_omission_reason(self) -> str | None:
        """The explicit omission label for this logical request, if any."""
        return self._body_omission_reason

    @classmethod
    def omit_current_http_bodies(cls, reason: str) -> None:
        """Omit raw HTTP bodies for the active logical request and its retries.

        Call before the first HTTP dispatch, for example before hydrating
        referenced content. The reason is a nonempty diagnostic label, not
        request content. Omission lasts for this recorder without changing its
        configured ``record_http_bodies`` value or affecting nested recorders.
        No active recorder is a no-op. Repeating the same reason is idempotent.
        Changing the reason or first requesting omission after a dispatch raises
        ``RuntimeError`` because previously observed bodies cannot be unobserved.
        """
        capture = _current_capture.get()
        if capture is None or capture._token is None:
            return
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("HTTP body omission reason must be a nonempty string")
        if capture._body_omission_reason == reason:
            return
        if capture._body_omission_reason is not None:
            raise RuntimeError("HTTP body omission reason cannot change within a logical request")
        if capture._entries:
            raise RuntimeError("HTTP body omission must be requested before the first dispatch")
        capture._body_omission_reason = reason

    @property
    def _effective_record_http_bodies(self) -> bool:
        return self.record_http_bodies and self._body_omission_reason is None

    def observe_stream(
        self, event: str, *, attempt_index: int, error: BaseException | None = None,
    ) -> None:
        """Observe an SDK stream against its bound zero-based HTTP attempt.

        Events are entered, exhausted, cancel_requested, close_started,
        close_finished, and entry_error/iteration_error/exit_error/close_error.
        Each event writes ``sdk_<event>_ns``; errors also write
        ``sdk_<event>_type``. Error observations retain only the redacted exception type. No dispatch
        or a closed scope produces no observation, never a synthetic attempt.
        """
        if event not in {
            "entered", "exhausted", "cancel_requested", "close_started", "close_finished",
            "entry_error", "iteration_error", "exit_error", "close_error",
        }:
            raise ValueError("SDK stream observation event is invalid")
        if 0 <= attempt_index < len(self._entries):
            entry = self._entries[attempt_index]
            entry.record_observation(**{f"sdk_{event}_ns": entry.elapsed_ns},
                                     **({f"sdk_{event}_type": entry.error_type(error)} if error is not None else {}))

    def wrap_model(self, model: Model) -> Model:
        """Bind supported client dispatches to this logical request's recorder.

        The owner must keep this recorder open until the request and its stream
        finish. SDK streaming may open the model in a separate consumer task.
        Model preparation, connection ownership and response handling are unchanged.
        """
        return _ClientRecordingModel(model, self)

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Bind this open scope for one transport operation in the current task."""
        token = _current_capture.set(self)
        try:
            yield
        finally:
            _current_capture.reset(token)

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


class _ClientRecordingModel(WrapperModel):
    def __init__(self, model: Model, recorder: ModelClientRequestRecorder) -> None:
        super().__init__(model)
        self._recorder = recorder

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        with self._recorder.bind():
            return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with AsyncExitStack() as stack:
            # Client dispatch happens while opening the stream. Keep token reset
            # in this task; the SDK may close the stream from another task.
            with self._recorder.bind():
                response = await stack.enter_async_context(self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context,
                ))
            yield response


@dataclass(slots=True)
class _ClientRequestEntry:
    recorder: ModelClientRequestRecorder
    request: ModelClientRequest
    started_ns: int

    @property
    def api(self) -> str:
        return self.request.api

    @property
    def record_http_bodies(self) -> bool:
        return self.request.record_http_bodies

    @property
    def active(self) -> bool:
        return self.recorder._token is not None

    @property
    def elapsed_ns(self) -> int:
        return monotonic_ns() - self.started_ns

    @staticmethod
    def error_type(error: BaseException) -> str:
        return environ.logging.redact(f"{type(error).__module__}.{type(error).__qualname__}")

    def record_observation(self, **values: JsonValue) -> None:
        if self.active:
            try:
                self.request = replace(self.request, diagnostics={**(self.request.diagnostics or {}), **values})
            except Exception:
                # Diagnostics must not change transport or SDK outcomes.
                pass

    def record_response(self, response: ModelClientResponse) -> None:
        if self.active:
            self.request = replace(self.request, response=response)


def _record_http_bodies() -> bool:
    capture = _current_capture.get()
    return capture is not None and capture._token is not None and capture._effective_record_http_bodies


def _record_client_request(
    api: str, body: bytes | None, *, capture_error: str | None = None,
) -> _ClientRequestEntry | None:
    capture = _current_capture.get()
    if capture is not None and capture._token is not None:
        started_ns = monotonic_ns()
        record_http_bodies = capture._effective_record_http_bodies
        diagnostics: dict[str, JsonValue] = {"started_at": datetime.now(timezone.utc).isoformat()}
        if capture._body_omission_reason is not None:
            diagnostics["body_omission_reason"] = capture._body_omission_reason
        request = ModelClientRequest(
            api, body if record_http_bodies else None, capture_error,
            record_http_bodies=record_http_bodies, diagnostics=diagnostics,
        )
        entry = _ClientRequestEntry(capture, request, started_ns)
        capture._entries.append(entry)
        return entry
    return None


__all__ = ["ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse"]
