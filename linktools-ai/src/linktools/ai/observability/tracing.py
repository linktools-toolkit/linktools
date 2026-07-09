#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability.tracing: span abstraction for linktools.ai.

ObservabilitySink is the Protocol boundary an OpenTelemetry adapter can later
plug into (Decision D-2); it intentionally has no OTel dependency. Span nesting
is tracked with a contextvar so callers do not have to thread parent spans
through every call site -- use the `use_span` async context manager to set the
current span for the duration of a body, and `current_span()` to read it."""

import contextlib
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Span:
    """An in-flight or completed unit of work. `attributes` is captured as a
    plain mapping at construction time."""

    name: str
    span_id: str
    parent_id: "str | None"
    started_at: datetime
    attributes: "Mapping[str, Any]"


_span_context: "ContextVar[Span | None]" = ContextVar("linktools_ai_span", default=None)


def current_span() -> "Span | None":
    """Return the span active in the current context, or None."""
    return _span_context.get()


@runtime_checkable
class ObservabilitySink(Protocol):
    """Tracing boundary: start/record/end spans. Implementations are free to
    forward to OTel, emit log lines, or noop."""

    def start_span(
        self,
        name: str,
        *,
        attributes: "Mapping[str, Any] | None" = None,
        parent: "Span | None" = None,
    ) -> Span:
        """Start a span, optionally parented to `parent`. Returns the Span;
        the caller decides whether to set it as the current contextvar."""
        ...

    def record_event(self, name: str, *, attributes: "Mapping[str, Any] | None" = None) -> None:
        """Record a discrete event under the current span (if any)."""
        ...

    def end_span(self, span: Span) -> None:
        """End a previously started span. Must be safe to call exactly once."""
        ...


def _mint_span_id() -> str:
    """Return a short, unique-enough hex id for a span (no OTel trace context)."""
    return uuid.uuid4().hex[:16]


@contextlib.asynccontextmanager
async def use_span(
    sink: ObservabilitySink,
    name: str,
    *,
    attributes: "Mapping[str, Any] | None" = None,
):
    """Async context manager: start a span parented to `current_span()`, set it
    as the contextvar for the body, and end it on exit -- even on exception.

    Use as:

        async with use_span(sink, "op") as span:
            ...
    """
    parent = current_span()
    span = sink.start_span(name, attributes=attributes, parent=parent)
    token = _span_context.set(span)
    try:
        yield span
    finally:
        _span_context.reset(token)
        sink.end_span(span)
