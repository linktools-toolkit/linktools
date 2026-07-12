#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for observability.tracing: Span dataclass, contextvar nesting, and the
use_span async context manager (end-on-exception)."""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any, Mapping

import pytest

from linktools.ai.observability.logging import LoggingObservabilitySink
from linktools.ai.observability.tracing import (
    ObservabilitySink,
    Span,
    current_span,
    use_span,
)


class _RecordingSink:
    """Fake sink that records every call so tests can assert on nesting and
    end-on-exception behaviour without depending on logging."""

    def __init__(self) -> None:
        self.started: list[Span] = []
        self.events: list[tuple[str, dict]] = []
        self.ended: list[Span] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: "Mapping[str, Any] | None" = None,
        parent: "Span | None" = None,
    ) -> Span:
        span = Span(
            name=name,
            span_id=f"id-{len(self.started)}",
            parent_id=parent.span_id if parent is not None else None,
            started_at=datetime.now(timezone.utc),
            attributes=dict(attributes or {}),
        )
        self.started.append(span)
        return span

    def record_event(
        self, name: str, *, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    def end_span(self, span: Span) -> None:
        self.ended.append(span)


def test_span_is_frozen_and_constructs_with_all_fields():
    span = Span(
        name="op",
        span_id="abc",
        parent_id=None,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        attributes={"k": "v"},
    )
    assert span.name == "op"
    assert span.span_id == "abc"
    assert span.parent_id is None
    assert dict(span.attributes) == {"k": "v"}
    with pytest.raises(FrozenInstanceError):
        span.name = "other"  # type: ignore[misc]


def test_current_span_returns_none_by_default():
    assert current_span() is None


def test_use_span_sets_and_restores_current_span():
    sink = _RecordingSink()
    seen_inside: list[Span | None] = []

    async def _run() -> None:
        assert current_span() is None
        async with use_span(sink, "op") as span:
            seen_inside.append(current_span())
            assert span.name == "op"
        # restored after exit
        assert current_span() is None

    asyncio.run(_run())
    assert seen_inside[0] is not None
    assert seen_inside[0].name == "op"
    assert len(sink.started) == 1
    assert len(sink.ended) == 1


def test_nested_use_span_parents_inner_to_outer():
    sink = _RecordingSink()

    async def _run() -> None:
        async with use_span(sink, "outer"):
            outer = current_span()
            assert outer is not None
            async with use_span(sink, "inner"):
                inner = current_span()
                assert inner is not None
                assert inner.parent_id == outer.span_id
            # back to outer after inner exits
            assert current_span() is outer

    asyncio.run(_run())
    assert [s.name for s in sink.started] == ["outer", "inner"]
    # LIFO end order: inner ends before outer
    assert [s.name for s in sink.ended] == ["inner", "outer"]


def test_use_span_ends_span_even_when_body_raises():
    sink = _RecordingSink()

    async def _run() -> None:
        with pytest.raises(RuntimeError):
            async with use_span(sink, "op"):
                raise RuntimeError("boom")

    asyncio.run(_run())
    assert len(sink.started) == 1
    assert len(sink.ended) == 1
    assert sink.ended[0].name == "op"
    # contextvar restored even on exception
    assert current_span() is None


def test_observability_sink_is_runtime_checkable():
    assert isinstance(LoggingObservabilitySink(), ObservabilitySink)
