#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for observability.logging: LoggingObservabilitySink emits DEBUG span
lines and INFO metric lines through stdlib logging."""

import logging

from linktools.ai.observability.logging import LoggingObservabilitySink
from linktools.ai.observability.tracing import Span


def test_start_span_emits_debug_with_name_and_span_id(caplog):
    sink = LoggingObservabilitySink()
    with caplog.at_level(logging.DEBUG, logger="linktools.ai.observability"):
        span = sink.start_span("do_thing", attributes={"a": 1})
    assert isinstance(span, Span)
    assert span.name == "do_thing"
    assert "span.start" in caplog.text
    assert "name=do_thing" in caplog.text
    assert f"span_id={span.span_id}" in caplog.text
    assert "parent_id=None" in caplog.text


def test_end_span_emits_debug_with_duration_ms(caplog):
    sink = LoggingObservabilitySink()
    span = sink.start_span("op")
    with caplog.at_level(logging.DEBUG, logger="linktools.ai.observability"):
        sink.end_span(span)
    assert "span.end" in caplog.text
    assert "name=op" in caplog.text
    assert "duration_ms=" in caplog.text


def test_counter_emits_info_line(caplog):
    sink = LoggingObservabilitySink()
    with caplog.at_level(logging.INFO, logger="linktools.ai.observability"):
        sink.counter("hits", value=3, attributes={"route": "/x"})
    assert "counter" in caplog.text
    assert "name=hits" in caplog.text
    assert "value=3" in caplog.text


def test_histogram_emits_info_line(caplog):
    sink = LoggingObservabilitySink()
    with caplog.at_level(logging.INFO, logger="linktools.ai.observability"):
        sink.histogram("latency_ms", value=12.5)
    assert "histogram" in caplog.text
    assert "name=latency_ms" in caplog.text
    assert "value=12.5" in caplog.text


def test_gauge_emits_info_line(caplog):
    sink = LoggingObservabilitySink()
    with caplog.at_level(logging.INFO, logger="linktools.ai.observability"):
        sink.gauge("queue_depth", value=7.0)
    assert "gauge" in caplog.text
    assert "name=queue_depth" in caplog.text
    assert "value=7.0" in caplog.text


def test_record_event_emits_debug_line(caplog):
    sink = LoggingObservabilitySink()
    with caplog.at_level(logging.DEBUG, logger="linktools.ai.observability"):
        sink.record_event("thing_happened", attributes={"k": "v"})
    assert "event" in caplog.text
    assert "name=thing_happened" in caplog.text


def test_default_logger_name_when_none_supplied():
    sink = LoggingObservabilitySink(logger=None)
    assert sink._logger.name == "linktools.ai.observability"
