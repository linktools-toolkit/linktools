#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability.logging: LoggingObservabilitySink -- the default
ObservabilitySink + ObservabilityMetrics implementation, backed entirely by the
stdlib `logging` module.

Spans emit DEBUG start/end lines carrying span_id/parent_id (and duration_ms on
end); counters/histograms/gauges emit INFO. There is deliberately NO OTel
dependency: this is the stdlib default an OTel adapter can later plug into in
front of. The class structurally satisfies both
ObservabilitySink and ObservabilityMetrics without importing them."""

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from .tracing import Span, _mint_span_id

_DEFAULT_LOGGER_NAME = "linktools.ai.observability"


class LoggingObservabilitySink:
    """Implements both ObservabilitySink and ObservabilityMetrics via stdlib
    logging. Span durations are tracked in-process by span_id."""

    def __init__(self, logger: "logging.Logger | None" = None) -> None:
        self._logger = logger or logging.getLogger(_DEFAULT_LOGGER_NAME)
        self._started_at: "dict[str, datetime]" = {}

    # --- ObservabilitySink ------------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        attributes: "Mapping[str, Any] | None" = None,
        parent: "Span | None" = None,
    ) -> Span:
        span_id = _mint_span_id()
        now = datetime.now(timezone.utc)
        self._started_at[span_id] = now
        parent_id = parent.span_id if parent is not None else None
        self._logger.debug(
            "span.start name=%s span_id=%s parent_id=%s attrs=%s",
            name,
            span_id,
            parent_id,
            dict(attributes or {}),
        )
        return Span(
            name=name,
            span_id=span_id,
            parent_id=parent_id,
            started_at=now,
            attributes=dict(attributes or {}),
        )

    def record_event(
        self, name: str, *, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self._logger.debug("event name=%s attrs=%s", name, dict(attributes or {}))

    def end_span(self, span: Span) -> None:
        started = self._started_at.pop(span.span_id, None)
        duration_ms: "float | None" = None
        if started is not None:
            duration_ms = round(
                (datetime.now(timezone.utc) - started).total_seconds() * 1000, 2
            )
        self._logger.debug(
            "span.end name=%s span_id=%s duration_ms=%s",
            span.name,
            span.span_id,
            duration_ms,
        )

    # --- ObservabilityMetrics --------------------------------------------

    def counter(
        self,
        name: str,
        *,
        value: int = 1,
        attributes: "Mapping[str, Any] | None" = None,
    ) -> None:
        self._logger.info(
            "counter name=%s value=%s attrs=%s", name, value, dict(attributes or {})
        )

    def histogram(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self._logger.info(
            "histogram name=%s value=%s attrs=%s", name, value, dict(attributes or {})
        )

    def gauge(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        self._logger.info(
            "gauge name=%s value=%s attrs=%s", name, value, dict(attributes or {})
        )
