#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability.metrics: counter/histogram/gauge abstraction for linktools.ai.

ObservabilityMetrics is the Protocol boundary an OpenTelemetry adapter can
later plug into; it has no OTel dependency. A single class can
implement both this and ObservabilitySink -- LoggingObservabilitySink does."""

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ObservabilityMetrics(Protocol):
    """Metrics boundary: named numeric instruments with optional attributes."""

    def counter(self, name: str, *, value: int = 1, attributes: "Mapping[str, Any] | None" = None) -> None:
        """Increment a monotonically-increasing counter by `value` (>= 0)."""
        ...

    def histogram(self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None) -> None:
        """Record an observation into a distribution (e.g. latency)."""
        ...

    def gauge(self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None) -> None:
        """Set a instantaneous gauge to `value` (may go up or down)."""
        ...
