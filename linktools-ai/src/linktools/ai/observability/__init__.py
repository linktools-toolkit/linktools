#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability: tracing/metrics/logging Protocol boundary for linktools.ai.

This package defines the abstractions an OpenTelemetry adapter can later plug
into: ObservabilitySink (spans), ObservabilityMetrics
(counters/histograms/gauges), and LoggingObservabilitySink, the stdlib-only
default."""

from .logging import LoggingObservabilitySink
from .metrics import InMemoryMetrics, ObservabilityMetrics
from .tracing import ObservabilitySink, Span, current_span

__all__ = [
    "InMemoryMetrics",
    "LoggingObservabilitySink",
    "ObservabilityMetrics",
    "ObservabilitySink",
    "Span",
    "current_span",
]
