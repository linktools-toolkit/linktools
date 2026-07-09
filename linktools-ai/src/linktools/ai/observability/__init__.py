#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability: tracing/metrics/logging Protocol boundary for linktools.ai.

This package defines the abstractions an OpenTelemetry adapter can later plug
into (Decision D-2): ObservabilitySink (spans), ObservabilityMetrics
(counters/histograms/gauges), and LoggingObservabilitySink, the stdlib-only
default. Submodules are imported explicitly; nothing is re-exported here."""
