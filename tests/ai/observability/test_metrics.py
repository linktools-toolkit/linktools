#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for observability.metrics: ObservabilityMetrics Protocol is
runtime-checkable and satisfied by LoggingObservabilitySink."""

from linktools.ai.observability.logging import LoggingObservabilitySink
from linktools.ai.observability.metrics import ObservabilityMetrics


def test_observability_metrics_is_runtime_checkable():
    # A class that implements counter/histogram/gauge satisfies the Protocol.
    assert isinstance(LoggingObservabilitySink(), ObservabilityMetrics)


def test_bare_object_does_not_satisfy_metrics_protocol():
    class _Bare:
        pass

    assert not isinstance(_Bare(), ObservabilityMetrics)
