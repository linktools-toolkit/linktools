#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical model metric error classification regressions."""

from linktools.ai.errors import ErrorCode
from linktools.ai.runtime._metric_capability import _http_error_code


def test_model_metric_http_error_classification_is_canonical() -> None:
    assert _http_error_code(408) is ErrorCode.MODEL_TIMEOUT
    assert _http_error_code(429) is ErrorCode.MODEL_RATE_LIMITED
    assert _http_error_code(500) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(503) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(400) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(499) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(302) is ErrorCode.MODEL_API_ERROR
