#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical model metric error classification regressions."""

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._metric_capability import _http_error_code, _model_error_code


def test_model_metric_http_error_classification_is_canonical() -> None:
    assert _http_error_code(408) is ErrorCode.MODEL_TIMEOUT
    assert _http_error_code(429) is ErrorCode.MODEL_RATE_LIMITED
    assert _http_error_code(500) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(503) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(400) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(499) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(302) is ErrorCode.MODEL_API_ERROR


def test_model_metric_preserves_runtime_ai_error_code() -> None:
    assert (
        _model_error_code(AIError(ErrorCode.STORAGE_UNAVAILABLE))
        == ErrorCode.STORAGE_UNAVAILABLE.value
    )


def test_model_metric_unknown_error_matches_runtime_internal_error() -> None:
    assert _model_error_code(RuntimeError("provider wrapper bug")) == ErrorCode.INTERNAL_ERROR.value
