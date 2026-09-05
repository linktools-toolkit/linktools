#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatibility regressions for the observability refactor."""

from linktools.ai.errors import ErrorCode


def test_removed_middleware_error_code_remains_decodable() -> None:
    assert ErrorCode("MIDDLEWARE_FAILED") is ErrorCode.MIDDLEWARE_FAILED
