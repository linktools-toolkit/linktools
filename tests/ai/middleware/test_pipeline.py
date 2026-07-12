#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/middleware/test_pipeline.py"""

import pytest

from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline


class _RecordingMiddleware(Middleware):
    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self.log = log

    async def before_run(self, context) -> None:
        self.log.append(f"{self.name}.before_run")

    async def after_run(self, context, result):
        self.log.append(f"{self.name}.after_run")
        return result

    async def on_error(self, context, error) -> None:
        self.log.append(f"{self.name}.on_error")


@pytest.mark.asyncio
async def test_before_run_fires_in_registration_order():
    log = []
    pipeline = MiddlewarePipeline(
        middlewares=(
            _RecordingMiddleware("M1", log),
            _RecordingMiddleware("M2", log),
            _RecordingMiddleware("M3", log),
        )
    )
    await pipeline.run_before_run(context=None)
    assert log == ["M1.before_run", "M2.before_run", "M3.before_run"]


@pytest.mark.asyncio
async def test_after_run_fires_in_reverse_order():
    log = []
    pipeline = MiddlewarePipeline(
        middlewares=(
            _RecordingMiddleware("M1", log),
            _RecordingMiddleware("M2", log),
            _RecordingMiddleware("M3", log),
        )
    )
    result = await pipeline.run_after_run(context=None, result="initial")
    assert log == ["M3.after_run", "M2.after_run", "M1.after_run"]
    assert result == "initial"


@pytest.mark.asyncio
async def test_on_error_calls_all_middlewares_even_if_one_raises():
    log = []

    class _RaisingMiddleware(Middleware):
        async def on_error(self, context, error) -> None:
            log.append("raiser.on_error")
            raise RuntimeError("boom")

    pipeline = MiddlewarePipeline(
        middlewares=(
            _RecordingMiddleware("M1", log),
            _RaisingMiddleware(),
            _RecordingMiddleware("M3", log),
        )
    )
    await pipeline.run_on_error(context=None, error=ValueError("original"))
    assert log == ["M3.on_error", "raiser.on_error", "M1.on_error"]


@pytest.mark.asyncio
async def test_middleware_base_methods_are_all_no_ops_by_default():
    middleware = Middleware()
    await middleware.before_run(context=None)
    result = await middleware.after_run(context=None, result="x")
    assert result == "x"
    await middleware.on_error(context=None, error=ValueError("x"))
