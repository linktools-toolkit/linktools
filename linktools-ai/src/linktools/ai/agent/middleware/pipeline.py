#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiddlewarePipeline: orchestrates before_run/after_run/on_error and
before_model/after_model/before_tool/after_tool across a registered set of
Middlewares. Entering: M1 -> M2 -> M3 (registration order). Returning/erroring:
M3 -> M2 -> M1 (reverse order)."""

from typing import Any

from .base import Middleware


class MiddlewarePipeline:
    def __init__(self, *, middlewares: "tuple[Middleware, ...]") -> None:
        self._middlewares = middlewares

    async def run_before_run(self, context: Any) -> None:
        for middleware in self._middlewares:
            await middleware.before_run(context)

    async def run_after_run(self, context: Any, result: Any) -> Any:
        current = result
        for middleware in reversed(self._middlewares):
            current = await middleware.after_run(context, current)
        return current

    async def run_on_error(self, context: Any, error: Exception) -> None:
        for middleware in reversed(self._middlewares):
            try:
                await middleware.on_error(context, error)
            except Exception:
                continue

    async def run_before_model(self, context: Any, request: Any) -> Any:
        current = request
        for middleware in self._middlewares:
            current = await middleware.before_model(context, current)
        return current

    async def run_after_model(self, context: Any, response: Any) -> Any:
        current = response
        for middleware in reversed(self._middlewares):
            current = await middleware.after_model(context, current)
        return current

    async def run_before_tool(self, context: Any, request: Any) -> Any:
        current = request
        for middleware in self._middlewares:
            current = await middleware.before_tool(context, current)
        return current

    async def run_after_tool(self, context: Any, request: Any, result: Any) -> Any:
        current = result
        for middleware in reversed(self._middlewares):
            current = await middleware.after_tool(context, request, current)
        return current
