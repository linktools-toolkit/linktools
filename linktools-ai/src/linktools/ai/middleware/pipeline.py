#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiddlewarePipeline: orchestrates before_run/after_run/on_error across a
registered set of Middlewares. Entering: M1 -> M2 -> M3 (registration order).
Returning/erroring: M3 -> M2 -> M1 (reverse order), per spec section 24."""

from .base import Middleware


class MiddlewarePipeline:
    def __init__(self, *, middlewares: "tuple[Middleware, ...]") -> None:
        self._middlewares = middlewares

    async def run_before_run(self, context) -> None:
        for middleware in self._middlewares:
            await middleware.before_run(context)

    async def run_after_run(self, context, result):
        current = result
        for middleware in reversed(self._middlewares):
            current = await middleware.after_run(context, current)
        return current

    async def run_on_error(self, context, error: Exception) -> None:
        for middleware in reversed(self._middlewares):
            try:
                await middleware.on_error(context, error)
            except Exception:
                continue
