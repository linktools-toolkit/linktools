#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local run, resume and reconciliation actions."""


class LocalActions:
    """Forward local-only actions to the injected executor boundary."""

    def __init__(self, executor: object) -> None:
        self._executor = executor

    async def run(self, request: object) -> object: return await self._executor.run(request)
    async def resume(self, request: object) -> object: return await self._executor.resume(request)
    async def cancel(self, execution_id: str) -> object: return await self._executor.cancel(execution_id)


__all__ = ["LocalActions"]
