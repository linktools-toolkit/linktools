#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local execution boundary for generated static Bundles."""

from typing import Protocol


class AgentExecutor(Protocol):
    async def run(self, request: object) -> object: ...
    async def resume(self, request: object) -> object: ...
    async def cancel(self, execution_id: str) -> object: ...


class LocalAgentExecutor:
    """Call an explicitly injected bundle callable; no scheduler is owned."""

    def __init__(self, runner: object) -> None:
        self._runner = runner

    async def run(self, request: object) -> object:
        return await self._runner.run(request)

    async def resume(self, request: object) -> object:
        return await self._runner.resume(request)

    async def cancel(self, execution_id: str) -> object:
        return await self._runner.cancel(execution_id)


__all__ = ["AgentExecutor", "LocalAgentExecutor"]
