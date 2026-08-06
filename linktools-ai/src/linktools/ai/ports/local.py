#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local execution protocol."""

from typing import Protocol


class LocalAgentExecutorPort(Protocol):
    async def run(self, request: object) -> object: ...
    async def resume(self, request: object) -> object: ...
    async def cancel(self, execution_id: str) -> object: ...


__all__ = ["LocalAgentExecutorPort"]
