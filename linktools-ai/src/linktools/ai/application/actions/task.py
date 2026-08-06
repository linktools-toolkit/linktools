#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Task user and worker actions."""


class TaskActions:
    """Forward each task action to an injected Task service."""

    def __init__(self, service: object) -> None:
        self._service = service

    async def submit(self, request: object) -> object: return await self._service.submit(request)
    async def inspect(self, task_id: str) -> object: return await self._service.inspect(task_id)
    async def claim(self, request: object) -> object: return await self._service.claim(request)
    async def renew(self, request: object) -> object: return await self._service.renew(request)
    async def complete(self, request: object) -> object: return await self._service.complete(request)
    async def fail(self, request: object) -> object: return await self._service.fail(request)


__all__ = ["TaskActions"]
