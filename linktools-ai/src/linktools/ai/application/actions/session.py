#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session user actions."""

from ...domain.execution import ExecutionHandle, Page


class SessionActions:
    """One-to-one action boundary for Session lifecycle calls."""

    def __init__(self, service: object) -> None:
        self._service = service

    async def create(self, request: object) -> object: return await self._service.create(request)
    async def get(self, session_id: str) -> object: return await self._service.get(session_id)
    async def list(self, request: object) -> "Page[object]": return await self._service.list(request)
    async def load(self, session_id: str, request: object) -> object: return await self._service.load(session_id, request)
    async def resume(self, session_id: str, request: object) -> ExecutionHandle: return await self._service.resume(session_id, request)
    async def fork(self, session_id: str, request: object) -> object: return await self._service.fork(session_id, request)
    async def update(self, session_id: str, request: object) -> object: return await self._service.update(session_id, request)
    async def close(self, session_id: str, request: object) -> object: return await self._service.close(session_id, request)


__all__ = ["SessionActions"]
