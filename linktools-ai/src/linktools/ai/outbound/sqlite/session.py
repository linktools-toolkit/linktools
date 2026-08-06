#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Local ACP Session persistence adapter."


class LocalFileACPSessionStore:
    def __init__(self, path: object) -> None:
        self._path = path

    async def save(self, session: object) -> None:
        await self._path.save(session)

    async def load(self, session_id: str) -> "object | None":
        return await self._path.load(session_id)

    async def delete(self, session_id: str) -> None:
        await self._path.delete(session_id)


__all__ = ["LocalFileACPSessionStore"]
