#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session CLI input adapter."""


class SessionCommand:
    def __init__(self, application: object) -> None:
        self._application = application

    async def execute(self, request: object) -> object:
        return await self._application.execute(request)


__all__ = ["SessionCommand"]
