#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"executions HTTP protocol mapping."


class ExecutionsApi:
    def __init__(self, application: object) -> None:
        self._application = application

    async def handle(self, request: object) -> object:
        return await self._application.handle(request)


__all__ = ["ExecutionsApi"]
