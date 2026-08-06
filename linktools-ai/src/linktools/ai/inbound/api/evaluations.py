#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"evaluations HTTP protocol mapping."


class EvaluationsApi:
    def __init__(self, application: object) -> None:
        self._application = application

    async def handle(self, request: object) -> object:
        return await self._application.handle(request)


__all__ = ["EvaluationsApi"]
