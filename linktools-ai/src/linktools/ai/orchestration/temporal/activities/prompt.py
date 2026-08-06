#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Temporal prompt Activity boundary."


class ResolvePromptActivity:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def execute(self, request: object) -> object:
        return await self._operation(request)
