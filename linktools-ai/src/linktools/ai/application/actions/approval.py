#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Submit one approval decision."""


class ApproveExecution:
    def __init__(self, service: object) -> None:
        self._service = service

    async def execute(self, execution_id: str, request: object) -> object:
        return await self._service.decide(execution_id, request)


__all__ = ["ApproveExecution"]
