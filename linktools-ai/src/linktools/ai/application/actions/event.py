#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Durable event query and live stream actions."""


class ListExecutionEvents:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, execution_id: str, after_sequence: int = 0, limit: int = 100) -> object: return await self._service.list_after(execution_id, after_sequence, min(limit, 200))


class StreamExecutionEvents:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, execution_id: str, after_sequence: int = 0) -> object: return await self._service.stream(execution_id, after_sequence)


__all__ = ["ListExecutionEvents", "StreamExecutionEvents"]
