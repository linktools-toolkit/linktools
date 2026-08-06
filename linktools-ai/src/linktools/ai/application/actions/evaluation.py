#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation run, comparison and replay actions."""


class RunEvaluation:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, request: object) -> object: return await self._service.run(request)


class CompareEvaluation:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, request: object) -> object: return await self._service.compare(request)


class ReplayEvaluation:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, request: object) -> object: return await self._service.replay(request)


class InspectEvaluation:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, request: object) -> object: return await self._service.inspect(request)


__all__ = ["CompareEvaluation", "InspectEvaluation", "ReplayEvaluation", "RunEvaluation"]
