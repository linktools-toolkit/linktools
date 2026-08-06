#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auditable deletion action boundary."""


class DeleteData:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, request: object) -> object: return await self._repository.create(request)


__all__ = ["DeleteData"]
