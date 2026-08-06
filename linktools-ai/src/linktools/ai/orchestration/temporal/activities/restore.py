#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Temporal Activity boundary for workspace restore."""


class RestoreWorkspaceActivity:
    def __init__(self, service: object) -> None:
        self._service = service

    async def execute(self, request: object) -> object:
        return await self._service.restore(request)


__all__ = ["RestoreWorkspaceActivity"]
