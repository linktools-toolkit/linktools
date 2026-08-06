#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Temporal Activity boundary for workspace checkpoints."""


class CheckpointWorkspaceActivity:
    def __init__(self, service: object) -> None:
        self._service = service

    async def execute(self, request: object) -> object:
        return await self._service.capture(request)


__all__ = ["CheckpointWorkspaceActivity"]
