#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pinned Worker route protocol."""

from typing import Protocol


class WorkerRouteRepository(Protocol):
    async def resolve(self, bundle_id: str) -> "object | None": ...
    async def heartbeat(self, bundle_id: str) -> object: ...


__all__ = ["WorkerRouteRepository"]
