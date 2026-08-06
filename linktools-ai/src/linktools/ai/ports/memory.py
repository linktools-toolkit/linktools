#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tenant-scoped memory protocol."""

from typing import Protocol


class MemoryStore(Protocol):
    async def get(self, namespace: str, key: str) -> "object | None": ...
    async def put(self, namespace: str, key: str, value: object) -> None: ...


__all__ = ["MemoryStore"]
