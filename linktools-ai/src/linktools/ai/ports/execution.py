#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution projection protocol."""

from typing import Protocol


class ExecutionRepository(Protocol):
    async def upsert_projection(self, view: object) -> object: ...
    async def get(self, execution_id: str) -> "object | None": ...
    async def repair(self, execution_id: str) -> object: ...


__all__ = ["ExecutionRepository"]
