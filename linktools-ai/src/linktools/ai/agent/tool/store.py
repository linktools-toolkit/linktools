#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural tool-state contract implemented directly by backends."""


from datetime import timedelta
from typing import Protocol

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...json import JsonValue
    from ...storage.database import CoordinationScope
    from .models import ToolOperation

class ToolStateStore(Protocol):
    coordination_scope: "CoordinationScope"

    async def prepare(self, operation: "ToolOperation") -> "ToolOperation": ...

    async def get(self, operation_id: str) -> "ToolOperation | None": ...

    async def claim(
        self,
        operation_id: str,
        *,
        owner: str,
        duration: timedelta = timedelta(minutes=5),
    ) -> "ToolOperation": ...

    async def renew(
        self,
        operation_id: str,
        *,
        owner: str,
        fence: int,
        duration: timedelta = timedelta(minutes=5),
    ) -> "ToolOperation": ...

    async def complete(
        self, operation_id: str, *, owner: str, fence: int, result: "JsonValue"
    ) -> "ToolOperation": ...

    async def fail(
        self, operation_id: str, *, owner: str, fence: int, error: "JsonValue"
    ) -> "ToolOperation": ...

    async def mark_indeterminate(
        self, operation_id: str, *, owner: str, fence: int, error: "JsonValue"
    ) -> "ToolOperation": ...


__all__ = ["ToolStateStore"]
