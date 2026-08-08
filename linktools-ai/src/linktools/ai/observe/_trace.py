#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace DTOs and an in-memory recorder."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TraceItem:
    execution_id: str
    sequence: int
    kind: str
    timestamp: datetime
    payload: str


class TraceRecorder(Protocol):
    async def append(self, item: TraceItem) -> TraceItem: ...
    async def list(self, execution_id: str, after_sequence: int = 0) -> 'tuple[TraceItem, ...]': ...


class InMemoryTraceRecorder:
    def __init__(self) -> None:
        self._items: dict[str, list[TraceItem]] = {}

    async def append(self, item: TraceItem) -> TraceItem:
        items = self._items.setdefault(item.execution_id, [])
        if items and item.sequence < items[-1].sequence:
            raise ValueError("trace sequence must be monotonic")
        if items and item.sequence == items[-1].sequence:
            return items[-1]
        items.append(item)
        return item

    async def list(self, execution_id: str, after_sequence: int = 0) -> 'tuple[TraceItem, ...]':
        return tuple(item for item in self._items.get(execution_id, ()) if item.sequence > after_sequence)


__all__ = ["InMemoryTraceRecorder", "TraceItem", "TraceRecorder"]
