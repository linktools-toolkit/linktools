#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Semantic Trace append and immutable Snapshot coordination."""

from ...domain.trace import RunSnapshot, TraceEvent


class TraceService:
    def __init__(self, repository: object) -> None:
        self._repository = repository

    async def append(self, event: TraceEvent) -> TraceEvent:
        return await self._repository.append(event)

    async def list_after(self, execution_id: str, after_sequence: int = 0) -> "tuple[TraceEvent, ...]":
        return await self._repository.list(execution_id, after_sequence)

    async def snapshot(self, snapshot: RunSnapshot) -> RunSnapshot:
        return await self._repository.commit_snapshot(snapshot)


__all__ = ["TraceService"]
