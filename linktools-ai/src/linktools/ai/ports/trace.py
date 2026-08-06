#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Semantic Trace and immutable Snapshot protocol."""

from typing import Protocol


class TraceRepository(Protocol):
    async def append(self, event: object) -> object: ...
    async def list(self, execution_id: str, after_sequence: int = 0) -> "tuple[object, ...]": ...
    async def commit_snapshot(self, snapshot: object) -> object: ...


__all__ = ["TraceRepository"]
