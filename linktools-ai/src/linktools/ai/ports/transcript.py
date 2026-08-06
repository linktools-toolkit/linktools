#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Transcript persistence protocol."""

from typing import Protocol


class TranscriptRepository(Protocol):
    async def append(self, segment: object) -> object: ...
    async def list(self, execution_id: str) -> "tuple[object, ...]": ...
    async def search(self, execution_id: str, query: str) -> "tuple[object, ...]": ...


__all__ = ["TranscriptRepository"]
