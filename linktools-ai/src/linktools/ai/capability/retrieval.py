#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrieval provider protocol."""

from typing import Protocol


class RetrievalProvider(Protocol):
    async def search(self, query: str, limit: int = 10) -> 'tuple[str, ...]': ...


__all__ = ["RetrievalProvider"]
