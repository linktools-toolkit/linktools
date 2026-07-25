#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The storage kernel's content-cache Protocol.

A ContentCache holds only bytes, keyed by an opaque caller-chosen string; it
defines no object existence, carries no business model, produces no revision,
and can be cleared at any time without affecting correctness -- a cache
failure must never change the facts a caller reads from the origin."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContentCache(Protocol):
    async def get(self, key: str) -> "bytes | None": ...

    async def put(self, key: str, content: bytes) -> None: ...

    async def delete(self, key: str) -> None: ...


__all__: "list[str]" = ["ContentCache"]
