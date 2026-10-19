#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional owner-scoped write-buffer capability."""

from collections.abc import Hashable, Sequence
from typing import Protocol, TypeAlias, runtime_checkable

WriteBufferEntry: TypeAlias = tuple[str, Hashable, bytes | None]


@runtime_checkable
class WriteBufferBackend(Protocol):
    """Ordered, non-destructive pending writes retained until acknowledgment."""

    async def append(
        self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool,
    ) -> bool: ...

    async def read_batch(self, scope: str, *, limit: int) -> tuple[WriteBufferEntry, ...]: ...

    async def ack(self, scope: str, ids: Sequence[str]) -> None: ...


__all__ = ["WriteBufferBackend", "WriteBufferEntry"]
