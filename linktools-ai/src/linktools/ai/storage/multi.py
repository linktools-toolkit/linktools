#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reader/writer protocols and bounded batch fallback.

The layer topology (primary-first ordered fallback) is owned by
``StorageComposition``; this module owns the generic reader/writer Protocols
and the per-backend batch fallback that fans single ``get`` calls out under
bounded concurrency when a backend does not implement ``get_many``."""

import asyncio
from collections.abc import Mapping
from typing import Protocol, TypeVar, runtime_checkable

from .revision import RevisionT

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")


@runtime_checkable
class StorageReader(Protocol[KeyT, ValueT, InfoT]):
    async def get(self, key: KeyT) -> "ValueT | None": ...

    async def list_info(self) -> "tuple[InfoT, ...]": ...


@runtime_checkable
class BatchStorageReader(Protocol[KeyT, ValueT]):
    async def get_many(
        self,
        keys: "tuple[KeyT, ...]",
    ) -> "Mapping[KeyT, ValueT]": ...


@runtime_checkable
class StorageWriter(Protocol[KeyT, ValueT, RevisionT]):
    """``put``/``delete``/``reset`` always report the revision the write
    landed at -- computed in the same transaction as the write -- so
    ``StorageComposition`` can use it directly instead of paying a separate
    ``head_revision()`` probe after every write solely to learn it. A backend
    with no revision concept (e.g. a local directory backend) reports
    ``None``."""

    async def put(self, value: ValueT) -> "tuple[ValueT, RevisionT | None]": ...

    async def delete(self, key: KeyT) -> "RevisionT | None": ...

    async def reset(self, values: "tuple[ValueT, ...]") -> "RevisionT | None": ...


@runtime_checkable
class BatchStorageWriter(Protocol[KeyT, ValueT, RevisionT]):
    async def apply_batch(
        self,
        puts: "tuple[ValueT, ...]",
        deletes: "tuple[KeyT, ...]",
    ) -> "RevisionT | None": ...


async def batch_get(
    reader: object,
    keys: "tuple[KeyT, ...]",
    *,
    concurrency: int = 8,
    timeout: "float | None" = None,
) -> "dict[KeyT, ValueT]":
    """Resolve ``keys`` from one reader. Uses ``get_many`` when the reader
    supports it; otherwise runs single ``get`` calls under a bounded semaphore.
    ``None`` results are dropped (a miss). ``concurrency`` must be positive.

    ``timeout`` bounds each backend call (the whole ``get_many`` call, or each
    individual ``get`` in the fallback). ``None`` (default) waits forever, matching
    the prior behavior; a positive number raises ``TimeoutError`` if a call exceeds
    it, so a stuck backend cannot hold all semaphore permits indefinitely."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not keys:
        return {}
    if isinstance(reader, BatchStorageReader):
        loaded = await asyncio.wait_for(reader.get_many(keys), timeout=timeout)
        return {key: value for key, value in loaded.items() if value is not None}
    semaphore = asyncio.Semaphore(concurrency)

    async def load(key: KeyT) -> "tuple[KeyT, ValueT | None]":
        async with semaphore:
            value = await asyncio.wait_for(reader.get(key), timeout=timeout)  # type: ignore[attr-defined]
            return key, value

    return {
        key: value
        for key, value in await asyncio.gather(*(load(key) for key in keys))
        if value is not None
    }


__all__ = [
    "BatchStorageReader",
    "BatchStorageWriter",
    "StorageReader",
    "StorageWriter",
    "batch_get",
]
