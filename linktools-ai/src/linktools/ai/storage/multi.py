#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reader protocols and bounded batch fallback.

The layer topology (primary-first ordered fallback) is owned by
``StorageComposition``; this module owns only the generic reader/writer
Protocols and the per-backend batch fallback that fans single ``get`` calls
out under bounded concurrency when a backend does not implement
``get_many``."""


import asyncio
from collections.abc import Mapping
from typing import Protocol, TypeVar, runtime_checkable

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
class StorageWriter(Protocol[KeyT, ValueT]):
    async def put(self, value: ValueT) -> ValueT: ...

    async def delete(self, key: KeyT) -> None: ...

    async def reset(self, values: "tuple[ValueT, ...]") -> None: ...


async def batch_get(
    reader: object,
    keys: "tuple[KeyT, ...]",
    *,
    concurrency: int = 8,
) -> "dict[KeyT, ValueT]":
    """Resolve ``keys`` from one reader. Uses ``get_many`` when the reader
    supports it; otherwise runs single ``get`` calls under a bounded semaphore.
    ``None`` results are dropped (a miss). ``concurrency`` must be positive."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if isinstance(reader, BatchStorageReader):
        loaded = await reader.get_many(keys)
        return {key: value for key, value in loaded.items() if value is not None}
    semaphore = asyncio.Semaphore(concurrency)

    async def load(key: KeyT) -> "tuple[KeyT, ValueT | None]":
        async with semaphore:
            return key, await reader.get(key)  # type: ignore[attr-defined]

    return {
        key: value
        for key, value in await asyncio.gather(*(load(key) for key in keys))
        if value is not None
    }


__all__ = [
    "BatchStorageReader",
    "StorageReader",
    "StorageWriter",
    "batch_get",
]
