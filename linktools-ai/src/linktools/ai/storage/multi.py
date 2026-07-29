"""Composable readers and overlay resolution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from collections.abc import Awaitable, Callable
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

from .revision import RevisionSource

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")
StorageInitializer = Callable[..., Awaitable[None]]


@runtime_checkable
class StorageReader(Protocol[KeyT, ValueT, InfoT]):
    async def get(self, key: KeyT) -> ValueT | None: ...

    async def list_info(self) -> tuple[InfoT, ...]: ...


@runtime_checkable
class BatchStorageReader(Protocol[KeyT, ValueT]):
    async def get_many(
        self,
        keys: tuple[KeyT, ...],
    ) -> Mapping[KeyT, ValueT]: ...


@runtime_checkable
class StorageWriter(Protocol[KeyT, ValueT]):
    async def put(self, value: ValueT) -> ValueT: ...

    async def delete(self, key: KeyT) -> None: ...

    async def reset(self, values: tuple[ValueT, ...]) -> None: ...


class OverlayRefreshPolicy(str, Enum):
    STATIC = "static"
    ALWAYS = "always"
    REVISIONED = "revisioned"


@dataclass(frozen=True, slots=True)
class StorageLayer(Generic[KeyT, ValueT, InfoT]):
    reader: StorageReader[KeyT, ValueT, InfoT]
    refresh: OverlayRefreshPolicy = OverlayRefreshPolicy.STATIC
    revision: RevisionSource | None = None
    initializer: StorageInitializer | None = None

    def __post_init__(self) -> None:
        if self.refresh is OverlayRefreshPolicy.REVISIONED and self.revision is None:
            raise ValueError("a revisioned storage layer requires a revision source")
        if self.refresh is not OverlayRefreshPolicy.REVISIONED and self.revision is not None:
            raise ValueError("a revision source requires the revisioned refresh policy")


class MultiBackend(Generic[KeyT, ValueT, InfoT]):
    """Resolve a primary reader and ordered, read-only overlay layers."""

    def __init__(
        self,
        primary: StorageReader[KeyT, ValueT, InfoT],
        overlays: tuple[StorageLayer[KeyT, ValueT, InfoT], ...] = (),
        *,
        info_key: Callable[[InfoT], KeyT] | None = None,
    ) -> None:
        self.primary = primary
        self.overlays = overlays
        self.info_key = info_key or (
            lambda value: getattr(value, "path", value)
        )

    @property
    def readers(self) -> tuple[StorageReader[KeyT, ValueT, InfoT], ...]:
        return (self.primary, *(layer.reader for layer in self.overlays))

    @property
    def always_refresh(self) -> bool:
        return any(
            layer.refresh is OverlayRefreshPolicy.ALWAYS
            for layer in self.overlays
        )

    @property
    def overlay_revisions(self) -> tuple[RevisionSource, ...]:
        return tuple(
            layer.revision
            for layer in self.overlays
            if layer.revision is not None
        )

    async def get(self, key: KeyT) -> ValueT | None:
        for reader in self.readers:
            value = await reader.get(key)
            if value is not None:
                return value
        return None

    async def get_many(
        self,
        keys: tuple[KeyT, ...],
        *,
        concurrency: int = 8,
    ) -> dict[KeyT, ValueT]:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        remaining = dict.fromkeys(keys)
        values: dict[KeyT, ValueT] = {}
        for reader in self.readers:
            pending = tuple(remaining)
            if not pending:
                break
            if isinstance(reader, BatchStorageReader):
                loaded = await reader.get_many(pending)
            else:
                semaphore = asyncio.Semaphore(concurrency)

                async def load(key: KeyT) -> tuple[KeyT, ValueT | None]:
                    async with semaphore:
                        return key, await reader.get(key)

                loaded = {
                    key: value
                    for key, value in await asyncio.gather(
                        *(load(key) for key in pending)
                    )
                    if value is not None
                }
            for key, value in loaded.items():
                if key in remaining:
                    values[key] = value
                    remaining.pop(key)
        return values

    async def list_info(
        self,
        *,
        key: Callable[[InfoT], KeyT] | None = None,
    ) -> tuple[InfoT, ...]:
        get_key = key or self.info_key
        merged: dict[KeyT, InfoT] = {}
        for reader in self.readers:
            for value in await reader.list_info():
                merged.setdefault(get_key(value), value)
        return tuple(
            merged[item]
            for item in sorted(merged, key=lambda value: str(value))
        )


__all__ = [
    "MultiBackend",
    "BatchStorageReader",
    "OverlayRefreshPolicy",
    "StorageLayer",
    "StorageReader",
    "StorageWriter",
]
