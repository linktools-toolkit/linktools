#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent filesystem backend for Asset files."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TypeVar

from linktools.core import environ

from ..storage import (
    FilesystemMutationLock,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    read_json,
    write_json_atomic,
)
from ._backend import InMemoryAssetBackend
from ._domain import AssetInfo, AssetKey, AssetRoot

_logger = environ.get_logger("ai.asset.filesystem")
_ResultT = TypeVar("_ResultT")


class FilesystemAssetBackend(InMemoryAssetBackend):
    """Persist the Asset file ledger as one atomic filesystem state file."""

    def __init__(self, root: "AssetRoot | str", *, writable: bool = True) -> None:
        resolved = filesystem_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("FilesystemAssetBackend requires a filesystem root")
        super().__init__(resolved, writable=writable)
        self._directory = Path(resolved.locator)
        self._state_path = self._directory / ".asset-state.json"
        self._persistence_lock = asyncio.Lock()
        self._mutation_lock = FilesystemMutationLock(self._directory / ".asset-state.lock")

    async def initialize(self) -> None:
        async with self._persistence_lock, self._mutation_lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            if self._state_path.exists():
                self.import_state(await asyncio.to_thread(read_json, self._state_path))
            else:
                self.import_state(self._empty_state())
        _logger.debug(
            "filesystem asset backend initialized: root=%s revision=%s",
            self._directory,
            self._revision,
        )

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StoragePutResult[AssetInfo]":
        return await self._mutate(
            lambda: super(FilesystemAssetBackend, self).put(
                key,
                value,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        return await self._mutate(
            lambda: super(FilesystemAssetBackend, self).delete(
                key,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        return await self._mutate(
            lambda: super(FilesystemAssetBackend, self).reset(
                key,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        return await self._mutate(
            lambda: super(FilesystemAssetBackend, self).apply_batch(
                changes,
                expected_revision=expected_revision,
            )
        )

    async def _mutate(self, operation: "Callable[[], Awaitable[_ResultT]]") -> _ResultT:
        async with self._persistence_lock, self._mutation_lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            if self._state_path.exists():
                self.import_state(await asyncio.to_thread(read_json, self._state_path))
            else:
                self.import_state(self._empty_state())
            previous = self.export_state()
            try:
                result = await operation()
                current = self.export_state()
                if current != previous:
                    await asyncio.to_thread(write_json_atomic, self._state_path, current, fsync=True)
                return result
            except BaseException:
                self.import_state(previous)
                raise

    def _empty_state(self) -> "dict[str, object]":
        return InMemoryAssetBackend(self.root, writable=self.writable).export_state()


def filesystem_root(locator: str) -> AssetRoot:
    path = Path(locator).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


__all__ = ["FilesystemAssetBackend", "filesystem_root"]
