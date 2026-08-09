#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crash-recoverable filesystem AssetStore backend."""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

from linktools.core import environ

from ..storage import read_json, write_json_atomic
from ._backend import InMemoryAssetBackend
from ._domain import (
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryInfo,
    AssetEntryKey,
    AssetEntryRevision,
    AssetInfo,
    AssetKey,
    AssetRevision,
    AssetRoot,
    AssetStoreRevision,
)

_logger = environ.get_logger("ai.asset.filesystem")


class FilesystemAssetBackend(InMemoryAssetBackend):
    """Filesystem backend persisting the same tree ledger as the memory backend."""

    def __init__(self, root: "AssetRoot | str", *, writable: bool = True) -> None:
        resolved = filesystem_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("FilesystemAssetBackend requires a filesystem root")
        super().__init__(resolved, writable=writable)
        self._directory = Path(resolved.locator)
        self._state_path = self._directory / ".asset-tree.json"

    async def initialize(self) -> None:
        async with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            if self._state_path.exists():
                self._load_state(read_json(self._state_path))
        _logger.info("filesystem asset tree initialized: root=%s revision=%s", self._directory, self._store_revision)

    def _persist(self) -> None:
        write_json_atomic(self._state_path, self._dump_state(), fsync=True)

    async def put_file(
        self,
        key: AssetEntryKey,
        value: bytes,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryInfo:
        result = await super().put_file(key, value, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        self._persist()
        return result

    async def delete_file(
        self,
        key: AssetEntryKey,
        *,
        primary_path: str,
        expected_entry_revision: AssetEntryRevision | None,
        expected_revision: AssetRevision | None,
    ) -> AssetEntryDeleteResult:
        result = await super().delete_file(key, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        self._persist()
        return result

    async def apply_file_batch(
        self,
        asset: AssetKey,
        changes: "Sequence[AssetEntryChange]",
        *,
        primary_path: str,
        expected_revision: "AssetRevision | None",
        expected_store_revision: "AssetStoreRevision | None",
    ) -> AssetEntryBatchResult:
        result = await super().apply_file_batch(asset, changes, primary_path=primary_path, expected_revision=expected_revision, expected_store_revision=expected_store_revision)
        self._persist()
        return result

    async def apply_asset_batch(
        self,
        changes: "Sequence[tuple[AssetKey, bytes | None, str, Literal['PUT', 'DELETE'], AssetRevision | None]]",
        *,
        expected_store_revision: AssetStoreRevision | None,
    ) -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        result = await super().apply_asset_batch(changes, expected_store_revision=expected_store_revision)
        self._persist()
        return result

    async def replace_tree(
        self,
        asset: AssetKey,
        files: "Mapping[str, bytes]",
        *,
        deleted_rel_paths: "Collection[str]",
        primary_path: str,
        expected_revision: "AssetRevision | None",
    ) -> AssetInfo:
        result = await super().replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=primary_path, expected_revision=expected_revision)
        self._persist()
        return result

    async def delete_asset(self, key: AssetKey, *, expected_revision: AssetRevision | None) -> "tuple[bool, AssetInfo | None]":
        result = await super().delete_asset(key, expected_revision=expected_revision)
        self._persist()
        return result

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        result = await super().restore_asset(key, revision, expected_revision=expected_revision)
        self._persist()
        return result

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        result = await super().rename_asset(source, target, expected_source_revision=expected_source_revision)
        self._persist()
        return result

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: "Mapping[str, str]") -> AssetStoreRevision:
        result = await super().sync_sources(source_files, primary_paths)
        self._persist()
        return result


def filesystem_root(locator: str) -> AssetRoot:
    path = Path(locator).resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


__all__ = ["FilesystemAssetBackend", "filesystem_root"]
