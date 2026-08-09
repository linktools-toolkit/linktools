#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer-aware composition for Asset container and file-tree operations."""

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..errors import AIError, ErrorCode
from ..storage import ReadableStorageBackend, StorageComposition, StorageLayer
from ._domain import (
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryInfo,
    AssetEntryKey,
    AssetEntryRevision,
    AssetEntrySnapshot,
    AssetEntryVersion,
    AssetInfo,
    AssetKey,
    AssetRevision,
    AssetStoreRevision,
    AssetVersion,
)


@runtime_checkable
class AssetBackend(
    ReadableStorageBackend[AssetKey, dict[str, bytes], AssetInfo],
    Protocol,
):
    """Provide durable Asset tree operations for one storage layer."""

    @property
    def atomic_batch(self) -> bool: ...

    async def current_revision(self) -> AssetStoreRevision: ...
    async def stat_asset(self, key: AssetKey) -> "AssetInfo | None": ...
    async def list_assets(self) -> "tuple[AssetInfo, ...]": ...
    async def list_asset_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]": ...
    async def asset_revision_files(self, key: AssetKey, revision: AssetRevision) -> "tuple[AssetEntrySnapshot, ...]": ...
    async def current_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "tuple[AssetEntryInfo, bytes] | None": ...
    async def list_current_files(self, asset: AssetKey, *, prefix: "str | None", include_deleted: bool) -> "tuple[tuple[AssetEntryInfo, bytes], ...]": ...
    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]": ...
    async def get_file_at_revision(self, key: AssetEntryKey, revision: AssetEntryRevision) -> "bytes | None": ...
    async def snapshot_files(self, asset: AssetKey, revision: "AssetRevision | None", include_deleted: bool) -> "tuple[AssetEntrySnapshot, ...]": ...
    async def put_file(self, key: AssetEntryKey, value: bytes, *, primary_path: str, expected_entry_revision: "AssetEntryRevision | None", expected_revision: "AssetRevision | None") -> AssetEntryInfo: ...
    async def delete_file(self, key: AssetEntryKey, *, primary_path: str, expected_entry_revision: "AssetEntryRevision | None", expected_revision: "AssetRevision | None") -> AssetEntryDeleteResult: ...
    async def apply_file_batch(self, asset: AssetKey, changes: "Sequence[AssetEntryChange]", *, primary_path: str, expected_revision: "AssetRevision | None", expected_store_revision: "AssetStoreRevision | None") -> AssetEntryBatchResult: ...
    async def apply_asset_batch(self, changes: "Sequence[tuple[AssetKey, bytes | None, str, str, AssetRevision | None]]", *, expected_store_revision: "AssetStoreRevision | None") -> "tuple[AssetInfo | AssetDeleteResult, ...]": ...
    async def replace_tree(self, asset: AssetKey, files: "Mapping[str, bytes]", *, deleted_rel_paths: "Collection[str]", primary_path: str, expected_revision: "AssetRevision | None") -> AssetInfo: ...
    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]": ...
    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo: ...
    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo: ...
    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: Mapping[str, str]) -> AssetStoreRevision: ...


class AssetComposition(StorageComposition[AssetKey, dict[str, bytes], AssetInfo]):
    """Route Asset tree reads by layer ownership and mutations to the primary writer."""

    def __init__(
        self,
        primary: AssetBackend,
        *,
        layers: "Sequence[StorageLayer[AssetKey, dict[str, bytes], AssetInfo]]" = (),
    ) -> None:
        for layer in layers:
            if not isinstance(layer.backend, AssetBackend):
                raise TypeError("AssetComposition layers require Asset backends")
        super().__init__(primary, layers=layers)
        self._writer = primary
        self._backends = (primary, *(layer.backend for layer in layers))

    @property
    def atomic_batch(self) -> bool:
        return self._writer.atomic_batch

    async def current_store_revision(self) -> AssetStoreRevision:
        return AssetStoreRevision((await self.current_revision()).value)

    async def stat_asset(self, key: AssetKey) -> "AssetInfo | None":
        return await self.stat(key)

    async def list_assets(self) -> "tuple[AssetInfo, ...]":
        return await self.list_info()

    async def list_asset_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        _, versions = await self._asset_history(key)
        return versions

    async def asset_revision_files(self, key: AssetKey, revision: AssetRevision) -> "tuple[AssetEntrySnapshot, ...]":
        backend, versions = await self._asset_history(key)
        if backend is not None and any(version.revision == revision for version in versions):
            return await backend.asset_revision_files(key, revision)
        raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)

    async def current_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "tuple[AssetEntryInfo, bytes] | None":
        backend = await self._owner(key.asset)
        return None if backend is None else await backend.current_file(key, include_deleted=include_deleted)

    async def list_current_files(self, asset: AssetKey, *, prefix: "str | None", include_deleted: bool) -> "tuple[tuple[AssetEntryInfo, bytes], ...]":
        backend = await self._owner(asset)
        return () if backend is None else await backend.list_current_files(asset, prefix=prefix, include_deleted=include_deleted)

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        _, versions = await self._file_history(key)
        return versions

    async def get_file_at_revision(self, key: AssetEntryKey, revision: AssetEntryRevision) -> "bytes | None":
        backend, versions = await self._file_history(key)
        if backend is not None and any(version.entry_revision == revision for version in versions):
            return await backend.get_file_at_revision(key, revision)
        raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)

    async def snapshot_files(self, asset: AssetKey, revision: "AssetRevision | None", include_deleted: bool) -> "tuple[AssetEntrySnapshot, ...]":
        if revision is None:
            backend = await self._owner(asset)
            return () if backend is None else await backend.snapshot_files(asset, None, include_deleted)
        files = await self.asset_revision_files(asset, revision)
        return tuple(item for item in files if include_deleted or not item.deleted)

    async def put_file(self, key: AssetEntryKey, value: bytes, *, primary_path: str, expected_entry_revision: "AssetEntryRevision | None", expected_revision: "AssetRevision | None") -> AssetEntryInfo:
        result = await self._writer.put_file(key, value, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        self.invalidate()
        return result

    async def delete_file(self, key: AssetEntryKey, *, primary_path: str, expected_entry_revision: "AssetEntryRevision | None", expected_revision: "AssetRevision | None") -> AssetEntryDeleteResult:
        result = await self._writer.delete_file(key, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        self.invalidate()
        return result

    async def apply_file_batch(self, asset: AssetKey, changes: "Sequence[AssetEntryChange]", *, primary_path: str, expected_revision: "AssetRevision | None", expected_store_revision: "AssetStoreRevision | None") -> AssetEntryBatchResult:
        result = await self._writer.apply_file_batch(asset, changes, primary_path=primary_path, expected_revision=expected_revision, expected_store_revision=expected_store_revision)
        self.invalidate()
        return result

    async def apply_asset_batch(self, changes: "Sequence[tuple[AssetKey, bytes | None, str, str, AssetRevision | None]]", *, expected_store_revision: "AssetStoreRevision | None") -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        result = await self._writer.apply_asset_batch(changes, expected_store_revision=expected_store_revision)
        self.invalidate()
        return result

    async def replace_tree(self, asset: AssetKey, files: "Mapping[str, bytes]", *, deleted_rel_paths: "Collection[str]", primary_path: str, expected_revision: "AssetRevision | None") -> AssetInfo:
        result = await self._writer.replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=primary_path, expected_revision=expected_revision)
        self.invalidate()
        return result

    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]":
        result = await self._writer.delete_asset(key, expected_revision=expected_revision)
        self.invalidate()
        return result

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        result = await self._writer.restore_asset(key, revision, expected_revision=expected_revision)
        self.invalidate()
        return result

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        result = await self._writer.rename_asset(source, target, expected_source_revision=expected_source_revision)
        self.invalidate()
        return result

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: Mapping[str, str]) -> AssetStoreRevision:
        await self._writer.sync_sources(source_files, primary_paths)
        self.invalidate()
        return await self.current_store_revision()

    async def _owner(self, key: AssetKey) -> "AssetBackend | None":
        location = await self.locate(key)
        if location is None:
            return None
        if not isinstance(location.backend, AssetBackend):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset owner lacks tree capabilities")
        return location.backend

    async def _asset_history(self, key: AssetKey) -> "tuple[AssetBackend | None, tuple[AssetVersion, ...]]":
        owner = await self._owner(key)
        if owner is not None:
            return owner, await owner.list_asset_versions(key)
        for backend in self._backends:
            versions = await backend.list_asset_versions(key)
            if versions:
                return backend, versions
        return None, ()

    async def _file_history(self, key: AssetEntryKey) -> "tuple[AssetBackend | None, tuple[AssetEntryVersion, ...]]":
        owner = await self._owner(key.asset)
        if owner is not None:
            return owner, await owner.list_file_versions(key)
        for backend in self._backends:
            versions = await backend.list_file_versions(key)
            if versions:
                return backend, versions
        return None, ()


__all__ = ["AssetBackend", "AssetComposition"]
