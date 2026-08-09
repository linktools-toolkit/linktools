#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight local-directory backend for the current AssetStore tree."""

import asyncio
import hashlib
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import read_bytes, write_bytes_atomic
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
    AssetRoot,
    AssetStoreRevision,
    AssetVersion,
    validate_rel_path,
)

_logger = environ.get_logger("ai.asset.directory")


@runtime_checkable
class AssetPathAdapter(Protocol):
    def to_disk(self, asset_path: str) -> str:
        ...

    def from_disk(self, disk_path: str) -> str:
        ...


@dataclass(frozen=True, slots=True)
class PrefixAssetPathAdapter:
    """Map logical path prefixes to directory-specific prefixes."""

    mapping: "Mapping[str, str]"

    def to_disk(self, asset_path: str) -> str:
        prefix, separator, rest = asset_path.partition("/")
        return f"{self.mapping.get(prefix, prefix)}{separator}{rest}"

    def from_disk(self, disk_path: str) -> str:
        prefix, separator, rest = disk_path.partition("/")
        reverse = {value: key for key, value in self.mapping.items()}
        return f"{reverse.get(prefix, prefix)}{separator}{rest}"


@dataclass(frozen=True, slots=True)
class _IdentityAssetPathAdapter:
    def to_disk(self, asset_path: str) -> str:
        return asset_path

    def from_disk(self, disk_path: str) -> str:
        return disk_path


class LocalDirectoryAssetBackend:
    """Store the current AssetStore tree directly below a local directory."""

    def __init__(
        self,
        root: "AssetRoot | str" = ".linktools/assets",
        *,
        writable: bool = True,
        path_adapter: "AssetPathAdapter | None" = None,
    ) -> None:
        resolved = local_directory_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("LocalDirectoryAssetBackend requires a filesystem root")
        self._root = Path(resolved.locator)
        self._asset_root = resolved
        self._writable = writable
        self._path_adapter = path_adapter or _IdentityAssetPathAdapter()
        self._lock = asyncio.Lock()

    @property
    def root(self) -> AssetRoot:
        return self._asset_root

    @property
    def writable(self) -> bool:
        return self._writable

    async def initialize(self) -> None:
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        _logger.info("local asset directory initialized: root=%s", self._root)

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def current_revision(self) -> AssetStoreRevision:
        return await asyncio.to_thread(self._tree_revision)

    async def stat_asset(self, key: AssetKey) -> "AssetInfo | None":
        files = await asyncio.to_thread(self._scan_asset, key)
        return await self._asset_info(key, files) if files else None

    async def list_assets(self) -> "tuple[AssetInfo, ...]":
        assets = await asyncio.to_thread(self._scan_assets)
        result: "list[AssetInfo]" = []
        for key, files in assets:
            info = await self._asset_info(key, files)
            if info is not None:
                result.append(info)
        return tuple(result)

    async def list_asset_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        del key
        return ()

    async def asset_revision_files(self, key: AssetKey, revision: AssetRevision) -> "tuple[AssetEntrySnapshot, ...]":
        del key, revision
        raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND, "local directory backend has no asset history")

    async def current_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "tuple[AssetEntryInfo, bytes] | None":
        del include_deleted
        files = await asyncio.to_thread(self._scan_asset, key.asset)
        for item in files:
            if item[0] == key.rel_path:
                info = await self._asset_info(key.asset, files)
                return next(entry for entry in _info_files(info, files) if entry[0].key == key)
        return None

    async def list_current_files(self, asset: AssetKey, *, prefix: "str | None", include_deleted: bool) -> "tuple[tuple[AssetEntryInfo, bytes], ...]":
        del include_deleted
        files = await asyncio.to_thread(self._scan_asset, asset)
        if not files:
            return ()
        info = await self._asset_info(asset, files)
        return tuple(item for item in _info_files(info, files) if prefix is None or item[0].key.rel_path.startswith(prefix))

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        del key
        return ()

    async def get_file_at_revision(self, key: AssetEntryKey, revision: AssetEntryRevision) -> "bytes | None":
        del key, revision
        raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND, "local directory backend has no file history")

    async def snapshot_files(self, asset: AssetKey, revision: "AssetRevision | None", include_deleted: bool) -> "tuple[AssetEntrySnapshot, ...]":
        del include_deleted
        if revision is not None:
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND, "local directory backend has no asset history")
        files = await asyncio.to_thread(self._scan_asset, asset)
        info = await self._asset_info(asset, files)
        if info is None:
            return ()
        return tuple(_snapshot(entry, content) for entry, content in _info_files(info, files))

    async def put_file(
        self,
        key: AssetEntryKey,
        value: bytes,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryInfo:
        del primary_path
        self._require_writable()
        async with self._lock:
            files = await asyncio.to_thread(self._scan_asset, key.asset)
            self._check_expected(key.asset, expected_revision, files)
            self._check_entry_expected(key, expected_entry_revision, files)
            await asyncio.to_thread(write_bytes_atomic, self._file_path(key), bytes(value))
            result = await self.current_file(key)
            if result is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info("local asset file stored: asset=%s/%s path=%s", key.asset.kind, key.asset.id, key.rel_path)
            return result[0]

    async def delete_file(
        self,
        key: AssetEntryKey,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryDeleteResult:
        if key.rel_path == primary_path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "primary asset file cannot be deleted")
        self._require_writable()
        async with self._lock:
            files = await asyncio.to_thread(self._scan_asset, key.asset)
            self._check_expected(key.asset, expected_revision, files)
            self._check_entry_expected(key, expected_entry_revision, files)
            existing = next((item for item in files if item[0] == key.rel_path), None)
            asset = await self._asset_info(key.asset, files)
            if existing is not None:
                await asyncio.to_thread(self._unlink, self._file_path(key))
                _logger.info("local asset file deleted: asset=%s/%s path=%s", key.asset.kind, key.asset.id, key.rel_path)
            token = await asyncio.to_thread(self._tree_revision)
            return AssetEntryDeleteResult(key, existing is not None, AssetEntryRevision(1) if existing is not None else None, AssetRevision(1) if asset is not None else AssetRevision(1), token)

    async def apply_file_batch(
        self,
        asset: AssetKey,
        changes: "Sequence[AssetEntryChange]",
        *,
        primary_path: str,
        expected_revision: "AssetRevision | None",
        expected_store_revision: "AssetStoreRevision | None",
    ) -> AssetEntryBatchResult:
        self._require_writable()
        async with self._lock:
            files = await asyncio.to_thread(self._scan_asset, asset)
            self._check_expected(asset, expected_revision, files)
            if expected_store_revision is not None and expected_store_revision != AssetStoreRevision(self._tree_revision_sync()):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if len({change.rel_path for change in changes}) != len(changes):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "duplicate asset file path")
            before = {path: content for path, content, _modified in files}
            for change in changes:
                self._check_entry_expected(AssetEntryKey(asset, change.rel_path), change.expected_entry_revision, files)
                if change.operation == "DELETE" and change.rel_path == primary_path:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "primary asset file cannot be deleted")
                if change.operation not in {"PUT", "DELETE"}:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if change.operation == "PUT" and change.value is None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            for change in changes:
                target = self._file_path(AssetEntryKey(asset, change.rel_path))
                if change.operation == "PUT":
                    await asyncio.to_thread(write_bytes_atomic, target, bytes(change.value or b""))
                elif change.rel_path in before:
                    await asyncio.to_thread(self._unlink, target)
            final_files = await asyncio.to_thread(self._scan_asset, asset)
            info = await self._asset_info(asset, final_files)
            token = await asyncio.to_thread(self._tree_revision)
            results: "list[AssetEntryInfo | AssetEntryDeleteResult]" = []
            for change in changes:
                key = AssetEntryKey(asset, change.rel_path)
                if change.operation == "PUT":
                    results.append(next(entry for entry, _content in _info_files(info, final_files) if entry.key == key))
                else:
                    results.append(AssetEntryDeleteResult(key, change.rel_path in before, AssetEntryRevision(1) if change.rel_path in before else None, AssetRevision(1), token))
            return AssetEntryBatchResult(asset, AssetRevision(1), token, True, tuple(results))

    async def apply_asset_batch(
        self,
        changes: "Sequence[tuple[AssetKey, bytes | None, str, Literal['PUT', 'DELETE'], AssetRevision | None]]",
        *,
        expected_store_revision: "AssetStoreRevision | None",
    ) -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        self._require_writable()
        async with self._lock:
            current_token = AssetStoreRevision(self._tree_revision_sync())
            if expected_store_revision is not None and expected_store_revision != current_token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if len({item[0] for item in changes}) != len(changes):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "duplicate asset key")
            existing = {asset: await asyncio.to_thread(self._scan_asset, asset) for asset, _value, _path, _operation, _expected in changes}
            for asset, value, primary_path, operation, expected in changes:
                self._check_expected(asset, expected, existing[asset])
                if operation == "PUT":
                    if value is None:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    await asyncio.to_thread(write_bytes_atomic, self._file_path(AssetEntryKey(asset, primary_path)), bytes(value))
                elif operation == "DELETE":
                    await asyncio.to_thread(self._remove_asset, asset)
                else:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            token = AssetStoreRevision(self._tree_revision_sync())
            results: "list[AssetInfo | AssetDeleteResult]" = []
            for asset, _value, _path, operation, _expected in changes:
                files = await asyncio.to_thread(self._scan_asset, asset)
                info = await self._asset_info(asset, files)
                if operation == "DELETE":
                    results.append(AssetDeleteResult(asset, bool(existing[asset]), AssetRevision(1) if existing[asset] else None, token))
                elif info is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                else:
                    results.append(info)
            return tuple(results)

    async def replace_tree(
        self,
        asset: AssetKey,
        files: "Mapping[str, bytes]",
        *,
        deleted_rel_paths: "Collection[str]",
        primary_path: str,
        expected_revision: "AssetRevision | None",
    ) -> AssetInfo:
        del deleted_rel_paths
        self._require_writable()
        if primary_path not in files:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            current = await asyncio.to_thread(self._scan_asset, asset)
            self._check_expected(asset, expected_revision, current)
            await asyncio.to_thread(self._remove_asset, asset)
            for path, value in files.items():
                await asyncio.to_thread(write_bytes_atomic, self._file_path(AssetEntryKey(asset, path)), bytes(value))
            result = await self.stat_asset(asset)
            if result is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return result

    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]":
        self._require_writable()
        async with self._lock:
            files = await asyncio.to_thread(self._scan_asset, key)
            self._check_expected(key, expected_revision, files)
            info = await self._asset_info(key, files)
            if info is None:
                return False, None
            await asyncio.to_thread(self._remove_asset, key)
            _logger.info("local asset directory removed: asset=%s/%s", key.kind, key.id)
            return True, info

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        del key, revision, expected_revision
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "local directory backend has no asset history")

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        del source, target, expected_source_revision
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "local directory backend does not support rename")

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: "Mapping[str, str]") -> AssetStoreRevision:
        del source_files, primary_paths
        return await self.current_revision()

    def _asset_dir(self, key: AssetKey) -> Path:
        _validate_component(key.kind)
        _validate_component(key.id)
        return self._disk_path(f"{key.kind}/{key.id}")

    def _file_path(self, key: AssetEntryKey) -> Path:
        target = self._disk_path(f"{key.asset.kind}/{key.asset.id}/{key.rel_path}")
        return target

    def _disk_path(self, logical_path: str) -> Path:
        validate_rel_path(logical_path)
        disk_path = self._path_adapter.to_disk(logical_path)
        if not isinstance(disk_path, str) or not disk_path or "\x00" in disk_path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path adapter returned an invalid path")
        target = (self._root / disk_path).resolve()
        root = self._root.resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path escapes local directory") from error
        return target

    def _logical_file(self, path: Path) -> tuple[AssetKey, str]:
        relative = path.relative_to(self._root.resolve()).as_posix()
        logical = self._path_adapter.from_disk(relative)
        if not isinstance(logical, str) or not logical:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset path adapter returned an invalid logical path")
        validate_rel_path(logical)
        parts = logical.split("/")
        if len(parts) < 3:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset path adapter returned an incomplete path")
        return AssetKey(parts[0], parts[1]), "/".join(parts[2:])

    def _scan_asset(self, key: AssetKey) -> "tuple[tuple[str, bytes, datetime], ...]":
        directory = self._asset_dir(key)
        if not directory.is_dir():
            return ()
        root = self._root.resolve()
        result: "list[tuple[str, bytes, datetime]]" = []
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "local asset file escapes root") from error
            asset, relative = self._logical_file(path)
            if asset != key:
                continue
            result.append((relative, read_bytes(path), datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)))
        return tuple(sorted(result, key=lambda item: item[0]))

    def _scan_assets(self) -> "tuple[tuple[AssetKey, tuple[tuple[str, bytes, datetime], ...]], ...]":
        if not self._root.is_dir():
            return ()
        keys: set[AssetKey] = set()
        for path in self._root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                key, _relative = self._logical_file(path)
                keys.add(key)
        return tuple((key, files) for key in sorted(keys, key=lambda value: (value.kind, value.id)) if (files := self._scan_asset(key)))

    async def _asset_info(self, key: AssetKey, files: tuple[tuple[str, bytes, datetime], ...]) -> "AssetInfo | None":
        if not files:
            return None
        token = await asyncio.to_thread(self._tree_revision)
        entries = tuple(_entry_info(key, path, content, modified, token) for path, content, modified in files)
        etag = canonical_sha256([{"path": entry.key.rel_path, "etag": entry.etag, "deleted": False} for entry in entries])
        modified = max(entry.modified_at for entry in entries)
        return AssetInfo(key, AssetRevision(1), token, etag, sum(entry.size for entry in entries), len(entries), False, None, etag, ("directory",), modified)

    def _tree_revision(self) -> AssetStoreRevision:
        values = []
        for key, files in self._scan_assets():
            values.extend({"kind": key.kind, "id": key.id, "path": path, "etag": hashlib.sha256(content).hexdigest()} for path, content, _modified in files)
        return AssetStoreRevision(str(int(canonical_sha256(values), 16)))

    def _tree_revision_sync(self) -> str:
        return self._tree_revision().value

    def _check_expected(self, key: AssetKey, expected: "AssetRevision | None", files: tuple[tuple[str, bytes, datetime], ...]) -> None:
        if expected is not None and (not files or expected.value != 1):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _check_entry_expected(self, key: AssetEntryKey, expected: "AssetEntryRevision | None", files: tuple[tuple[str, bytes, datetime], ...]) -> None:
        if expected is not None and (not any(path == key.rel_path for path, _content, _modified in files) or expected.value != 1):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _require_writable(self) -> None:
        if not self._writable:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)

    def _remove_asset(self, key: AssetKey) -> None:
        shutil.rmtree(self._asset_dir(key))

    @staticmethod
    def _unlink(path: Path) -> None:
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != parent.parent:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def local_directory_root(locator: str) -> AssetRoot:
    path = Path(locator).resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


def _info_files(info: AssetInfo, files: "tuple[tuple[str, bytes, datetime], ...]") -> "tuple[tuple[AssetEntryInfo, bytes], ...]":
    return tuple((_entry_info(info.key, path, content, modified, info.store_revision), content) for path, content, modified in files)


def _entry_info(key: AssetKey, path: str, content: bytes, modified: datetime, store_revision: AssetStoreRevision) -> AssetEntryInfo:
    entry_key = AssetEntryKey(key, path)
    return AssetEntryInfo(entry_key, entry_key.file_id, AssetEntryRevision(1), AssetRevision(1), store_revision, hashlib.sha256(content).hexdigest(), len(content), False, "OVERRIDE", None, "directory", True, modified)


def _snapshot(info: AssetEntryInfo, content: bytes) -> AssetEntrySnapshot:
    return AssetEntrySnapshot(info.key, info.entry_revision, info.etag, False, info.origin, info.source_digest, content)


def _validate_component(value: str) -> None:
    try:
        validate_rel_path(value)
    except AIError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset directory component is invalid") from error
    if "/" in value:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset directory component is invalid")


__all__ = ["AssetPathAdapter", "LocalDirectoryAssetBackend", "PrefixAssetPathAdapter", "local_directory_root"]
