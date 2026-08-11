#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct local-directory backend for Asset files."""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOperation,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    read_bytes,
    write_bytes_atomic,
)
from ._domain import AssetInfo, AssetKey, AssetRoot

_logger = environ.get_logger("ai.asset.directory")


@runtime_checkable
class AssetPathAdapter(Protocol):
    """Map Asset keys to files under a local directory."""

    def to_path(self, key: AssetKey) -> str: ...

    def from_path(self, path: str) -> "AssetKey | None": ...


class PrefixAssetPathAdapter:
    """Map Asset kinds to optional directory prefixes."""

    def __init__(self, prefixes: "Mapping[str, str] | None" = None) -> None:
        values = dict(prefixes or {})
        if (
            len(set(values.values())) != len(values)
            or any(not kind or not prefix or "/" in prefix or "\\" in prefix for kind, prefix in values.items())
        ):
            raise ValueError("asset path prefixes must be non-empty and unique")
        self._prefixes = values
        self._kinds = {prefix: kind for kind, prefix in values.items()}

    def to_path(self, key: AssetKey) -> str:
        return f"{self._prefixes.get(key.kind, key.kind)}/{key.id}"

    def from_path(self, path: str) -> "AssetKey | None":
        prefix, separator, identifier = path.partition("/")
        if not separator or not prefix or not identifier:
            return None
        return AssetKey(self._kinds.get(prefix, prefix), identifier)


class LocalDirectoryAssetBackend:
    """Expose existing local files without maintaining version history."""

    def __init__(
        self,
        root: "AssetRoot | str" = ".linktools/assets",
        *,
        writable: bool = False,
        path_adapter: "AssetPathAdapter | None" = None,
    ) -> None:
        resolved = local_directory_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("LocalDirectoryAssetBackend requires a filesystem root")
        self._root = resolved
        self._directory = Path(resolved.locator)
        self._writable = writable
        self._path_adapter = path_adapter or PrefixAssetPathAdapter()
        self._lock = asyncio.Lock()

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return self._writable

    @property
    def atomic_batch(self) -> bool:
        return False

    async def initialize(self) -> None:
        if self._writable:
            await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True)
        _logger.debug(
            "local directory asset backend initialized: root=%s writable=%s",
            self._directory,
            self._writable,
        )

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def head_revision(self) -> StorageRevision:
        async with self._lock:
            entries = await asyncio.to_thread(self._scan)
            return _store_revision(entries)

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        async with self._lock:
            entries = await asyncio.to_thread(self._scan)
            revision = _store_revision(entries)
            if after_revision == revision:
                return MetadataLoad(MetadataLoadMode.PATCH, revision, ())
            return MetadataLoad(
                MetadataLoadMode.REPLACE,
                revision,
                tuple(
                    MetadataChange(key, self._info(key, content, modified, revision))
                    for key, content, modified in entries
                ),
            )

    async def get(self, key: AssetKey) -> "bytes | None":
        async with self._lock:
            path = self._file_path(key)
            if not path.is_file():
                return None
            return await asyncio.to_thread(read_bytes, path)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "dict[AssetKey, bytes]":
        result: dict[AssetKey, bytes] = {}
        async with self._lock:
            for key in keys:
                path = self._file_path(key)
                if path.is_file():
                    result[key] = await asyncio.to_thread(read_bytes, path)
        return result

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        async with self._lock:
            path = self._file_path(key)
            if not path.is_file():
                return None
            content = await asyncio.to_thread(read_bytes, path)
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            revision = _store_revision(await asyncio.to_thread(self._scan))
            return self._info(key, content, modified, revision)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StoragePutResult[AssetInfo]":
        async with self._lock:
            self._require_writable()
            path = self._file_path(key)
            exists = path.is_file()
            current = await asyncio.to_thread(read_bytes, path) if exists else None
            current_revision = None if current is None else _entry_revision(current)
            self._check_revision(current_revision, expected_entry_revision)
            if current is not None and _etag(current) == _etag(value):
                entries = await asyncio.to_thread(self._scan)
                revision = _store_revision(entries)
                current = next(item for item in entries if item[0] == key)
                info = self._info(key, current[1], current[2], revision)
                return StoragePutResult(info, info.revision, revision, False)
            await asyncio.to_thread(write_bytes_atomic, path, bytes(value))
            entries = await asyncio.to_thread(self._scan)
            revision = _store_revision(entries)
            info = self._info(
                key,
                bytes(value),
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
                revision,
            )
            _logger.debug("local asset file stored: kind=%s id=%s", key.kind, key.id)
            return StoragePutResult(info, info.revision, revision, True)

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        async with self._lock:
            self._require_writable()
            path = self._file_path(key)
            exists = path.is_file()
            current = await asyncio.to_thread(read_bytes, path) if exists else None
            current_revision = None if current is None else _entry_revision(current)
            self._check_revision(current_revision, expected_entry_revision)
            if not exists:
                return StorageDeleteResult(key, False, None, _store_revision(await asyncio.to_thread(self._scan)))
            await asyncio.to_thread(path.unlink)
            self._remove_empty_parents(path.parent)
            revision = _store_revision(await asyncio.to_thread(self._scan))
            _logger.debug("local asset file deleted: kind=%s id=%s", key.kind, key.id)
            return StorageDeleteResult(key, True, current_revision, revision)

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        async with self._lock:
            self._require_writable()
            path = self._file_path(key)
            content = await asyncio.to_thread(read_bytes, path) if path.is_file() else None
            current_revision = None if content is None else _entry_revision(content)
            self._check_revision(current_revision, expected_entry_revision)
            if content is None:
                return StorageResetResult(key, False, _store_revision(await asyncio.to_thread(self._scan)))
            await asyncio.to_thread(path.unlink)
            self._remove_empty_parents(path.parent)
            _logger.debug("local asset file reset: kind=%s id=%s", key.kind, key.id)
            return StorageResetResult(key, True, _store_revision(await asyncio.to_thread(self._scan)))

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        current = await self.head_revision()
        if expected_revision is not None and expected_revision != current:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        results: list[StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]] = []
        for change in changes:
            if change.operation is StorageOperation.PUT:
                results.append(
                    await self.put(
                        change.key,
                        bytes(change.value or b""),
                        expected_entry_revision=change.expected_entry_revision,
                    )
                )
            elif change.operation is StorageOperation.DELETE:
                results.append(
                    await self.delete(
                        change.key,
                        expected_entry_revision=change.expected_entry_revision,
                    )
                )
            else:
                results.append(
                    await self.reset(
                        change.key,
                        expected_entry_revision=change.expected_entry_revision,
                    )
                )
        return StorageBatchResult(await self.head_revision(), False, tuple(results))

    def _scan(self) -> "tuple[tuple[AssetKey, bytes, datetime], ...]":
        if not self._directory.is_dir():
            return ()
        values: dict[AssetKey, tuple[bytes, datetime]] = {}
        for path in self._directory.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self._directory).as_posix()
            key = self._path_adapter.from_path(relative)
            if key is None:
                continue
            if key in values:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset path adapter produced duplicate keys")
            values[key] = (
                read_bytes(path),
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            )
        return tuple(
            (key, content, modified)
            for key, (content, modified) in sorted(
                values.items(),
                key=lambda item: (item[0].kind, item[0].id),
            )
        )

    def _file_path(self, key: AssetKey) -> Path:
        relative = self._path_adapter.to_path(key)
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or "\\" in relative
            or "\x00" in relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path adapter returned an invalid path")
        path = (self._directory / Path(*pure.parts)).resolve()
        try:
            path.relative_to(self._directory.resolve())
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path escapes local directory") from error
        return path

    def _info(
        self,
        key: AssetKey,
        content: bytes,
        modified_at: datetime,
        store_revision: StorageRevision,
    ) -> AssetInfo:
        return AssetInfo(
            key,
            _entry_revision(content),
            store_revision,
            _etag(content),
            len(content),
            StorageEntryStatus.NORMAL,
            self._root.root_id,
            self._root.digest,
            modified_at,
        )

    @staticmethod
    def _check_revision(current: "StorageEntryRevision | None", expected: "StorageEntryRevision | None") -> None:
        if expected is not None and current != expected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _require_writable(self) -> None:
        if not self._writable:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)

    def _remove_empty_parents(self, path: Path) -> None:
        root = self._directory.resolve()
        while path.resolve() != root:
            try:
                path.rmdir()
            except OSError:
                return
            path = path.parent


def local_directory_root(locator: str) -> AssetRoot:
    path = Path(locator).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


def _store_revision(entries: "Sequence[tuple[AssetKey, bytes, datetime]]") -> StorageRevision:
    return StorageRevision(
        canonical_sha256(
            [
                {
                    "kind": key.kind,
                    "id": key.id,
                    "etag": _etag(content),
                    "modified_at": modified.isoformat(),
                }
                for key, content, modified in entries
            ]
        )
    )


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _entry_revision(value: bytes) -> StorageEntryRevision:
    return StorageEntryRevision(int.from_bytes(hashlib.sha256(value).digest(), "big"))


__all__ = [
    "AssetPathAdapter",
    "LocalDirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "local_directory_root",
]
