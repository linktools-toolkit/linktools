#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct local-directory backend for Asset files."""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    StorageEntryRevision,
    StorageEntryStatus,
    StorageRevision,
    read_bytes,
)
from ._domain import AssetInfo, AssetKey, AssetRoot

_logger = environ.get_logger("ai.asset.directory")


@dataclass(frozen=True, slots=True)
class _DirectoryEntry:
    key: AssetKey
    digest: str
    size: int
    modified_at: datetime


@dataclass(frozen=True, slots=True)
class _DirectorySignature:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@runtime_checkable
class AssetPathAdapter(Protocol):
    """Map Asset keys to files under a local directory."""

    def validate(self, kinds: Sequence[str]) -> None: ...

    def root_path(self, kind: str) -> str: ...

    def to_path(self, key: AssetKey) -> str: ...

    def from_path(self, path: str) -> "AssetKey | None": ...


class PrefixAssetPathAdapter:
    """Map Asset kinds to optional directory prefixes."""

    def __init__(self, prefixes: "Mapping[str, str] | None" = None) -> None:
        values = dict(prefixes or {})
        if any(not isinstance(kind, str) or not kind for kind in values):
            raise ValueError("asset path kinds must be non-empty strings")
        for prefix in values.values():
            _validate_prefix(prefix)
        self._prefixes = values
        self._validate_effective_prefixes(tuple(values))

    def validate(self, kinds: Sequence[str]) -> None:
        values = tuple(kinds)
        if len(set(values)) != len(values) or any(not isinstance(kind, str) or not kind for kind in values):
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        unknown = set(self._prefixes).difference(values)
        if unknown:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        self._validate_effective_prefixes(values)
        for kind in values:
            probe = AssetKey(kind, "linktools-probe")
            try:
                if self.from_path(self.to_path(probe)) != probe:
                    raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
            except (AIError, ValueError) as error:
                if isinstance(error, AIError):
                    raise
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT) from error

    def to_path(self, key: AssetKey) -> str:
        return f"{self.root_path(key.kind)}/{key.id}"

    def root_path(self, kind: str) -> str:
        return self._prefixes.get(kind, kind)

    def from_path(self, path: str) -> "AssetKey | None":
        if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
            return None
        parts = path.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            return None
        for prefix, kind in sorted(
            ((prefix, kind) for kind, prefix in self._prefixes.items()),
            key=lambda item: len(item[0].split("/")),
            reverse=True,
        ):
            prefix_parts = prefix.split("/")
            if parts[: len(prefix_parts)] == prefix_parts and len(parts) > len(prefix_parts):
                return _asset_key(kind, "/".join(parts[len(prefix_parts) :]))
        return _asset_key(parts[0], "/".join(parts[1:])) if len(parts) > 1 else None

    def _validate_effective_prefixes(self, kinds: Sequence[str]) -> None:
        effective = {kind: self._prefixes.get(kind, kind) for kind in kinds}
        prefixes = tuple(effective.values())
        if len(set(prefixes)) != len(prefixes):
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        segments = {kind: prefix.split("/") for kind, prefix in effective.items()}
        for prefix in segments.values():
            if any(prefix != other and _is_parent_path(prefix, other) for other in segments.values()):
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)


def _validate_prefix(prefix: str) -> None:
    if not isinstance(prefix, str) or not prefix or prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError("asset path prefix is invalid")
    if "\\" in prefix or "\x00" in prefix:
        raise ValueError("asset path prefix is invalid")
    parts = prefix.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("asset path prefix is invalid")


def _is_parent_path(parent: Sequence[str], child: Sequence[str]) -> bool:
    return len(parent) < len(child) and tuple(child[: len(parent)]) == tuple(parent)


def _asset_key(kind: str, identifier: str) -> "AssetKey | None":
    try:
        return AssetKey(kind, identifier)
    except ValueError:
        return None


class DirectoryAssetBackend:
    """Expose existing local files without maintaining version history."""

    def __init__(
        self,
        root: "AssetRoot | str" = "assets",
        *,
        path_adapter: "AssetPathAdapter | None" = None,
        kinds: "Sequence[str] | None" = None,
    ) -> None:
        resolved = directory_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("DirectoryAssetBackend requires a filesystem root")
        self._root = resolved
        self._directory = Path(resolved.locator)
        self._path_adapter = path_adapter or PrefixAssetPathAdapter()
        self._kinds = None if kinds is None else frozenset(kinds)
        if kinds is not None:
            self._path_adapter.validate(tuple(kinds))
            for kind in self._kinds:
                _safe_asset_parts(self._path_adapter.root_path(kind))
        self._lock = asyncio.Lock()
        self._entries: dict[AssetKey, tuple[_DirectorySignature, _DirectoryEntry]] = {}
        self._revision = _store_revision(())

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return False

    @property
    def atomic_batch(self) -> bool:
        return False

    async def initialize(self) -> None:
        _logger.debug(
            "local directory asset backend initialized: root=%s writable=%s",
            self._directory,
            False,
        )

    async def close(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._revision = _store_revision(())
        _logger.debug("local directory asset backend closed: root=%s", self._directory)

    async def head_revision(self) -> StorageRevision:
        async with self._lock:
            entries = await asyncio.to_thread(self._scan)
            self._revision = _store_revision(entries)
            return self._revision

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        async with self._lock:
            entries = await asyncio.to_thread(self._scan)
            revision = _store_revision(entries)
            self._revision = revision
            if after_revision == revision:
                return MetadataLoad(MetadataLoadMode.PATCH, revision, ())
            return MetadataLoad(
                MetadataLoadMode.REPLACE,
                revision,
                tuple(
                    MetadataChange(entry.key, self._info(entry, revision))
                    for entry in entries
                ),
            )

    async def get(self, key: AssetKey) -> "bytes | None":
        async with self._lock:
            self._validate_key(key)
            path = await asyncio.to_thread(self._file_path, key)
        return await asyncio.to_thread(_read_optional, path)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "dict[AssetKey, bytes]":
        async with self._lock:
            paths = await asyncio.to_thread(
                lambda: {
                    key: self._file_path(self._validate_key(key))
                    for key in dict.fromkeys(keys)
                }
            )
        return await asyncio.to_thread(_read_many_sync, paths)

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        async with self._lock:
            self._validate_key(key)
            path = await asyncio.to_thread(self._file_path, key)
        result = await asyncio.to_thread(_stat_file, path)
        if result is None:
            return None
        signature, modified = result
        async with self._lock:
            entry = self._cached_entry(key, path, signature, modified)
            return self._info(entry, self._revision)

    def _scan(self) -> "tuple[_DirectoryEntry, ...]":
        if not self._directory.is_dir():
            self._entries.clear()
            return ()
        values: dict[AssetKey, _DirectoryEntry] = {}
        roots = (
            (self._path_adapter.root_path(kind), kind)
            for kind in sorted(self._kinds or ())
        ) if self._kinds is not None else (("", None),)
        for relative_root, expected_kind in roots:
            scan_root = self._directory / Path(*PurePosixPath(relative_root).parts)
            if not scan_root.is_dir():
                continue
            for path in scan_root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(self._directory).as_posix()
                key = self._path_adapter.from_path(relative)
                if key is None or self._kinds is not None and key.kind not in self._kinds:
                    continue
                if expected_kind is not None and key.kind != expected_kind:
                    continue
                if key in values:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset path adapter produced duplicate keys")
                stat = path.stat()
                signature = _directory_signature(stat)
                cached = self._entries.get(key)
                if cached is not None and cached[0] == signature:
                    entry = cached[1]
                else:
                    content = read_bytes(path)
                    entry = _DirectoryEntry(
                        key,
                        _etag(content),
                        len(content),
                        datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                    )
                values[key] = entry
                self._entries[key] = (signature, entry)
        self._entries = {
            key: value for key, value in self._entries.items() if key in values
        }
        return tuple(
            entry
            for key, entry in sorted(
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
        root_parts = _safe_asset_parts(self._path_adapter.root_path(key.kind))
        if pure.parts[: len(root_parts)] != root_parts or len(pure.parts) <= len(root_parts):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path is outside its kind root")
        path = (self._directory / Path(*pure.parts)).resolve()
        try:
            path.relative_to(self._directory.resolve())
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset path escapes local directory") from error
        return path

    def _validate_key(self, key: AssetKey) -> AssetKey:
        if self._kinds is not None and key.kind not in self._kinds:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset kind is not registered")
        return key

    def _info(self, entry: _DirectoryEntry, store_revision: StorageRevision) -> AssetInfo:
        return AssetInfo(
            entry.key,
            _entry_revision(entry.digest),
            store_revision,
            entry.digest,
            entry.size,
            StorageEntryStatus.NORMAL,
            self._root.root_id,
            self._root.digest,
            entry.modified_at,
        )

    def _cached_entry(
        self,
        key: AssetKey,
        path: Path,
        signature: _DirectorySignature,
        modified_at: datetime,
    ) -> _DirectoryEntry:
        cached = self._entries.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        content = read_bytes(path)
        entry = _DirectoryEntry(key, _etag(content), len(content), modified_at)
        self._entries[key] = (signature, entry)
        return entry


def directory_root(locator: str) -> AssetRoot:
    path = Path(locator).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


def _store_revision(entries: "Sequence[_DirectoryEntry]") -> StorageRevision:
    return StorageRevision(
        canonical_sha256(
            [
                {
                    "kind": entry.key.kind,
                    "id": entry.key.id,
                    "etag": entry.digest,
                    "modified_at": entry.modified_at.isoformat(),
                }
                for entry in entries
            ]
        )
    )


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _entry_revision(value: str) -> StorageEntryRevision:
    return StorageEntryRevision(int(value, 16))


def _safe_asset_parts(value: str) -> tuple[str, ...]:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
    return pure.parts


def _read_many_sync(paths: Mapping[AssetKey, Path]) -> dict[AssetKey, bytes]:
    return {key: read_bytes(path) for key, path in paths.items() if path.is_file()}


def _read_optional(path: Path) -> bytes | None:
    return read_bytes(path) if path.is_file() else None


def _stat_file(path: Path) -> tuple[_DirectorySignature, datetime] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return _directory_signature(stat), datetime.fromtimestamp(stat.st_mtime, timezone.utc)


def _directory_signature(stat: object) -> _DirectorySignature:
    return _DirectorySignature(
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


__all__ = [
    "AssetPathAdapter",
    "DirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "directory_root",
]
