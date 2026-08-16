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
    StorageEntryRevision,
    StorageEntryStatus,
    StorageRevision,
    read_bytes,
)
from ._domain import AssetInfo, AssetKey, AssetRoot

_logger = environ.get_logger("ai.asset.directory")


@runtime_checkable
class AssetPathAdapter(Protocol):
    """Map Asset keys to files under a local directory."""

    def validate(self, kinds: Sequence[str]) -> None: ...

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
        return f"{self._prefixes.get(key.kind, key.kind)}/{key.id}"

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
    ) -> None:
        resolved = directory_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("DirectoryAssetBackend requires a filesystem root")
        self._root = resolved
        self._directory = Path(resolved.locator)
        self._path_adapter = path_adapter or PrefixAssetPathAdapter()
        self._lock = asyncio.Lock()

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


def directory_root(locator: str) -> AssetRoot:
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
    "DirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "directory_root",
]
