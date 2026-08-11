#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw file AssetStore backed by StorageOverlay."""

import base64
import binascii
import hashlib
import json
from collections.abc import Sequence

from linktools.core import environ

from ..core import Page
from ..errors import AIError, ErrorCode
from ..storage import (
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOverlay,
    StorageOwnedInfo,
    StorageResetResult,
    StorageRevision,
    VersionSummary,
)
from ._domain import AssetInfo, AssetKey

_logger = environ.get_logger("ai.asset.store")


class AssetCacheAdapter:
    """Cache raw Asset file bytes using immutable metadata."""

    def cache_key(self, key: AssetKey, info: AssetInfo) -> str:
        return ":".join(
            ("asset", info.root_digest, key.kind, key.id, str(info.revision.value), info.etag)
        )

    def to_cache(self, value: bytes) -> bytes:
        return hashlib.sha256(value).digest() + value

    def from_cache(self, value: bytes) -> bytes:
        if len(value) < 32 or value[:32] != hashlib.sha256(value[32:]).digest():
            raise ValueError("asset cache payload is invalid")
        return value[32:]


class AssetStore:
    """Expose Asset files without interpreting their contents."""

    def __init__(
        self,
        storage: "StorageOverlay[AssetKey, bytes, AssetInfo]",
    ) -> None:
        """Create a raw Asset file store from one storage overlay."""
        self._storage = storage
        self._ready = False

    @property
    def ready(self) -> bool:
        """Return whether storage initialization completed successfully."""
        return self._ready

    async def initialize(self) -> None:
        """Initialize configured storage backends before serving requests."""
        if self._ready:
            return
        await self._storage.initialize()
        self._ready = True
        _logger.info("asset store initialized")

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        """Return current effective file metadata and status."""
        self._ensure_ready()
        return await self._storage.stat(key)

    async def get(self, key: AssetKey) -> "bytes | None":
        """Return current effective file bytes, or None when no file is visible."""
        self._ensure_ready()
        return await self._storage.get(key)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "tuple[bytes | None, ...]":
        """Return current file bytes in the same order as the requested keys."""
        self._ensure_ready()
        return await self._storage.get_many(keys)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
    ) -> AssetInfo:
        """Store one file with an optional current-revision check."""
        self._ensure_ready()
        result = await self._storage.put(
            key,
            bytes(value),
            expected_entry_revision=expected_revision,
        )
        _logger.info(
            "asset file put: kind=%s id=%s revision=%s changed=%s",
            key.kind,
            key.id,
            result.entry_revision,
            result.changed,
        )
        return result.info

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        """Delete one file with an optional current-revision check."""
        self._ensure_ready()
        result = await self._storage.delete(key, expected_entry_revision=expected_revision)
        _logger.info("asset file delete: kind=%s id=%s deleted=%s", key.kind, key.id, result.deleted)
        return result

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        """Reset one file so a lower read layer becomes effective."""
        self._ensure_ready()
        result = await self._storage.reset(key, expected_entry_revision=expected_revision)
        _logger.info("asset file reset: kind=%s id=%s reset=%s", key.kind, key.id, result.reset)
        return result

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        """Apply file changes using the writer backend's batch guarantees."""
        self._ensure_ready()
        return await self._storage.apply_batch(
            changes,
            expected_revision=expected_revision,
        )

    async def list_info(
        self,
        *,
        kind: "str | None" = None,
        prefix: "str | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[AssetInfo]":
        """Page active file metadata by kind and key prefix."""
        self._ensure_ready()
        _validate_limit(limit)
        values = [
            info
            for info in await self._storage.list_info()
            if info.status is StorageEntryStatus.NORMAL
            and (kind is None or info.key.kind == kind)
            and (prefix is None or info.key.id.startswith(prefix))
        ]
        ordered = tuple(sorted(values, key=lambda info: (info.key.kind, info.key.id)))
        revision = await self._storage.current_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered)
        selected = ordered[start : start + limit]
        next_key = selected[-1].key if selected and start + len(selected) < len(ordered) else None
        return Page(selected, _make_cursor(revision, kind, prefix, next_key))

    async def list_info_with_owners(
        self,
        *,
        kind: "str | None" = None,
        prefix: "str | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[StorageOwnedInfo[AssetInfo]]":
        """Page active file metadata together with its effective storage owner."""
        self._ensure_ready()
        _validate_limit(limit)
        values = [
            owned
            for owned in await self._storage.list_info_with_owners()
            if owned.info.status is StorageEntryStatus.NORMAL
            and (kind is None or owned.info.key.kind == kind)
            and (prefix is None or owned.info.key.id.startswith(prefix))
        ]
        ordered = tuple(sorted(values, key=lambda owned: (owned.info.key.kind, owned.info.key.id)))
        revision = await self._storage.current_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered)
        selected = ordered[start : start + limit]
        next_key = _info_key(selected[-1]) if selected and start + len(selected) < len(ordered) else None
        return Page(selected, _make_cursor(revision, kind, prefix, next_key))

    async def list_versions(self, key: AssetKey) -> "tuple[VersionSummary, ...]":
        """List immutable file versions from newest to oldest."""
        self._ensure_ready()
        return await self._storage.list_versions(key)

    async def get_at_revision(
        self,
        key: AssetKey,
        revision: StorageEntryRevision,
    ) -> "bytes | None":
        """Return bytes for one immutable file revision, including None for tombstones."""
        self._ensure_ready()
        versions = await self._storage.list_versions(key)
        if not any(version.entry_revision == revision for version in versions):
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        return await self._storage.get_at_revision(key, revision)

    async def get_at_version(self, key: AssetKey, version: int) -> "bytes | None":
        """Return bytes for one positive integer file version."""
        return await self.get_at_revision(key, StorageEntryRevision(version))

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "asset store is not initialized")


def _validate_limit(limit: int) -> None:
    if limit < 1 or limit > 200:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _make_cursor(
    revision: StorageRevision,
    kind: "str | None",
    prefix: "str | None",
    key: "AssetKey | None",
) -> "str | None":
    if key is None:
        return None
    payload = json.dumps(
        [revision.value, kind, prefix, key.kind, key.id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _cursor_start(
    cursor: "str | None",
    revision: StorageRevision,
    kind: "str | None",
    prefix: "str | None",
    values: "Sequence[AssetInfo | StorageOwnedInfo[AssetInfo]]",
) -> int:
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode((cursor + padding).encode("ascii")))
        if (
            not isinstance(payload, list)
            or len(payload) != 5
            or payload[0] != revision.value
            or payload[1] != kind
            or payload[2] != prefix
            or not isinstance(payload[3], str)
            or not isinstance(payload[4], str)
        ):
            raise ValueError
        last = payload[3], payload[4]
    except (
        ValueError,
        TypeError,
        UnicodeEncodeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        binascii.Error,
    ):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    return next(
        (
            index
            for index, value in enumerate(values)
            if (_info_key(value).kind, _info_key(value).id) > last
        ),
        len(values),
    )


def _info_key(value: "AssetInfo | StorageOwnedInfo[AssetInfo]") -> AssetKey:
    return value.info.key if isinstance(value, StorageOwnedInfo) else value.key


__all__ = ["AssetCacheAdapter", "AssetStore"]
