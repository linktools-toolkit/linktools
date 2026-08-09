#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single typed and file-tree AssetStore boundary."""

import base64
import binascii
import json
from collections.abc import Collection, Mapping, Sequence
from typing import TypeVar

from linktools.core import environ

from ..core import Page
from ..errors import AIError, ErrorCode
from ._codec import AssetCodec, AssetCodecManifest, AssetCodecRegistry
from ._composition import AssetComposition
from ._domain import (
    AssetBatchResult,
    AssetChange,
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
    AssetRequest,
    AssetRevision,
    AssetSource,
    AssetStoreRevision,
    AssetValue,
    AssetVersion,
)

AssetT = TypeVar("AssetT", bound=AssetValue)
_logger = environ.get_logger("ai.asset.store")


class AssetStore:
    """Typed asset facade configured with codecs and read-only sources."""

    def __init__(
        self,
        storage: AssetComposition,
        *,
        codecs: AssetCodecRegistry,
        sources: "Sequence[AssetSource]" = (),
    ) -> None:
        """Create a store from storage, typed codecs, and optional read-only sources."""
        self._storage = storage
        self._codecs = codecs
        self._sources = tuple(sources)
        self._ready = False

    @property
    def ready(self) -> bool:
        """Return whether initialization and source synchronization completed."""
        return self._ready

    @property
    def codec_manifest(self) -> "AssetCodecManifest":
        """Return the frozen manifest of registered codecs."""
        return self._codecs.manifest()

    async def initialize(self) -> None:
        """Freeze codecs, initialize storage, and synchronize registered sources."""
        self._ready = False
        self._codecs.freeze()
        await self._storage.initialize()
        try:
            await self.refresh_sources()
        except Exception:
            _logger.exception("asset store source refresh failed during initialization")
            raise
        self._ready = True
        _logger.info("asset store ready: source_count=%s", len(self._sources))

    async def refresh_sources(self) -> AssetStoreRevision:
        """Synchronize all registered sources and return the resulting store revision."""
        source_files: dict[AssetKey, dict[str, tuple[bytes, str]]] = {}
        for entry in self._codecs.manifest().entries:
            codec = self._codecs.codec(entry.kind)
            for source in self._sources:
                for asset in await source.list_assets(entry.kind):
                    files = source_files.setdefault(asset, {})
                    for rel_path in await source.list_files(asset):
                        key = AssetEntryKey(asset, rel_path)
                        content = await source.read_file(key)
                        if rel_path == entry.primary_path:
                            _decode(codec, asset, bytes(content))
                        files[rel_path] = (bytes(content), source.identity(content))
        return await self._storage.sync_sources(source_files, {entry.kind: entry.primary_path for entry in self._codecs.manifest().entries})

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        """Return current container metadata, or None when the asset is absent."""
        self._ensure_ready()
        return await self._storage.stat_asset(key)

    async def get(self, key: AssetKey, *, expected: "type[AssetT]") -> "AssetT | None":
        """Decode the current primary file as the explicitly expected asset type."""
        self._ensure_ready()
        codec = self._codecs.resolve(key.kind, expected)
        content = await self.get_file(AssetEntryKey(key, codec.primary_path))
        if content is None:
            return None
        return _decode(codec, key, content)

    async def get_many(self, requests: "Sequence[AssetRequest[AssetValue]]") -> "tuple[AssetValue | None, ...]":
        """Resolve typed asset requests while preserving request order."""
        values: list[AssetValue | None] = []
        for request in requests:
            values.append(await self.get(request.key, expected=request.expected))
        return tuple(values)

    async def put(self, key: AssetKey, value: AssetValue, *, expected_revision: "AssetRevision | None" = None) -> AssetInfo:
        """Encode and store a typed asset with optional container-revision CAS."""
        self._ensure_ready()
        codec = self._codecs.resolve(key.kind, type(value))
        codec.validate_key(key, value)
        result = await self.put_file(AssetEntryKey(key, codec.primary_path), codec.encode(value), expected_revision=expected_revision)
        info = await self.stat(key)
        if info is None or info.revision != result.asset_revision:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info("asset primary stored: asset=%s/%s revision=%s", key.kind, key.id, info.revision.value)
        return info

    async def apply_batch(self, changes: "Sequence[AssetChange]", *, expected_store_revision: "AssetStoreRevision | None" = None) -> AssetBatchResult:
        """Apply typed container changes with optional whole-store CAS."""
        self._ensure_ready()
        encoded: list[tuple[AssetKey, bytes | None, str, str, AssetRevision | None]] = []
        for change in changes:
            if change.operation == "PUT":
                if change.value is None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                codec = self._codecs.resolve(change.key.kind, type(change.value))
                codec.validate_key(change.key, change.value)
                encoded.append((change.key, codec.encode(change.value), codec.primary_path, "PUT", change.expected_revision))
            elif change.operation == "DELETE":
                encoded.append((change.key, None, self._primary_path(change.key.kind), "DELETE", change.expected_revision))
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        results = await self._storage.apply_asset_batch(encoded, expected_store_revision=expected_store_revision)
        return AssetBatchResult(await self._storage.current_store_revision(), self._storage.atomic_batch, results)

    async def delete(self, key: AssetKey, *, expected_revision: "AssetRevision | None" = None) -> AssetDeleteResult:
        """Delete a container with optional container-revision CAS."""
        self._ensure_ready()
        deleted, info = await self._storage.delete_asset(key, expected_revision=expected_revision)
        return AssetDeleteResult(key, deleted, None if info is None else info.revision, await self._storage.current_store_revision())

    async def list_info(self, *, kind: "str | None" = None, prefix: "str | None" = None, cursor: "str | None" = None, limit: int = 100) -> "Page[AssetInfo]":
        """Page current containers by kind and ID prefix using the returned cursor."""
        self._ensure_ready()
        _validate_limit(limit)
        values = [item for item in await self._storage.list_assets() if (kind is None or item.key.kind == kind) and (prefix is None or item.key.id.startswith(prefix))]
        ordered = tuple(sorted(values, key=lambda item: (item.key.kind, item.key.id)))
        revision = await self._storage.current_store_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered)
        selected = ordered[start:start + limit]
        next_key = selected[-1].key if selected and start + len(selected) < len(ordered) else None
        next_cursor = _make_cursor(revision, kind, prefix, next_key)
        return Page(selected, next_cursor)

    async def list_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        """List immutable container versions from newest to oldest."""
        self._ensure_ready()
        return await self._storage.list_asset_versions(key)

    async def get_at_revision(self, key: AssetKey, revision: AssetRevision, *, expected: "type[AssetT]") -> "AssetT | None":
        """Decode a typed asset from an immutable container revision."""
        self._ensure_ready()
        codec = self._codecs.resolve(key.kind, expected)
        files = await self._storage.asset_revision_files(key, revision)
        primary = next((item for item in files if item.key.rel_path == codec.primary_path), None)
        if primary is None or primary.deleted or primary.content is None:
            return None
        return _decode(codec, key, primary.content)

    async def get_at_version(self, key: AssetKey, version: int, *, expected: "type[AssetT]") -> "AssetT | None":
        """Decode a typed asset using an integer container version."""
        return await self.get_at_revision(key, AssetRevision(version), expected=expected)

    async def stat_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "AssetEntryInfo | None":
        """Return current file metadata, optionally including a tombstone."""
        self._ensure_ready()
        current = await self._storage.current_file(key, include_deleted=include_deleted)
        return None if current is None else current[0]

    async def get_file(self, key: AssetEntryKey) -> "bytes | None":
        """Return current file bytes, or None for missing and deleted files."""
        self._ensure_ready()
        current = await self._storage.current_file(key)
        return None if current is None else current[1]

    async def put_file(self, key: AssetEntryKey, value: bytes, *, expected_entry_revision: "AssetEntryRevision | None" = None, expected_revision: "AssetRevision | None" = None) -> AssetEntryInfo:
        """Store one file with optional file and container revision CAS checks."""
        self._ensure_ready()
        self._validate_primary_bytes(key, value)
        return await self._storage.put_file(key, value, primary_path=self._primary_path(key.asset.kind), expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)

    async def delete_file(self, key: AssetEntryKey, *, expected_entry_revision: "AssetEntryRevision | None" = None, expected_revision: "AssetRevision | None" = None) -> AssetEntryDeleteResult:
        """Delete one non-primary file with optional file and container CAS checks."""
        self._ensure_ready()
        return await self._storage.delete_file(key, primary_path=self._primary_path(key.asset.kind), expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)

    async def apply_file_batch(self, asset: AssetKey, changes: "Sequence[AssetEntryChange]", *, expected_revision: "AssetRevision | None" = None, expected_store_revision: "AssetStoreRevision | None" = None) -> AssetEntryBatchResult:
        """Apply ordered file changes with optional container and store CAS checks."""
        self._ensure_ready()
        for change in changes:
            if change.operation == "PUT" and change.value is not None:
                self._validate_primary_bytes(AssetEntryKey(asset, change.rel_path), change.value)
        return await self._storage.apply_file_batch(asset, changes, primary_path=self._primary_path(asset.kind), expected_revision=expected_revision, expected_store_revision=expected_store_revision)

    async def list_files(self, asset: AssetKey, *, prefix: "str | None" = None, include_deleted: bool = False) -> "tuple[AssetEntryInfo, ...]":
        """List current file metadata under an asset and optional path prefix."""
        self._ensure_ready()
        return tuple(item[0] for item in await self._storage.list_current_files(asset, prefix=prefix, include_deleted=include_deleted))

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        """List immutable versions of one file from newest to oldest."""
        self._ensure_ready()
        return await self._storage.list_file_versions(key)

    async def get_file_at_revision(self, key: AssetEntryKey, entry_revision: AssetEntryRevision) -> "bytes | None":
        """Return file bytes at an immutable file-entry revision."""
        self._ensure_ready()
        return await self._storage.get_file_at_revision(key, entry_revision)

    async def get_file_at_version(self, key: AssetEntryKey, version: int) -> "bytes | None":
        """Return file bytes using an integer file-entry version."""
        return await self.get_file_at_revision(key, AssetEntryRevision(version))

    async def snapshot_files(self, asset: AssetKey, *, revision: "AssetRevision | None" = None, include_deleted: bool = False) -> "tuple[AssetEntrySnapshot, ...]":
        """Return a complete file snapshot for the current or specified container revision."""
        self._ensure_ready()
        return await self._storage.snapshot_files(asset, revision, include_deleted)

    async def replace_tree(self, asset: AssetKey, files: "Mapping[str, bytes]", *, deleted_rel_paths: "Collection[str]" = (), expected_revision: "AssetRevision | None" = None) -> AssetInfo:
        """Replace an asset tree with optional tombstones and container-revision CAS."""
        self._ensure_ready()
        for path, value in files.items():
            self._validate_primary_bytes(AssetEntryKey(asset, path), value)
        return await self._storage.replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=self._primary_path(asset.kind), expected_revision=expected_revision)

    async def restore(self, asset: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        """Restore a historical container revision against the expected current revision."""
        self._ensure_ready()
        return await self._storage.restore_asset(asset, revision, expected_revision=expected_revision)

    async def rename(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        """Rename a container against its expected current revision."""
        self._ensure_ready()
        return await self._storage.rename_asset(source, target, expected_source_revision=expected_source_revision)

    def _primary_path(self, kind: str) -> str:
        """Resolve the registered primary file path for an asset kind."""
        return self._codecs.primary_path(kind)

    def _ensure_ready(self) -> None:
        """Reject operations until initialize() completes successfully."""
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _validate_primary_bytes(self, key: AssetEntryKey, value: bytes) -> None:
        """Decode primary-file bytes before they reach a backend mutation."""
        if key.rel_path != self._primary_path(key.asset.kind):
            return
        raw_codec = self._codecs.codec(key.asset.kind)
        _decode(raw_codec, key.asset, value)


def _decode(codec: "AssetCodec[AssetValue]", key: AssetKey, content: bytes) -> AssetValue:
    try:
        value = codec.decode(content)
        codec.validate_key(key, value)
        return value
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error


def _validate_limit(limit: int) -> None:
    if limit < 1 or limit > 200:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _make_cursor(revision: AssetStoreRevision, kind: "str | None", prefix: "str | None", key: "AssetKey | None") -> "str | None":
    if key is None:
        return None
    payload = json.dumps(
        [1, revision.value, kind, prefix, key.kind, key.id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _cursor_start(cursor: "str | None", revision: AssetStoreRevision, kind: "str | None", prefix: "str | None", values: "Sequence[AssetInfo]") -> int:
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode((cursor + padding).encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, list)
            or len(payload) != 6
            or payload[0] != 1
            or payload[1] != revision.value
            or payload[2] != kind
            or payload[3] != prefix
            or not isinstance(payload[4], str)
            or not isinstance(payload[5], str)
        ):
            raise ValueError
        last = payload[4], payload[5]
    except (ValueError, TypeError, UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    return next((index for index, value in enumerate(values) if (value.key.kind, value.key.id) > last), len(values))


__all__ = ["AssetStore"]
