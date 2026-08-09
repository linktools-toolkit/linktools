#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single typed and file-tree AssetStore boundary."""

import binascii
import json
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from linktools.core import environ

from ..core import CursorPayload, CursorSigner, Page
from ..errors import AIError, ErrorCode
from ..storage import StorageComposition
from ._codec import AssetCodec, AssetCodecManifest, AssetCodecRegistry
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


@runtime_checkable
class _TreeBackend(Protocol):
    async def initialize(self) -> None: ...
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


class AssetStore:
    def __init__(
        self,
        *,
        storage: "StorageComposition | _TreeBackend",
        codecs: AssetCodecRegistry,
        cursor_signer: CursorSigner,
        sources: Sequence[AssetSource] = (),
    ) -> None:
        self._backend = _backend_from_storage(storage)
        self._codecs = codecs
        self._cursor_signer = cursor_signer
        self._sources = tuple(sources)
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def codec_manifest(self) -> "AssetCodecManifest":
        return self._codecs.manifest()

    async def initialize(self) -> None:
        self._ready = False
        await self._backend.initialize()
        try:
            await self.refresh_sources()
        except Exception:
            _logger.exception("asset store source refresh failed during initialization")
            raise
        self._ready = True
        _logger.info("asset store ready: source_count=%s", len(self._sources))

    async def refresh_sources(self) -> AssetStoreRevision:
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
        return await self._backend.sync_sources(source_files, {entry.kind: entry.primary_path for entry in self._codecs.manifest().entries})

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        self._ensure_ready()
        return await self._backend.stat_asset(key)

    async def get(self, key: AssetKey, *, expected: "type[AssetT]") -> "AssetT | None":
        self._ensure_ready()
        codec = self._codecs.resolve(key.kind, expected)
        content = await self.get_file(AssetEntryKey(key, codec.primary_path))
        if content is None:
            return None
        return _decode(codec, key, content)

    async def get_many(self, requests: "Sequence[AssetRequest[AssetValue]]") -> "tuple[AssetValue | None, ...]":
        return tuple(await self.get(request.key, expected=request.expected) for request in requests)

    async def put(self, key: AssetKey, value: AssetValue, *, expected_revision: "AssetRevision | None" = None) -> AssetInfo:
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
        results = await self._backend.apply_asset_batch(encoded, expected_store_revision=expected_store_revision)
        return AssetBatchResult(await self._backend.current_revision(), True, results)

    async def delete(self, key: AssetKey, *, expected_revision: "AssetRevision | None" = None) -> AssetDeleteResult:
        self._ensure_ready()
        deleted, info = await self._backend.delete_asset(key, expected_revision=expected_revision)
        return AssetDeleteResult(key, deleted, None if info is None else info.revision, await self._backend.current_revision())

    async def list_info(self, *, kind: "str | None" = None, prefix: "str | None" = None, cursor: "str | None" = None, limit: int = 100) -> "Page[AssetInfo]":
        self._ensure_ready()
        _validate_limit(limit)
        values = [item for item in await self._backend.list_assets() if (kind is None or item.key.kind == kind) and (prefix is None or item.key.id.startswith(prefix))]
        ordered = tuple(sorted(values, key=lambda item: (item.key.kind, item.key.id)))
        revision = await self._backend.current_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered, self._cursor_signer)
        selected = ordered[start:start + limit]
        next_cursor = _make_cursor(revision, kind, prefix, selected[-1].key if len(selected) == limit else None, self._cursor_signer)
        return Page(selected, next_cursor)

    async def list_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        self._ensure_ready()
        return await self._backend.list_asset_versions(key)

    async def get_at_revision(self, key: AssetKey, revision: AssetRevision, *, expected: "type[AssetT]") -> "AssetT | None":
        self._ensure_ready()
        codec = self._codecs.resolve(key.kind, expected)
        files = await self._backend.asset_revision_files(key, revision)
        primary = next((item for item in files if item.key.rel_path == codec.primary_path), None)
        if primary is None or primary.deleted or primary.content is None:
            return None
        return _decode(codec, key, primary.content)

    async def get_at_version(self, key: AssetKey, version: int, *, expected: "type[AssetT]") -> "AssetT | None":
        return await self.get_at_revision(key, AssetRevision(version), expected=expected)

    async def stat_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "AssetEntryInfo | None":
        self._ensure_ready()
        current = await self._backend.current_file(key, include_deleted=include_deleted)
        return None if current is None else current[0]

    async def get_file(self, key: AssetEntryKey) -> "bytes | None":
        self._ensure_ready()
        current = await self._backend.current_file(key)
        return None if current is None else current[1]

    async def put_file(self, key: AssetEntryKey, value: bytes, *, expected_entry_revision: "AssetEntryRevision | None" = None, expected_revision: "AssetRevision | None" = None) -> AssetEntryInfo:
        self._ensure_ready()
        self._validate_primary_bytes(key, value)
        return await self._backend.put_file(key, value, primary_path=self._primary_path(key.asset.kind), expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)

    async def delete_file(self, key: AssetEntryKey, *, expected_entry_revision: "AssetEntryRevision | None" = None, expected_revision: "AssetRevision | None" = None) -> AssetEntryDeleteResult:
        self._ensure_ready()
        return await self._backend.delete_file(key, primary_path=self._primary_path(key.asset.kind), expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)

    async def apply_file_batch(self, asset: AssetKey, changes: "Sequence[AssetEntryChange]", *, expected_revision: "AssetRevision | None" = None, expected_store_revision: "AssetStoreRevision | None" = None) -> AssetEntryBatchResult:
        self._ensure_ready()
        for change in changes:
            if change.operation == "PUT" and change.value is not None:
                self._validate_primary_bytes(AssetEntryKey(asset, change.rel_path), change.value)
        return await self._backend.apply_file_batch(asset, changes, primary_path=self._primary_path(asset.kind), expected_revision=expected_revision, expected_store_revision=expected_store_revision)

    async def list_files(self, asset: AssetKey, *, prefix: "str | None" = None, include_deleted: bool = False) -> "tuple[AssetEntryInfo, ...]":
        self._ensure_ready()
        return tuple(item[0] for item in await self._backend.list_current_files(asset, prefix=prefix, include_deleted=include_deleted))

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        self._ensure_ready()
        return await self._backend.list_file_versions(key)

    async def get_file_at_revision(self, key: AssetEntryKey, entry_revision: AssetEntryRevision) -> "bytes | None":
        self._ensure_ready()
        return await self._backend.get_file_at_revision(key, entry_revision)

    async def get_file_at_version(self, key: AssetEntryKey, version: int) -> "bytes | None":
        return await self.get_file_at_revision(key, AssetEntryRevision(version))

    async def snapshot_files(self, asset: AssetKey, *, revision: "AssetRevision | None" = None, include_deleted: bool = False) -> "tuple[AssetEntrySnapshot, ...]":
        self._ensure_ready()
        return await self._backend.snapshot_files(asset, revision, include_deleted)

    async def replace_tree(self, asset: AssetKey, files: "Mapping[str, bytes]", *, deleted_rel_paths: "Collection[str]" = (), expected_revision: "AssetRevision | None" = None) -> AssetInfo:
        self._ensure_ready()
        for path, value in files.items():
            self._validate_primary_bytes(AssetEntryKey(asset, path), value)
        return await self._backend.replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=self._primary_path(asset.kind), expected_revision=expected_revision)

    async def restore(self, asset: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        self._ensure_ready()
        return await self._backend.restore_asset(asset, revision, expected_revision=expected_revision)

    async def rename(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        self._ensure_ready()
        return await self._backend.rename_asset(source, target, expected_source_revision=expected_source_revision)

    def _primary_path(self, kind: str) -> str:
        return self._codecs.primary_path(kind)

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _validate_primary_bytes(self, key: AssetEntryKey, value: bytes) -> None:
        if key.rel_path != self._primary_path(key.asset.kind):
            return
        raw_codec = self._codecs.codec(key.asset.kind)
        _decode(raw_codec, key.asset, value)


def _backend_from_storage(storage: "StorageComposition | _TreeBackend") -> _TreeBackend:
    backend = storage.primary if isinstance(storage, StorageComposition) else storage
    if not isinstance(backend, _TreeBackend):
        raise TypeError("AssetStore requires a tree-aware Asset backend")
    return backend


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


def _make_cursor(revision: AssetStoreRevision, kind: "str | None", prefix: "str | None", key: "AssetKey | None", signer: CursorSigner) -> "str | None":
    if key is None:
        return None
    return signer.encode(CursorPayload(1, "asset", "ASSET", f"{kind}:{prefix}", json.dumps([key.kind, key.id], separators=(",", ":")), int(revision.value), int(time.time()) + 3600, False))


def _cursor_start(cursor: "str | None", revision: AssetStoreRevision, kind: "str | None", prefix: "str | None", values: "Sequence[AssetInfo]", signer: CursorSigner) -> int:
    if cursor is None:
        return 0
    try:
        payload = signer.decode(cursor)
        if payload.cursor_version != 1 or payload.resource_kind != "ASSET" or payload.tenant_id != "asset" or payload.filter_digest != f"{kind}:{prefix}" or payload.snapshot_or_store_revision != int(revision.value):
            raise ValueError
        last = json.loads(payload.sort_key)
        if not isinstance(last, list) or len(last) != 2 or not all(isinstance(item, str) for item in last):
            raise ValueError
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    return next((index for index, value in enumerate(values) if (value.key.kind, value.key.id) > tuple(last)), len(values))


__all__ = ["AssetStore"]
