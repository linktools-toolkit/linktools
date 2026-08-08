#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed AssetStore backed by the generic StorageComposition."""

import binascii
import hashlib
import json
import time
from collections.abc import Sequence
from typing import TypeVar

from linktools.core import environ

from ..core import CursorPayload, CursorSigner, Page
from ..core.errors import ErrorCode, AIError
from ..core.ids import canonical_sha256
from ..storage.composition import CacheAdapter, StorageAdapter, StorageComposition
from ..storage.layer import StorageWriteVisibility
from ..storage.model import StorageChange, StorageDeleteResult, StorageOperation
from ..storage.model import StorageBatchPartialError
from ._codec import AssetCodecManifest, AssetCodecRegistry
from .model import (
    AssetBatchResult,
    AssetBatchPartialError,
    AssetChange,
    AssetDeleteResult,
    AssetInfo,
    AssetKey,
    AssetRequest,
    AssetRevision,
    AssetStoreRevision,
    AssetValue,
    AssetVersion,
    OwnedAssetInfo,
)

AssetT = TypeVar("AssetT", bound=AssetValue)
_logger = environ.get_logger("ai.asset.store")


class _AssetStorageAdapter(StorageAdapter[AssetKey, bytes, AssetKey, bytes, AssetInfo]):
    def to_storage_key(self, key: AssetKey) -> AssetKey:
        return key

    def from_storage_key(self, key: AssetKey) -> AssetKey:
        return key

    def from_storage_value(self, value: bytes) -> bytes:
        return value

    def to_storage_value(self, value: bytes) -> bytes:
        return value

    def validate_value(self, key: AssetKey, value: bytes, info: AssetInfo) -> None:
        if info.key != key or len(value) != info.size or hashlib.sha256(value).hexdigest() != info.etag:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)


class _AssetCacheAdapter(CacheAdapter[AssetKey, bytes, AssetInfo]):
    def cache_key(self, key: AssetKey, info: AssetInfo) -> str:
        return f"asset:{info.root_digest}:{key.kind}:{key.id}:{info.entry_revision.value}:{info.etag}"

    def to_cache(self, value: bytes) -> bytes:
        return value

    def from_cache(self, value: bytes) -> bytes:
        return value


class AssetStore:
    def __init__(
        self,
        *,
        storage: 'StorageComposition[AssetKey, bytes, AssetKey, bytes, AssetInfo, AssetRevision, AssetStoreRevision]',
        codecs: AssetCodecRegistry,
        cursor_signer: CursorSigner,
    ) -> None:
        if storage.write_visibility is not StorageWriteVisibility.READABLE:
            raise ValueError("AssetStore requires readable writes")
        if not storage.writer_is_primary:
            raise ValueError("AssetStore requires the primary backend as writer")
        self._storage = storage
        self._codecs = codecs
        self._cursor_signer = cursor_signer

    @property
    def codec_manifest(self) -> AssetCodecManifest:
        return self._codecs.manifest()

    async def stat(self, key: AssetKey) -> 'AssetInfo | None':
        return await self._storage.stat(key)

    async def get(self, key: AssetKey, *, expected: 'type[AssetT]') -> 'AssetT | None':
        codec = self._codecs.resolve(key.kind, expected)
        info = await self.stat(key)
        if info is None or info.deleted:
            return None
        encoded = await self._storage.get(key)
        if encoded is None:
            return None
        try:
            value = codec.decode(encoded)
            codec.validate_key(key, value)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
        return value

    async def get_many(self, requests: 'Sequence[AssetRequest[AssetValue]]') -> 'tuple[AssetValue | None, ...]':
        codecs = tuple(self._codecs.resolve(request.key.kind, request.expected) for request in requests)
        infos = {info.key: info for info in await self._storage.list_info()}
        active_keys = tuple(
            request.key
            for request in requests
            if (info := infos.get(request.key)) is not None and not info.deleted
        )
        active_values = await self._storage.get_many(active_keys)
        encoded_by_key = dict(zip(active_keys, active_values))
        values: list[AssetValue | None] = []
        for request, codec in zip(requests, codecs):
            info = infos.get(request.key)
            content = encoded_by_key.get(request.key)
            if info is None or info.deleted or content is None:
                values.append(None)
                continue
            try:
                value = codec.decode(content)
                codec.validate_key(request.key, value)
            except AIError:
                raise
            except Exception as error:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
            values.append(value)
        return tuple(values)

    async def put(
        self,
        key: AssetKey,
        value: AssetValue,
        *,
        expected_entry_revision: 'AssetRevision | None' = None,
    ) -> AssetInfo:
        codec = self._codecs.resolve(key.kind, type(value))
        codec.validate_key(key, value)
        result = await self._storage.put(key, codec.encode(value), expected_entry_revision=expected_entry_revision)
        _logger.info("asset stored: kind=%s id=%s revision=%s", key.kind, key.id, result.entry_revision)
        return result.info

    async def apply_batch(
        self,
        changes: 'Sequence[AssetChange]',
        *,
        expected_store_revision: 'AssetStoreRevision | None' = None,
    ) -> AssetBatchResult:
        encoded: list[StorageChange[AssetKey, bytes, AssetRevision]] = []
        for change in changes:
            if change.operation == "PUT":
                if change.value is None:
                    raise ValueError("PUT changes require a value")
                codec = self._codecs.resolve(change.key.kind, type(change.value))
                codec.validate_key(change.key, change.value)
                encoded.append(StorageChange(StorageOperation.PUT, change.key, codec.encode(change.value), change.expected_entry_revision))
            elif change.operation == "DELETE":
                encoded.append(StorageChange(StorageOperation.DELETE, change.key, None, change.expected_entry_revision))
            else:
                raise ValueError(f"unsupported asset operation: {change.operation}")
        try:
            result = await self._storage.apply_batch(
                encoded,
                expected_store_revision=expected_store_revision,
            )
        except StorageBatchPartialError as exc:
            raise AssetBatchPartialError(exc.failure) from exc
        values: list[AssetInfo | AssetDeleteResult] = []
        for index, item in enumerate(result.results):
            if isinstance(item, StorageDeleteResult):
                values.append(AssetDeleteResult(changes[index].key, item.deleted, item.entry_revision, item.store_revision))
            else:
                values.append(item.info)
        return AssetBatchResult(result.store_revision, result.atomic, tuple(values))

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: 'AssetRevision | None' = None,
    ) -> AssetDeleteResult:
        result = await self._storage.delete(key, expected_entry_revision=expected_entry_revision)
        return AssetDeleteResult(key, result.deleted, result.entry_revision, result.store_revision)

    async def list_info(
        self,
        *,
        kind: 'str | None' = None,
        prefix: 'str | None' = None,
        cursor: 'str | None' = None,
        limit: int = 100,
    ) -> 'Page[AssetInfo]':
        infos = [info for info in await self._storage.list_info() if not info.deleted]
        infos = [info for info in infos if (kind is None or info.key.kind == kind) and (prefix is None or info.key.id.startswith(prefix))]
        revision = await self._storage.current_revision()
        return _page_infos(infos, revision, kind, prefix, cursor, limit, self._cursor_signer)

    async def list_info_with_owners(
        self,
        *,
        kind: 'str | None' = None,
        prefix: 'str | None' = None,
        cursor: 'str | None' = None,
        limit: int = 100,
    ) -> 'Page[OwnedAssetInfo]':
        if limit < 1 or limit > 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        owned = [item for item in await self._storage.list_info_with_owners() if not item.info.deleted]
        owned = [item for item in owned if (kind is None or item.info.key.kind == kind) and (prefix is None or item.info.key.id.startswith(prefix))]
        ordered = sorted(owned, key=lambda item: (item.info.key.kind, item.info.key.id))
        revision = await self._storage.current_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered, self._cursor_signer)
        page = ordered[start : start + limit]
        next_cursor = _make_cursor(revision, kind, prefix, page[-1].info.key if len(page) == limit else None, self._cursor_signer)
        return Page(tuple(OwnedAssetInfo(item.info, item.layer, item.writable) for item in page), next_cursor)

    async def list_versions(self, key: AssetKey) -> 'tuple[AssetVersion, ...]':
        versions = await self._storage.list_versions(key)
        ordered = sorted(versions, key=lambda version: version.entry_revision.value, reverse=True)
        result: list[AssetVersion] = []
        for version in ordered:
            result.append(
                AssetVersion(
                    version.entry_revision,
                    version.digest,
                    version.size,
                    version.created_at,
                    version.deleted,
                )
            )
        return tuple(result)

    async def get_at_revision(self, key: AssetKey, entry_revision: AssetRevision, *, expected: 'type[AssetT]') -> 'AssetT | None':
        codec = self._codecs.resolve(key.kind, expected)
        versions = await self._storage.list_versions(key)
        version = next((item for item in versions if item.entry_revision == entry_revision), None)
        if version is None:
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        if version.deleted:
            return None
        content = await self._storage.get_at_revision(key, entry_revision)
        if content is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            value = codec.decode(content)
            codec.validate_key(key, value)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
        return value

    async def get_at_version(self, key: AssetKey, version: int, *, expected: 'type[AssetT]') -> 'AssetT | None':
        codec = self._codecs.resolve(key.kind, expected)
        versions = await self._storage.list_versions(key)
        summary = next((item for item in versions if item.entry_revision.value == version), None)
        if summary is None:
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        if summary.deleted:
            return None
        content = await self._storage.get_at_version(key, version)
        if content is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            value = codec.decode(content)
            codec.validate_key(key, value)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
        return value


def _page_infos(
    infos: 'Sequence[AssetInfo]',
    revision: AssetStoreRevision,
    kind: 'str | None',
    prefix: 'str | None',
    cursor: 'str | None',
    limit: int,
    signer: CursorSigner,
) -> 'Page[AssetInfo]':
    if limit < 1 or limit > 200:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
    ordered = sorted(infos, key=lambda info: (info.key.kind, info.key.id))
    start = _cursor_start(cursor, revision, kind, prefix, ordered, signer)
    page = ordered[start : start + limit]
    next_cursor = _make_cursor(revision, kind, prefix, page[-1].key if len(page) == limit else None, signer)
    return Page(tuple(page), next_cursor)


def _make_cursor(revision: AssetStoreRevision, kind: 'str | None', prefix: 'str | None', key: 'AssetKey | None', signer: CursorSigner) -> 'str | None':
    if key is None:
        return None
    return signer.encode(CursorPayload(1, "asset", "ASSET", canonical_sha256({"kind": kind, "prefix": prefix, "include_deleted": False}), json.dumps([key.kind, key.id], separators=(",", ":")), int(revision.value) if revision.value.isdigit() else 0, int(time.time()) + 3600, False))


def _cursor_start(cursor: 'str | None', revision: AssetStoreRevision, kind: 'str | None', prefix: 'str | None', values: 'Sequence[AssetInfo | OwnedAssetInfo]', signer: CursorSigner) -> int:
    if cursor is None:
        return 0
    try:
        payload = signer.decode(cursor)
        if payload.cursor_version != 1 or payload.resource_kind != "ASSET" or payload.tenant_id != "asset" or payload.include_deleted or payload.filter_digest != canonical_sha256({"kind": kind, "prefix": prefix, "include_deleted": False}) or not revision.value.isdigit() or str(payload.snapshot_or_store_revision) != revision.value:
            raise ValueError
        raw_last = json.loads(payload.sort_key)
        if not isinstance(raw_last, list) or len(raw_last) != 2 or not all(isinstance(item, str) for item in raw_last):
            raise ValueError
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    for index, value in enumerate(values):
        info = value.info if isinstance(value, OwnedAssetInfo) else value
        if (info.key.kind, info.key.id) > tuple(raw_last):
            return index
    return len(values)


__all__ = ["AssetStore"]
