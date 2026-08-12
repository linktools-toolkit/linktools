#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory backend for versioned Asset files."""

import asyncio
import base64
import binascii
import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone

from linktools.core import environ

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
    VersionSummary,
)
from ._domain import AssetInfo, AssetKey, AssetRoot

_logger = environ.get_logger("ai.asset.backend")


class InMemoryAssetBackend:
    """Store each AssetKey as one independently versioned bytes file."""

    def __init__(self, root: "AssetRoot | None" = None, *, writable: bool = True) -> None:
        self._root = root or AssetRoot(
            "memory:default",
            "memory",
            "memory",
            hashlib.sha256(b"memory:default").hexdigest(),
        )
        self._writable = writable
        self._entries: dict[AssetKey, tuple[AssetInfo, bytes]] = {}
        self._versions: dict[AssetKey, list[tuple[AssetInfo, bytes]]] = {}
        self._revision = 0
        self._lock = asyncio.Lock()

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return self._writable

    @property
    def atomic_batch(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def head_revision(self) -> StorageRevision:
        async with self._lock:
            return self._store_revision()

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        async with self._lock:
            current = self._store_revision()
            if after_revision == current:
                return MetadataLoad(MetadataLoadMode.PATCH, current, ())
            return MetadataLoad(
                MetadataLoadMode.REPLACE,
                current,
                tuple(MetadataChange(key, value[0]) for key, value in self._entries.items()),
            )

    async def get(self, key: AssetKey) -> "bytes | None":
        async with self._lock:
            current = self._entries.get(key)
            return None if current is None or current[0].status is not StorageEntryStatus.NORMAL else current[1]

    async def get_many(self, keys: "Sequence[AssetKey]") -> "dict[AssetKey, bytes]":
        async with self._lock:
            return {
                key: current[1]
                for key in keys
                if (current := self._entries.get(key)) is not None
                and current[0].status is StorageEntryStatus.NORMAL
            }

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        async with self._lock:
            current = self._entries.get(key)
            return None if current is None else current[0]

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StoragePutResult[AssetInfo]":
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            self._check_revision(previous, expected_entry_revision)
            if previous is not None and previous[0].status is StorageEntryStatus.NORMAL and previous[0].etag == _etag(value):
                return StoragePutResult(previous[0], previous[0].revision, self._store_revision(), False)
            self._revision += 1
            info = self._next_info(key, value, previous, status=StorageEntryStatus.NORMAL)
            self._record(info, value)
            _logger.debug("asset file stored: kind=%s id=%s revision=%s", key.kind, key.id, info.revision)
            return StoragePutResult(info, info.revision, info.store_revision, True)

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            self._check_revision(previous, expected_entry_revision)
            if previous is None or previous[0].status is StorageEntryStatus.DELETED:
                return StorageDeleteResult(key, False, None, self._store_revision())
            self._revision += 1
            info = self._next_info(key, b"", previous, status=StorageEntryStatus.DELETED)
            self._record(info, b"")
            _logger.debug("asset file deleted: kind=%s id=%s revision=%s", key.kind, key.id, info.revision)
            return StorageDeleteResult(key, True, info.revision, info.store_revision)

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            self._check_revision(previous, expected_entry_revision)
            if previous is None or previous[0].status is StorageEntryStatus.RESET:
                return StorageResetResult(key, False, self._store_revision())
            self._revision += 1
            info = self._next_info(key, b"", previous, status=StorageEntryStatus.RESET)
            self._record(info, b"")
            _logger.debug("asset file reset: kind=%s id=%s revision=%s", key.kind, key.id, self._revision)
            return StorageResetResult(key, True, self._store_revision())

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        async with self._lock:
            self._require_writable()
            if expected_revision is not None and expected_revision != self._store_revision():
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if len({change.key for change in changes}) != len(changes):
                raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
            previous = {change.key: self._entries.get(change.key) for change in changes}
            for change in changes:
                self._check_revision(previous[change.key], change.expected_entry_revision)
            mutates = any(_change_mutates(change, previous[change.key]) for change in changes)
            if mutates:
                self._revision += 1
            store_revision = self._store_revision()
            results: list[StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]] = []
            for change in changes:
                current = previous[change.key]
                if change.operation is StorageOperation.PUT:
                    value = bytes(change.value or b"")
                    if not _change_mutates(change, current) and current is not None:
                        results.append(StoragePutResult(current[0], current[0].revision, store_revision, False))
                        continue
                    info = self._next_info(change.key, value, current, status=StorageEntryStatus.NORMAL)
                    self._record(info, value)
                    results.append(StoragePutResult(info, info.revision, store_revision, True))
                elif change.operation is StorageOperation.DELETE and (
                    current is None or current[0].status is StorageEntryStatus.DELETED
                ):
                    results.append(StorageDeleteResult(change.key, False, None, store_revision))
                elif change.operation is StorageOperation.DELETE:
                    info = self._next_info(change.key, b"", current, status=StorageEntryStatus.DELETED)
                    self._record(info, b"")
                    results.append(StorageDeleteResult(change.key, True, info.revision, store_revision))
                elif current is None or current[0].status is StorageEntryStatus.RESET:
                    results.append(StorageResetResult(change.key, False, store_revision))
                else:
                    info = self._next_info(change.key, b"", current, status=StorageEntryStatus.RESET)
                    self._record(info, b"")
                    results.append(StorageResetResult(change.key, True, store_revision))
            if mutates:
                _logger.debug("asset batch committed: changes=%s revision=%s", len(changes), self._revision)
            return StorageBatchResult(store_revision, True, tuple(results))

    async def list_versions(self, key: AssetKey) -> "tuple[VersionSummary, ...]":
        async with self._lock:
            return tuple(
                VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status)
                for info, _ in self._versions.get(key, ())
            )

    async def get_at_revision(
        self,
        key: AssetKey,
        entry_revision: StorageEntryRevision,
    ) -> "bytes | None":
        async with self._lock:
            for info, value in self._versions.get(key, ()):
                if info.revision == entry_revision:
                    return None if info.status is not StorageEntryStatus.NORMAL else value
            return None

    async def get_at_version(self, key: AssetKey, version: int) -> "bytes | None":
        return await self.get_at_revision(key, StorageEntryRevision(version))

    def export_state(self) -> "dict[str, object]":
        return {
            "store_revision": self._revision,
            "entries": [
                _encode_entry(info, value)
                for info, value in self._entries.values()
            ],
            "versions": [
                _encode_entry(info, value)
                for history in self._versions.values()
                for info, value in history
            ],
        }

    def import_state(self, raw: object) -> None:
        if not isinstance(raw, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        revision = raw.get("store_revision")
        current = raw.get("entries")
        versions = raw.get("versions")
        if not isinstance(revision, int) or revision < 0 or not isinstance(versions, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        histories: dict[AssetKey, list[tuple[AssetInfo, bytes]]] = {}
        for item in versions:
            info, value = _decode_entry(item, self._root)
            histories.setdefault(info.key, []).append((info, value))
        if current is None:
            entries = {
                key: history[-1]
                for key, history in histories.items()
                if history
            }
        elif isinstance(current, list):
            entries = {}
            for item in current:
                info, value = _decode_entry(item, self._root)
                entries[info.key] = (info, value)
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._revision = revision
        self._entries = entries
        self._versions = histories

    def _next_info(
        self,
        key: AssetKey,
        value: bytes,
        previous: "tuple[AssetInfo, bytes] | None",
        *,
        status: StorageEntryStatus,
    ) -> AssetInfo:
        history = self._versions.get(key, ())
        if previous is not None:
            previous_revision = previous[0].revision.value
        elif history:
            previous_revision = history[-1][0].revision.value
        else:
            previous_revision = 0
        entry_revision = StorageEntryRevision(previous_revision + 1)
        return AssetInfo(
            key,
            entry_revision,
            self._store_revision(),
            _etag(value),
            len(value),
            status,
            self._root.root_id,
            self._root.digest,
            datetime.now(timezone.utc),
        )

    def _record(self, info: AssetInfo, value: bytes) -> None:
        self._entries[info.key] = (info, value)
        self._versions.setdefault(info.key, []).append((info, value))

    def _store_revision(self) -> StorageRevision:
        return StorageRevision(str(self._revision))

    def _require_writable(self) -> None:
        if not self._writable:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)

    @staticmethod
    def _check_revision(
        previous: "tuple[AssetInfo, bytes] | None",
        expected: "StorageEntryRevision | None",
    ) -> None:
        if expected is not None and (previous is None or previous[0].revision != expected):
            raise AIError(ErrorCode.STORAGE_CONFLICT)


def _change_mutates(
    change: "StorageChange[AssetKey, bytes]",
    previous: "tuple[AssetInfo, bytes] | None",
) -> bool:
    if change.operation is StorageOperation.DELETE:
        return previous is not None and previous[0].status is not StorageEntryStatus.DELETED
    if change.operation is StorageOperation.RESET:
        return previous is not None and previous[0].status is not StorageEntryStatus.RESET
    value = bytes(change.value or b"")
    return previous is None or previous[0].status is not StorageEntryStatus.NORMAL or previous[0].etag != _etag(value)


def _encode_entry(info: AssetInfo, value: bytes) -> "dict[str, object]":
    return {
        "kind": info.key.kind,
        "id": info.key.id,
        "revision": info.revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "status": info.status.value,
        "modified_at": info.modified_at.isoformat(),
        "content": base64.b64encode(value).decode("ascii"),
    }


def _decode_entry(raw: object, root: AssetRoot) -> "tuple[AssetInfo, bytes]":
    if not isinstance(raw, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = base64.b64decode(str(raw["content"]), validate=True)
        info = AssetInfo(
            AssetKey(str(raw["kind"]), str(raw["id"])),
            StorageEntryRevision(int(raw["revision"])),
            StorageRevision(str(raw["store_revision"])),
            str(raw["etag"]),
            int(raw["size"]),
            StorageEntryStatus(str(raw["status"])),
            root.root_id,
            root.digest,
            datetime.fromisoformat(str(raw["modified_at"])),
        )
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(value) != info.size or _etag(value) != info.etag:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return info, value


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = ["InMemoryAssetBackend"]
