#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw file AssetStore backed by StorageOverlay."""

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast

from linktools.core import environ

from ..core import (
    JsonValue,
    Page,
    canonical_json_bytes,
    canonical_sha256,
    validate_idempotency_key,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ..storage import (
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOverlay,
    StorageResetResult,
    StorageRevision,
    StorageOwnedInfo,
    StorageWriteState,
    VersionSummary,
)
from ..storage import ObjectRef, ObjectStore, read_object
from ._domain import AssetInfo, AssetKey

_logger = environ.get_logger("ai.asset.store")


class AssetCacheAdapter:
    """Cache raw Asset file bytes using immutable metadata."""

    def cache_key(self, key: AssetKey, info: AssetInfo) -> str:
        return ":".join(
            (
                "asset",
                info.root_digest,
                key.kind,
                key.id,
                str(info.revision.value),
                info.etag,
            )
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
        self._closing = False
        self._closed = False

    @property
    def ready(self) -> bool:
        """Return whether storage initialization completed successfully."""
        return self._ready

    @property
    def atomic_batch(self) -> bool:
        """Return whether the underlying storage commits batches atomically."""
        return self._storage.atomic_batch

    async def current_revision(self) -> StorageRevision:
        """Return the current effective storage revision."""
        self._ensure_ready()
        return await self._storage.current_revision()

    async def initialize(self) -> None:
        """Initialize configured storage backends before serving requests."""
        if self._closed or self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._ready:
            return
        try:
            await self._storage.initialize()
        except BaseException:
            self._closing = True
            raise
        self._ready = True
        _logger.debug("asset store initialized")

    async def close(self) -> None:
        """Close this store and every backend initialized by its overlay."""
        if self._closed:
            return
        if not self._ready and not self._closing:
            return
        self._closing = True
        self._ready = False
        await self._storage.close()
        self._closing = False
        self._closed = True
        _logger.debug("asset store closed")

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        """Return current effective file metadata and status."""
        self._ensure_ready()
        return await self._storage.stat(key)

    async def get(self, key: AssetKey) -> "bytes | None":
        """Return current file bytes, or None when no file is visible."""
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
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> AssetInfo:
        """Store one file with an optional current-revision check."""
        self._ensure_ready()
        result = await self._storage.put(
            key,
            bytes(value),
            expected_revision=expected_revision,
            metadata=metadata,
        )
        _logger.debug(
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
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        """Delete one file with an optional current-revision check."""
        self._ensure_ready()
        result = await self._storage.delete(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        _logger.debug(
            "asset file delete: kind=%s id=%s deleted=%s",
            key.kind,
            key.id,
            result.deleted,
        )
        return result

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        """Reset one file so a lower read layer becomes effective."""
        self._ensure_ready()
        result = await self._storage.reset(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        _logger.debug(
            "asset file reset: kind=%s id=%s reset=%s",
            key.kind,
            key.id,
            result.reset,
        )
        return result

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
        idempotency_key: "str | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        """Apply one atomic Asset batch, optionally with durable idempotency."""
        self._ensure_ready()
        if not self.atomic_batch:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        request_digest = None
        if idempotency_key is not None:
            validate_idempotency_key(idempotency_key)
            request_digest = _batch_request_digest(changes, expected_revision)
        result = await self._storage.apply_batch(
            changes,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_digest=request_digest,
        )
        if idempotency_key is not None and (
            result.idempotency_key != idempotency_key
            or result.request_digest != request_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result

    async def batch_result(
        self,
        idempotency_key: str,
    ) -> "StorageBatchResult[AssetInfo, AssetKey] | None":
        """Read a writer-owned committed batch receipt."""
        self._ensure_ready()
        validate_idempotency_key(idempotency_key)
        return await self._storage.batch_result(idempotency_key)

    async def write_states(
        self,
        keys: "Sequence[AssetKey]",
    ) -> "Mapping[AssetKey, StorageWriteState[AssetInfo]]":
        """Return effective and writer-local state for raw Asset keys."""
        self._ensure_ready()
        return await self._storage.write_states(keys)

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
        limit = validate_page_limit(limit)
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
        next_key = (
            selected[-1].key
            if selected and start + len(selected) < len(ordered)
            else None
        )
        return Page(selected, _make_cursor(revision, kind, prefix, next_key))

    async def metadata_snapshot(self) -> "tuple[AssetInfo, ...]":
        """Return one stable, active metadata snapshot for a freeze operation."""
        self._ensure_ready()
        values = await self._storage.list_info()
        return tuple(
            sorted(
                (
                    info
                    for info in values
                    if info.status is StorageEntryStatus.NORMAL
                ),
                key=lambda info: (info.key.kind, info.key.id),
            )
        )

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
        limit = validate_page_limit(limit)
        values = [
            owned
            for owned in await self._storage.list_info_with_owners()
            if owned.info.status is StorageEntryStatus.NORMAL
            and (kind is None or owned.info.key.kind == kind)
            and (prefix is None or owned.info.key.id.startswith(prefix))
        ]
        ordered = tuple(
            sorted(values, key=lambda owned: (owned.info.key.kind, owned.info.key.id))
        )
        revision = await self._storage.current_revision()
        start = _cursor_start(cursor, revision, kind, prefix, ordered)
        selected = ordered[start : start + limit]
        next_key = (
            _info_key(selected[-1])
            if selected and start + len(selected) < len(ordered)
            else None
        )
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
        """Return bytes for one immutable file revision."""
        self._ensure_ready()
        versions = await self._storage.list_versions(key)
        if not any(version.entry_revision == revision for version in versions):
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        return await self._storage.get_at_revision(key, revision)

    async def get_at_version(self, key: AssetKey, version: int) -> "bytes | None":
        """Return bytes for one positive integer file version."""
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def snapshot(
        self,
        keys: Sequence[AssetKey],
        *,
        object_store: ObjectStore,
        expected_revision: StorageRevision | None = None,
    ) -> ObjectRef:
        """Publish a deterministic, read-only snapshot of selected assets."""
        self._ensure_ready()
        captured_revision = await self.current_revision()
        if expected_revision is not None and captured_revision != expected_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        selected = tuple(keys)
        if len(set(selected)) != len(selected):
            raise ValueError("asset snapshot keys must be unique")
        infos = {info.key: info for info in await self.metadata_snapshot()}
        entries: list[dict[str, JsonValue]] = []
        for key in sorted(selected, key=lambda item: (item.kind, item.id)):
            info = infos.get(key)
            if info is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self.get(key)
            if value is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            content_key = f"v1/asset-content/{info.etag}"
            await _put_snapshot_object(object_store, content_key, value)
            entries.append(_snapshot_entry(info, content_key))
        if await self.current_revision() != captured_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        manifest: dict[str, JsonValue] = {
            "kind": "asset-snapshot",
            "format_version": 1,
            "captured_revision": captured_revision.value,
            "entries": entries,
        }
        payload = canonical_json_bytes(manifest)
        digest = hashlib.sha256(payload).hexdigest()
        key = f"v1/asset-snapshot/{digest}"
        await _put_snapshot_object(object_store, key, payload)
        _logger.info(
            "asset snapshot published: entries=%s revision=%s digest=%s",
            len(entries),
            captured_revision.value,
            digest,
        )
        return ObjectRef(object_store.store_id, key, digest, len(payload))

    @classmethod
    def from_snapshot(
        cls,
        ref: ObjectRef,
        *,
        object_store: ObjectStore,
    ) -> "AssetStore":
        """Create a read-only AssetStore backed only by a snapshot manifest."""
        if not isinstance(ref, ObjectRef):
            raise TypeError("asset snapshot reference is invalid")
        return cast(
            "AssetStore",
            _SnapshotAssetStore(ref, object_store),
        )

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "asset store is not initialized",
            )


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


async def _single_object_chunk(value: bytes):
    yield value


async def _put_snapshot_object(
    object_store: ObjectStore,
    key: str,
    value: bytes,
) -> None:
    digest = hashlib.sha256(value).hexdigest()
    current = await object_store.stat(key)
    if current is not None:
        if current.digest != digest or current.size != len(value):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await read_object(
            object_store,
            key,
            expected_digest=digest,
            expected_size=len(value),
        )
        return
    await object_store.put(
        key,
        _single_object_chunk(value),
        expected_size=len(value),
        expected_digest=digest,
    )


def _snapshot_entry(info: AssetInfo, content_key: str) -> dict[str, JsonValue]:
    return {
        "key": {"kind": info.key.kind, "id": info.key.id},
        "source": {"root_id": info.root_id, "root_digest": info.root_digest},
        "entry_revision": info.revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "status": info.status.value,
        "modified_at": info.modified_at.isoformat(),
        "metadata": dict(info.metadata),
        "content": {
            "store_id": "snapshot",
            "key": content_key,
            "digest": info.etag,
            "size": info.size,
        },
    }


class _SnapshotAssetStore(AssetStore):
    def __init__(self, ref: ObjectRef, object_store: ObjectStore) -> None:
        self._ref = ref
        self._object_store = object_store
        self._entries: dict[AssetKey, AssetInfo] = {}
        self._values: dict[AssetKey, str] = {}
        self._revision: str | None = None
        self._ready = False
        self._closing = False
        self._closed = False

    @property
    def atomic_batch(self) -> bool:
        return False

    async def initialize(self) -> None:
        if self._closed or self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._ready:
            return
        payload = await read_object(
            self._object_store,
            self._ref.key,
            expected_digest=self._ref.digest,
            expected_size=self._ref.size,
        )
        try:
            manifest = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("kind") != "asset-snapshot"
            or manifest.get("format_version") != 1
            or not isinstance(manifest.get("entries"), list)
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        entries: dict[AssetKey, AssetInfo] = {}
        values: dict[AssetKey, str] = {}
        for raw in manifest["entries"]:
            info, content_key = _decode_snapshot_entry(raw)
            if info.key in entries:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await read_object(
                self._object_store,
                content_key,
                expected_digest=info.etag,
                expected_size=info.size,
            )
            entries[info.key] = info
            values[info.key] = content_key
        self._entries = entries
        self._values = values
        self._revision = str(manifest.get("captured_revision"))
        if not self._revision or self._revision == "None":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._ready = True

    async def close(self) -> None:
        self._ready = False
        self._closing = False
        self._closed = True

    async def current_revision(self) -> StorageRevision:
        self._ensure_ready()
        if self._revision is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return StorageRevision(self._revision)

    async def stat(self, key: AssetKey) -> AssetInfo | None:
        self._ensure_ready()
        return self._entries.get(key)

    async def get(self, key: AssetKey) -> bytes | None:
        self._ensure_ready()
        content_key = self._values.get(key)
        if content_key is None:
            return None
        info = self._entries[key]
        return await read_object(
            self._object_store,
            content_key,
            expected_digest=info.etag,
            expected_size=info.size,
        )

    async def get_many(self, keys: Sequence[AssetKey]) -> tuple[bytes | None, ...]:
        return tuple([value async for value in _snapshot_values(self, keys)])

    async def list_info(
        self,
        *,
        kind: str | None = None,
        prefix: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page[AssetInfo]:
        self._ensure_ready()
        limit = validate_page_limit(limit)
        values = tuple(
            info
            for info in sorted(
                self._entries.values(), key=lambda item: (item.key.kind, item.key.id)
            )
            if (kind is None or info.key.kind == kind)
            and (prefix is None or info.key.id.startswith(prefix))
        )
        start = _snapshot_cursor_start(
            cursor,
            self._ref.digest,
            kind,
            prefix,
            len(values),
        )
        selected = values[start : start + limit]
        next_cursor = (
            _snapshot_cursor(
                self._ref.digest,
                kind,
                prefix,
                start + len(selected),
            )
            if start + len(selected) < len(values)
            else None
        )
        return Page(selected, next_cursor)

    async def metadata_snapshot(self) -> tuple[AssetInfo, ...]:
        self._ensure_ready()
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda item: (item.key.kind, item.key.id),
            )
        )

    async def list_info_with_owners(
        self,
        *,
        kind: str | None = None,
        prefix: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page[StorageOwnedInfo[AssetInfo]]:
        page = await self.list_info(
            kind=kind,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
        )
        return Page(
            tuple(StorageOwnedInfo(info, "snapshot", False) for info in page.items),
            page.next_cursor,
        )

    async def write_states(
        self,
        keys: Sequence[AssetKey],
    ) -> Mapping[AssetKey, StorageWriteState[AssetInfo]]:
        self._ensure_ready()
        return {
            key: StorageWriteState(
                None
                if (info := self._entries.get(key)) is None
                else StorageOwnedInfo(info, "snapshot", False),
                info,
                False,
            )
            for key in dict.fromkeys(keys)
        }

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        info = await self.stat(key)
        if info is None:
            return ()
        return (
            VersionSummary(
                info.revision,
                info.etag,
                info.size,
                info.modified_at,
                info.status,
                info.metadata,
            ),
        )

    async def get_at_revision(
        self,
        key: AssetKey,
        revision: StorageEntryRevision,
    ) -> bytes | None:
        info = await self.stat(key)
        if info is None or info.revision != revision:
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        return await self.get(key)

    async def get_at_version(self, key: AssetKey, version: int) -> bytes | None:
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def batch_result(
        self,
        idempotency_key: str,
    ) -> "StorageBatchResult[AssetInfo, AssetKey] | None":
        del idempotency_key
        self._ensure_ready()
        raise AIError(ErrorCode.STORAGE_READ_ONLY)

    async def apply_batch(self, *args: object, **kwargs: object) -> object:
        raise AIError(ErrorCode.STORAGE_READ_ONLY)

    async def put(self, *args: object, **kwargs: object) -> object:
        raise AIError(ErrorCode.STORAGE_READ_ONLY)

    async def delete(self, *args: object, **kwargs: object) -> object:
        raise AIError(ErrorCode.STORAGE_READ_ONLY)

    async def reset(self, *args: object, **kwargs: object) -> object:
        raise AIError(ErrorCode.STORAGE_READ_ONLY)

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


def _snapshot_cursor(
    snapshot_digest: str,
    kind: str | None,
    prefix: str | None,
    start: int,
) -> str:
    payload = json.dumps(
        [snapshot_digest, kind, prefix, start],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _snapshot_cursor_start(
    cursor: str | None,
    snapshot_digest: str,
    kind: str | None,
    prefix: str | None,
    size: int,
) -> int:
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        )
        if (
            not isinstance(payload, list)
            or len(payload) != 4
            or payload[0] != snapshot_digest
            or payload[1] != kind
            or payload[2] != prefix
            or isinstance(payload[3], bool)
            or not isinstance(payload[3], int)
        ):
            raise ValueError
        start = payload[3]
    except (
        ValueError,
        TypeError,
        UnicodeEncodeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        binascii.Error,
    ):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    if start < 0 or start > size:
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID)
    return start


async def _snapshot_values(
    store: _SnapshotAssetStore,
    keys: Sequence[AssetKey],
):
    for key in keys:
        yield await store.get(key)


def _decode_snapshot_entry(raw: object) -> tuple[AssetInfo, str]:
    if not isinstance(raw, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    key_payload = raw.get("key")
    source = raw.get("source")
    content = raw.get("content")
    try:
        if (
            not isinstance(key_payload, Mapping)
            or not isinstance(source, Mapping)
            or not isinstance(content, Mapping)
        ):
            raise ValueError
        key = AssetKey(str(key_payload["kind"]), str(key_payload["id"]))
        etag = str(raw["etag"])
        size = int(raw["size"])
        info = AssetInfo(
            key=key,
            revision=StorageEntryRevision(int(raw["entry_revision"])),
            store_revision=StorageRevision(str(raw["store_revision"])),
            etag=etag,
            size=size,
            status=StorageEntryStatus(str(raw["status"])),
            root_id=str(source["root_id"]),
            root_digest=str(source["root_digest"]),
            modified_at=datetime.fromisoformat(str(raw["modified_at"])),
            metadata=cast(Mapping[str, JsonValue], raw.get("metadata", {})),
        )
        if (
            content.get("store_id") != "snapshot"
            or content.get("digest") != info.etag
            or content.get("size") != info.size
        ):
            raise ValueError
        return info, str(content["key"])
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _batch_request_digest(
    changes: Sequence[StorageChange[AssetKey, bytes]],
    expected_revision: StorageRevision | None,
) -> str:
    return canonical_sha256(
        {
            "expected_revision": (
                None if expected_revision is None else expected_revision.value
            ),
            "changes": [
                {
                    "operation": change.operation.value,
                    "kind": change.key.kind,
                    "id": change.key.id,
                    "value_digest": (
                        None
                        if change.value is None
                        else hashlib.sha256(bytes(change.value)).hexdigest()
                    ),
                    "value_size": None if change.value is None else len(change.value),
                    "expected_revision": (
                        None
                        if change.expected_revision is None
                        else change.expected_revision.value
                    ),
                    "metadata": dict(change.metadata),
                }
                for change in changes
            ],
        }
    )


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
        payload = json.loads(
            base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        )
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
