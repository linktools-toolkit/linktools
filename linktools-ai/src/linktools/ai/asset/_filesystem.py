#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation-two filesystem Asset backend."""

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ..storage import (
    FilesystemMutationLock,
    FilesystemObjectStore,
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    ObjectStore,
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
    read_object,
    write_json_atomic,
)
from ._domain import AssetInfo, AssetKey, AssetRoot
from ._object import AssetObjectKeyFactory

_logger = environ.get_logger("ai.asset.filesystem")
_GENERATION = 2
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


class FilesystemAssetBackend:
    """Persist Asset metadata while content lives in a generic ObjectStore."""

    def __init__(
        self,
        root: AssetRoot | str | Path,
        *,
        writable: bool = True,
        object_store: ObjectStore | None = None,
    ) -> None:
        resolved = filesystem_root(str(root)) if isinstance(root, (str, Path)) else root
        if resolved.scheme != "file":
            raise ValueError("FilesystemAssetBackend requires a filesystem root")
        self._root = resolved
        self._directory = Path(resolved.locator)
        self._manifest_path = self._directory / "manifest.json"
        self._state_path = self._directory / "state.json"
        self._lock_path = self._directory / "asset.lock"
        self._writable = writable
        self._object_store = object_store if object_store is not None else FilesystemObjectStore(self._directory / "objects")
        self._object_keys = AssetObjectKeyFactory(resolved.locator)
        self._entries: dict[AssetKey, AssetInfo] = {}
        self._versions: dict[AssetKey, list[AssetInfo]] = {}
        self._revision = 0
        self._process_lock = asyncio.Lock()
        self._ready = False

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
        try:
            async with self._process_lock, FilesystemMutationLock(self._lock_path):
                self._directory.mkdir(parents=True, exist_ok=True)
                self._validate_or_write_manifest()
                if self._state_path.is_file():
                    await self._load_state()
                else:
                    self._revision = 0
                    self._entries.clear()
                    self._versions.clear()
                    await self._write_state()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        self._ready = True
        _logger.debug("filesystem Asset backend initialized: root=%s revision=%s", self._directory, self._revision)

    async def close(self) -> None:
        self._ready = False

    async def head_revision(self) -> StorageRevision:
        await self._ensure_ready()
        await self._reload()
        return StorageRevision(str(self._revision))

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        await self._ensure_ready()
        await self._reload()
        current = StorageRevision(str(self._revision))
        if after_revision == current:
            return MetadataLoad(MetadataLoadMode.PATCH, current, ())
        return MetadataLoad(
            MetadataLoadMode.REPLACE,
            current,
            tuple(MetadataChange(key, info) for key, info in self._entries.items()),
        )

    async def get(self, key: AssetKey) -> bytes | None:
        await self._ensure_ready()
        await self._reload()
        info = self._entries.get(key)
        if info is None or info.status is not StorageEntryStatus.NORMAL:
            return None
        return await self._read_info(info)

    async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
        await self._ensure_ready()
        await self._reload()
        result: dict[AssetKey, bytes] = {}
        for key in keys:
            info = self._entries.get(key)
            if info is not None and info.status is StorageEntryStatus.NORMAL:
                result[key] = await self._read_info(info)
        return result

    async def stat(self, key: AssetKey) -> AssetInfo | None:
        await self._ensure_ready()
        await self._reload()
        return self._entries.get(key)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_revision: StorageEntryRevision | None = None,
    ) -> StoragePutResult[AssetInfo]:
        return await self._mutate(StorageOperation.PUT, key, bytes(value), expected_revision=expected_revision)

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        return await self._mutate(StorageOperation.DELETE, key, None, expected_revision=expected_revision)

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
    ) -> StorageResetResult[AssetKey]:
        return await self._mutate(StorageOperation.RESET, key, None, expected_revision=expected_revision)

    async def apply_batch(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        *,
        expected_revision: StorageRevision | None = None,
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        await self._ensure_ready()
        async with self._process_lock, FilesystemMutationLock(self._lock_path):
            await self._load_state()
            self._require_writable()
            if expected_revision is not None and expected_revision != StorageRevision(str(self._revision)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._validate_batch(changes)
            prepared = {
                change.key: bytes(change.value or b"")
                for change in changes
                if change.operation is StorageOperation.PUT
            }
            for value in prepared.values():
                await self._put_content(value)
            previous = {change.key: self._entries.get(change.key) for change in changes}
            for change in changes:
                _check_entry_revision(previous[change.key], change.expected_revision)
            mutates = tuple(_mutates(change.operation, previous[change.key], prepared.get(change.key, b"")) for change in changes)
            next_revision = self._revision + (1 if any(mutates) else 0)
            before = self._snapshot()
            results: list[StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]] = []
            for change, mutates_change in zip(changes, mutates, strict=True):
                current = previous[change.key]
                if not mutates_change:
                    results.append(_result(change.operation, change.key, current, StorageRevision(str(next_revision)), changed=False))
                    continue
                info = _next_info(self._root, change.key, prepared.get(change.key, b""), current, change.operation, next_revision)
                self._record(info)
                results.append(_result(change.operation, change.key, info, StorageRevision(str(next_revision)), changed=True))
            self._revision = next_revision
            try:
                await self._write_state()
            except BaseException as error:
                self._restore(before)
                _raise_filesystem_error(error)
            _logger.debug("filesystem Asset batch committed: root=%s revision=%s changes=%s", self._directory, self._revision, len(changes))
            return StorageBatchResult(StorageRevision(str(self._revision)), True, tuple(results))

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        await self._ensure_ready()
        await self._reload()
        return tuple(
            VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status)
            for info in self._versions.get(key, ())
        )

    async def get_at_revision(self, key: AssetKey, entry_revision: StorageEntryRevision) -> bytes | None:
        await self._ensure_ready()
        await self._reload()
        info = next((item for item in self._versions.get(key, ()) if item.revision == entry_revision), None)
        if info is None or info.status is not StorageEntryStatus.NORMAL:
            return None
        return await self._read_info(info)

    async def get_at_version(self, key: AssetKey, version: int) -> bytes | None:
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def _mutate(
        self,
        operation: StorageOperation,
        key: AssetKey,
        value: bytes | None,
        *,
        expected_revision: StorageEntryRevision | None,
    ) -> object:
        await self._ensure_ready()
        async with self._process_lock, FilesystemMutationLock(self._lock_path):
            await self._load_state()
            self._require_writable()
            current = self._entries.get(key)
            _check_entry_revision(current, expected_revision)
            content = value or b""
            if not _mutates(operation, current, content):
                return _result(operation, key, current, StorageRevision(str(self._revision)), changed=False)
            if operation is StorageOperation.PUT:
                await self._put_content(content)
            before = self._snapshot()
            next_revision = self._revision + 1
            info = _next_info(self._root, key, content, current, operation, next_revision)
            self._record(info)
            self._revision = next_revision
            try:
                await self._write_state()
            except BaseException as error:
                self._restore(before)
                _raise_filesystem_error(error)
            _logger.debug("filesystem Asset mutation committed: root=%s key=%s revision=%s", self._directory, key.id, self._revision)
            return _result(operation, key, info, StorageRevision(str(self._revision)), changed=True)

    async def _put_content(self, value: bytes) -> None:
        digest = _etag(value)
        await self._object_store.put(self._object_keys.key(digest), _one(value), expected_size=len(value), expected_digest=digest)

    async def _read_info(self, info: AssetInfo) -> bytes:
        object_key = self._object_keys.key(info.etag)
        return await read_object(self._object_store, object_key, expected_digest=info.etag, expected_size=info.size)

    async def _reload(self) -> None:
        async with FilesystemMutationLock(self._lock_path):
            await self._load_state()

    async def _load_state(self) -> None:
        if not self._state_path.is_file():
            self._revision = 0
            self._entries.clear()
            self._versions.clear()
            return
        try:
            raw = json.loads(await asyncio.to_thread(self._state_path.read_text, encoding="utf-8"))
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if not isinstance(raw, dict):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        revision = raw.get("store_revision")
        entries = raw.get("entries")
        versions = raw.get("versions")
        if not isinstance(revision, int) or revision < 0 or not isinstance(entries, list) or not isinstance(versions, list):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        loaded_versions: dict[AssetKey, list[AssetInfo]] = {}
        for item in versions:
            info = _info_from_json(item, self._root, self._object_store.store_id)
            loaded_versions.setdefault(info.key, []).append(info)
        loaded_entries: dict[AssetKey, AssetInfo] = {}
        for item in entries:
            info = _info_from_json(item, self._root, self._object_store.store_id)
            if info.key in loaded_entries:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            loaded_entries[info.key] = info
        for key, history in loaded_versions.items():
            if [item.revision.value for item in history] != list(range(1, len(history) + 1)):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if key not in loaded_entries or loaded_entries[key] != history[-1]:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            history_revisions = [int(item.store_revision.value) for item in history]
            if any(value < 1 or value > revision for value in history_revisions) or history_revisions != sorted(history_revisions) or len(set(history_revisions)) != len(history_revisions):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if set(loaded_entries) != set(loaded_versions):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if loaded_entries and any(int(info.store_revision.value) < 1 or int(info.store_revision.value) > revision for info in loaded_entries.values()):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if loaded_entries and max(int(info.store_revision.value) for info in loaded_entries.values()) > revision:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        self._revision = revision
        self._entries = loaded_entries
        self._versions = loaded_versions
        checked: set[str] = set()
        for info in (*loaded_entries.values(), *(item for history in loaded_versions.values() for item in history)):
            if info.status is not StorageEntryStatus.NORMAL or info.etag in checked:
                continue
            checked.add(info.etag)
            stat = await self._object_store.stat(self._object_keys.key(info.etag))
            if stat is None or stat.digest != info.etag or stat.size != info.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _write_state(self) -> None:
        raw = {
            "store_revision": self._revision,
            "entries": [self._info_to_json(info) for info in self._entries.values()],
            "versions": [self._info_to_json(info) for history in self._versions.values() for info in history],
        }
        await asyncio.to_thread(write_json_atomic, self._state_path, raw, fsync=True)

    def _info_to_json(self, info: AssetInfo) -> dict[str, object]:
        normal = info.status is StorageEntryStatus.NORMAL
        return {
            "kind": info.key.kind,
            "id": info.key.id,
            "revision": info.revision.value,
            "store_revision": info.store_revision.value,
            "etag": info.etag,
            "size": info.size,
            "status": info.status.value,
            "root_id": info.root_id,
            "root_digest": info.root_digest,
            "modified_at": info.modified_at.astimezone(timezone.utc).isoformat(),
            "object_store_id": self._object_store.store_id if normal else None,
            "object_key": self._object_keys.key(info.etag) if normal else None,
        }

    def _validate_or_write_manifest(self) -> None:
        expected = {"format": "linktools-ai-asset", "generation": _GENERATION, "root_id": self._root.root_id, "root_digest": self._root.digest}
        if not self._manifest_path.is_file():
            if self._state_path.exists():
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            try:
                write_json_atomic(self._manifest_path, expected, fsync=True)
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            return
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
        if not isinstance(value, dict) or value.get("format") != expected["format"] or value.get("generation") != _GENERATION:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if value.get("root_id") != self._root.root_id or value.get("root_digest") != self._root.digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _record(self, info: AssetInfo) -> None:
        self._entries[info.key] = info
        self._versions.setdefault(info.key, []).append(info)

    def _snapshot(self) -> tuple[int, dict[AssetKey, AssetInfo], dict[AssetKey, list[AssetInfo]]]:
        return self._revision, dict(self._entries), {key: list(values) for key, values in self._versions.items()}

    def _restore(self, snapshot: tuple[int, dict[AssetKey, AssetInfo], dict[AssetKey, list[AssetInfo]]]) -> None:
        self._revision, self._entries, self._versions = snapshot

    def _validate_batch(self, changes: Sequence[StorageChange[AssetKey, bytes]]) -> None:
        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)

    def _require_writable(self) -> None:
        if not self._writable:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "asset store is not initialized")


async def _one(value: bytes):
    yield value


def _check_entry_revision(current: AssetInfo | None, expected: StorageEntryRevision | None) -> None:
    if expected is not None and (current is None or current.revision != expected):
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _mutates(operation: StorageOperation, current: AssetInfo | None, value: bytes) -> bool:
    if operation is StorageOperation.DELETE:
        return current is not None and current.status is not StorageEntryStatus.DELETED
    if operation is StorageOperation.RESET:
        return current is not None and current.status is not StorageEntryStatus.RESET
    return current is None or current.status is not StorageEntryStatus.NORMAL or current.etag != _etag(value)


def _next_info(root: AssetRoot, key: AssetKey, value: bytes, previous: AssetInfo | None, operation: StorageOperation, store_revision: int) -> AssetInfo:
    status = StorageEntryStatus.NORMAL if operation is StorageOperation.PUT else StorageEntryStatus.DELETED if operation is StorageOperation.DELETE else StorageEntryStatus.RESET
    content = value if status is StorageEntryStatus.NORMAL else b""
    entry_revision = 1 if previous is None else previous.revision.value + 1
    return AssetInfo(key, StorageEntryRevision(entry_revision), StorageRevision(str(store_revision)), _etag(content), len(content), status, root.root_id, root.digest, datetime.now(timezone.utc))


def _result(operation: StorageOperation, key: AssetKey, info: AssetInfo | None, revision: StorageRevision, *, changed: bool) -> object:
    if operation is StorageOperation.PUT:
        if info is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StoragePutResult(info, info.revision, revision, changed)
    if operation is StorageOperation.DELETE:
        return StorageDeleteResult(key, changed, info.revision if changed and info is not None else None, revision)
    return StorageResetResult(key, changed, revision)


def _info_from_json(raw: object, root: AssetRoot, store_id: str) -> AssetInfo:
    if not isinstance(raw, Mapping):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    try:
        if raw.get("root_id") != root.root_id or raw.get("root_digest") != root.digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        persisted_store_id = raw.get("object_store_id")
        persisted_object_key = raw.get("object_key")
        normal = raw.get("status") == StorageEntryStatus.NORMAL.value
        if normal and persisted_store_id != store_id:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if normal != all(value is not None for value in (persisted_store_id, persisted_object_key)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if not normal and any(value is not None for value in (persisted_store_id, persisted_object_key)):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        key = AssetKey(str(raw["kind"]), str(raw["id"]))
        if normal and persisted_object_key != AssetObjectKeyFactory(root.locator).key(str(raw["etag"])):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return AssetInfo(
            key,
            StorageEntryRevision(int(raw["revision"])),
            StorageRevision(str(raw["store_revision"])),
            str(raw["etag"]),
            int(raw["size"]),
            StorageEntryStatus(str(raw["status"])),
            root.root_id,
            root.digest,
            _utc(raw["modified_at"]),
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error


def _utc(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raise_filesystem_error(error: BaseException) -> None:
    if isinstance(error, AIError):
        raise error
    if isinstance(error, OSError):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    raise error


def filesystem_root(locator: str) -> AssetRoot:
    path = Path(locator).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return AssetRoot(f"file:{digest[:16]}", "file", str(path), digest)


__all__ = ["FilesystemAssetBackend", "filesystem_root"]
