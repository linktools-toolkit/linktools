#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory and crash-recoverable file Asset backends."""

import asyncio
import base64
import binascii
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from linktools.core import environ

from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.files import read_bytes, read_json, write_bytes_atomic, write_json_atomic
from ..storage.model import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    StorageDeleteResult,
    StoragePutResult,
    StorageResetResult,
    VersionSummary,
)
from .model import AssetInfo, AssetKey, AssetRevision, AssetRoot, AssetStoreRevision
from .path import asset_path, file_root

_logger = environ.get_logger("ai.asset.files")
_EMPTY_ETAG = hashlib.sha256(b"").hexdigest()


class MemoryAssetBackend:
    def __init__(self, root: 'AssetRoot | None' = None, *, writable: bool = True) -> None:
        self._root = root or AssetRoot("memory:default", "memory", "memory", "memory")
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

    async def initialize(self) -> None:
        return None

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def head_revision(self) -> AssetStoreRevision:
        async with self._lock:
            return AssetStoreRevision(str(self._revision))

    async def load_metadata(self, after_revision: 'AssetStoreRevision | None') -> 'MetadataLoad[AssetKey, AssetInfo, AssetStoreRevision]':
        async with self._lock:
            current = AssetStoreRevision(str(self._revision))
            if after_revision is not None and after_revision == current:
                return MetadataLoad(MetadataLoadMode.PATCH, current, ())
            return MetadataLoad(
                MetadataLoadMode.REPLACE,
                current,
                tuple(MetadataChange(key, info) for key, (info, _) in self._entries.items()),
            )

    async def get(self, key: AssetKey) -> 'bytes | None':
        async with self._lock:
            entry = self._entries.get(key)
            return None if entry is None or entry[0].deleted else entry[1]

    async def get_many(self, keys: 'tuple[AssetKey, ...]') -> 'dict[AssetKey, bytes]':
        values: dict[AssetKey, bytes] = {}
        async with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and not entry[0].deleted:
                    values[key] = entry[1]
        return values

    async def stat(self, key: AssetKey) -> 'AssetInfo | None':
        async with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry[0]

    async def list_info(self, *, kind: 'str | None' = None) -> 'tuple[AssetInfo, ...]':
        async with self._lock:
            return tuple(
                info
                for info, _ in sorted(
                    self._entries.values(),
                    key=lambda item: (item[0].key.kind, item[0].key.id),
                )
                if kind is None or info.key.kind == kind
            )

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: 'AssetRevision | None' = None,
    ) -> 'StoragePutResult[AssetInfo, AssetRevision, AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            _check_entry_revision(previous, expected_entry_revision)
            etag = _etag(value)
            if previous is not None and not previous[0].deleted and previous[0].etag == etag:
                return StoragePutResult(previous[0], previous[0].entry_revision, previous[0].store_revision, False)
            info = self._next_info(key, value, previous, deleted=False)
            self._entries[key] = (info, value)
            self._versions.setdefault(key, []).append((info, value))
            return StoragePutResult(info, info.entry_revision, info.store_revision, True)

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: 'AssetRevision | None' = None,
    ) -> 'StorageDeleteResult[AssetKey, AssetRevision, AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            _check_entry_revision(previous, expected_entry_revision)
            if previous is None or previous[0].deleted:
                return StorageDeleteResult(key, False, None, AssetStoreRevision(str(self._revision)))
            info = self._next_info(key, b"", previous, deleted=True)
            self._entries[key] = (info, b"")
            self._versions.setdefault(key, []).append((info, b""))
            return StorageDeleteResult(key, True, info.entry_revision, info.store_revision)

    async def reset(self) -> 'StorageResetResult[AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            active = tuple(
                key for key, (info, _) in self._entries.items() if not info.deleted
            )
            for key in active:
                previous = self._entries[key]
                info = self._next_info(key, b"", previous, deleted=True)
                self._entries[key] = (info, b"")
                self._versions.setdefault(key, []).append((info, b""))
            count = len(active)
            return StorageResetResult(AssetStoreRevision(str(self._revision)), count)

    async def list_versions(self, key: AssetKey) -> 'tuple[VersionSummary[AssetRevision], ...]':
        async with self._lock:
            return tuple(
                VersionSummary(info.entry_revision, info.etag, info.size, info.modified_at, info.deleted)
                for info, _ in self._versions.get(key, ())
            )

    async def get_at_revision(self, key: AssetKey, entry_revision: AssetRevision) -> 'bytes | None':
        async with self._lock:
            for info, value in self._versions.get(key, ()):
                if info.entry_revision == entry_revision:
                    return None if info.deleted else value
            return None

    async def get_at_version(self, key: AssetKey, version: int) -> 'bytes | None':
        return await self.get_at_revision(key, AssetRevision(version))

    def _next_info(self, key: AssetKey, value: bytes, previous: 'tuple[AssetInfo, bytes] | None', *, deleted: bool) -> AssetInfo:
        self._revision += 1
        entry_revision = AssetRevision(0 if previous is None else previous[0].entry_revision.value + 1)
        return AssetInfo(
            key,
            entry_revision,
            AssetStoreRevision(str(self._revision)),
            _etag(value),
            len(value),
            deleted,
            self._root.root_id,
            self._root.digest,
            datetime.now(timezone.utc),
        )

    def _require_writable(self) -> None:
        if not self._writable:
            raise LinktoolsAIError(ErrorCode.STORAGE_READ_ONLY)


class FileAssetBackend:
    def __init__(self, root: 'AssetRoot | str', *, writable: bool = True) -> None:
        resolved = file_root(root) if isinstance(root, str) else root
        if resolved.scheme != "file":
            raise ValueError("FileAssetBackend requires a file root")
        self._root = resolved
        self._directory = Path(resolved.locator)
        self._transaction_directory = self._directory / ".txn"
        self._history_directory = self._directory / ".history"
        self._marker = self._directory / ".asset-revision"
        self._writable = writable
        self._entries: dict[AssetKey, tuple[AssetInfo, bytes]] = {}
        self._versions: dict[AssetKey, list[tuple[AssetInfo, bytes]]] = {}
        self._changes: dict[int, tuple[MetadataChange[AssetKey, AssetInfo], ...]] = {}
        self._revision = 0
        self._lock = asyncio.Lock()

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return self._writable

    async def initialize(self) -> None:
        async with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._transaction_directory.mkdir(parents=True, exist_ok=True)
            self._recover()
            self._load_history()
            self._load_entries()
        _logger.info("file asset backend initialized: root=%s revision=%s", self._directory, self._revision)

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def head_revision(self) -> AssetStoreRevision:
        async with self._lock:
            return AssetStoreRevision(str(self._revision))

    async def load_metadata(self, after_revision: 'AssetStoreRevision | None') -> 'MetadataLoad[AssetKey, AssetInfo, AssetStoreRevision]':
        async with self._lock:
            current = AssetStoreRevision(str(self._revision))
            if after_revision is None:
                return self._snapshot(current)
            previous = _revision_number(after_revision)
            if previous == self._revision:
                return MetadataLoad(MetadataLoadMode.PATCH, current, ())
            changes = tuple(change for revision in sorted(self._changes) if revision > previous for change in self._changes[revision])
            if changes and previous >= min(self._changes):
                return MetadataLoad(MetadataLoadMode.PATCH, current, changes)
            return self._snapshot(current)

    async def get(self, key: AssetKey) -> 'bytes | None':
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0].deleted:
                return None
            try:
                content = read_bytes(asset_path(self._root, key))
            except FileNotFoundError:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from None
            if _etag(content) != entry[0].etag:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return content

    async def get_many(self, keys: 'tuple[AssetKey, ...]') -> 'dict[AssetKey, bytes]':
        values: dict[AssetKey, bytes] = {}
        async with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or entry[0].deleted:
                    continue
                try:
                    content = read_bytes(asset_path(self._root, key))
                except FileNotFoundError:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from None
                if _etag(content) != entry[0].etag:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[key] = content
        return values

    async def stat(self, key: AssetKey) -> 'AssetInfo | None':
        async with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry[0]

    async def list_info(self, *, kind: 'str | None' = None) -> 'tuple[AssetInfo, ...]':
        async with self._lock:
            return tuple(
                info
                for info, _ in sorted(
                    self._entries.values(),
                    key=lambda item: (item[0].key.kind, item[0].key.id),
                )
                if kind is None or info.key.kind == kind
            )

    async def put(self, key: AssetKey, value: bytes, *, expected_entry_revision: 'AssetRevision | None' = None) -> 'StoragePutResult[AssetInfo, AssetRevision, AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            _check_entry_revision(previous, expected_entry_revision)
            if previous is not None and not previous[0].deleted and previous[0].etag == _etag(value):
                return StoragePutResult(previous[0], previous[0].entry_revision, previous[0].store_revision, False)
            info = self._next_info(key, value, previous, deleted=False)
            self._publish(info, value, previous=previous)
            self._record(info, value)
            return StoragePutResult(info, info.entry_revision, info.store_revision, True)

    async def delete(self, key: AssetKey, *, expected_entry_revision: 'AssetRevision | None' = None) -> 'StorageDeleteResult[AssetKey, AssetRevision, AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            previous = self._entries.get(key)
            _check_entry_revision(previous, expected_entry_revision)
            if previous is None or previous[0].deleted:
                return StorageDeleteResult(key, False, None, AssetStoreRevision(str(self._revision)))
            info = self._next_info(key, b"", previous, deleted=True)
            self._publish(info, b"", previous=previous)
            self._record(info, b"")
            return StorageDeleteResult(key, True, info.entry_revision, info.store_revision)

    async def reset(self) -> 'StorageResetResult[AssetStoreRevision]':
        async with self._lock:
            self._require_writable()
            active = tuple(key for key, (info, _) in self._entries.items() if not info.deleted)
            deleted = 0
            for key in active:
                previous = self._entries[key]
                info = self._next_info(key, b"", previous, deleted=True)
                self._publish(info, b"", previous=previous)
                self._record(info, b"")
                deleted += 1
            return StorageResetResult(AssetStoreRevision(str(self._revision)), deleted)

    async def list_versions(self, key: AssetKey) -> 'tuple[VersionSummary[AssetRevision], ...]':
        async with self._lock:
            return tuple(VersionSummary(info.entry_revision, info.etag, info.size, info.modified_at, info.deleted) for info, _ in self._versions.get(key, ()))

    async def get_at_revision(self, key: AssetKey, entry_revision: AssetRevision) -> 'bytes | None':
        async with self._lock:
            for info, value in self._versions.get(key, ()):
                if info.entry_revision == entry_revision:
                    return None if info.deleted else value
            return None

    async def get_at_version(self, key: AssetKey, version: int) -> 'bytes | None':
        return await self.get_at_revision(key, AssetRevision(version))

    def _snapshot(self, revision: AssetStoreRevision) -> 'MetadataLoad[AssetKey, AssetInfo, AssetStoreRevision]':
        return MetadataLoad(MetadataLoadMode.REPLACE, revision, tuple(MetadataChange(key, info) for key, (info, _) in self._entries.items()))

    def _next_info(self, key: AssetKey, value: bytes, previous: 'tuple[AssetInfo, bytes] | None', *, deleted: bool) -> AssetInfo:
        self._revision += 1
        return AssetInfo(key, AssetRevision(0 if previous is None else previous[0].entry_revision.value + 1), AssetStoreRevision(str(self._revision)), _etag(value), len(value), deleted, self._root.root_id, self._root.digest, datetime.now(timezone.utc))

    def _record(self, info: AssetInfo, value: bytes) -> None:
        self._entries[info.key] = (info, value)
        self._versions.setdefault(info.key, []).append((info, value))
        self._changes[self._revision] = (MetadataChange(info.key, info),)

    def _publish(
        self,
        info: AssetInfo,
        value: bytes,
        *,
        previous: 'tuple[AssetInfo, bytes] | None',
    ) -> None:
        operation_id = uuid.uuid4().hex
        content = asset_path(self._root, info.key)
        metadata = _metadata_path(content)
        temporary_content = content.with_name(f".{content.name}.{operation_id}.tmp")
        temporary_metadata = metadata.with_name(f".{metadata.name}.{operation_id}.tmp")
        history = self._history_path(info.key, info.entry_revision)
        temporary_history = history.with_name(f".{history.name}.{operation_id}.tmp")
        journal = self._transaction_directory / f"{operation_id}.json"
        payload = {
            "operation_id": operation_id,
            "asset_key": f"{info.key.kind}/{info.key.id}",
            "kind": info.key.kind,
            "id": info.key.id,
            "deleted": info.deleted,
            "old_content_digest": None if previous is None else previous[0].etag,
            "new_content_digest": info.etag,
            "old_store_revision": None if previous is None else previous[0].store_revision.value,
            "entry_revision": info.entry_revision.value,
            "store_revision": info.store_revision.value,
            "temporary_content": str(temporary_content),
            "temporary_metadata": str(temporary_metadata),
            "content": str(content),
            "metadata": str(metadata),
            "temporary_history": str(temporary_history),
            "history": str(history),
            "final_content": str(content),
            "final_metadata": str(metadata),
            "final_history": str(history),
            "revision": info.store_revision.value,
            "content_etag": info.etag,
        }
        write_json_atomic(journal, payload, fsync=True)
        write_bytes_atomic(temporary_content, value, fsync=True) if not info.deleted else None
        write_json_atomic(temporary_metadata, _info_json(info), fsync=True)
        write_json_atomic(temporary_history, _version_json(info, value), fsync=True)
        try:
            content.parent.mkdir(parents=True, exist_ok=True)
            if not info.deleted:
                os.replace(temporary_content, content)
            else:
                content.unlink(missing_ok=True)
            os.replace(temporary_metadata, metadata)
            history.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_history, history)
            write_json_atomic(self._marker, {"revision": int(info.store_revision.value)}, fsync=True)
            journal.unlink(missing_ok=True)
            self._transaction_directory_fdatasync()
        except BaseException:
            _logger.error("file asset publish interrupted: operation=%s", operation_id, exc_info=environ.debug)
            raise

    def _recover(self) -> None:
        for journal in sorted(self._transaction_directory.glob("*.json")):
            payload = read_json(journal)
            deleted = bool(payload.get("deleted", False))
            temporary_content = self._journal_path(payload, "temporary_content", journal)
            temporary_metadata = self._journal_path(payload, "temporary_metadata", journal)
            temporary_history = self._journal_path(payload, "temporary_history", journal)
            content = self._journal_path(payload, "content", journal)
            metadata = self._journal_path(payload, "metadata", journal)
            history = self._journal_path(payload, "history", journal)
            revision = _positive_int(payload.get("revision"), ErrorCode.ASSET_RECOVERY_REQUIRED)
            if not temporary_metadata.exists() and not metadata.exists():
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            if not deleted and temporary_content.exists():
                content_matches = False
                if content.exists():
                    content_matches = _etag(content.read_bytes()) == str(payload.get("content_etag"))
                if not content_matches:
                    content.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary_content, content)
            elif deleted:
                content.unlink(missing_ok=True)
            if temporary_metadata.exists():
                metadata_matches = False
                if metadata.exists():
                    existing = _info_from_json(read_json(metadata), self._root)
                    metadata_matches = existing.store_revision.value == str(revision)
                if not metadata_matches:
                    metadata.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary_metadata, metadata)
            if temporary_history.exists():
                history_matches = False
                if history.exists():
                    existing = _info_from_json(read_json(history), self._root)
                    history_matches = existing.store_revision.value == str(revision)
                if not history_matches:
                    history.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary_history, history)
            if not metadata.exists() or not history.exists() or (not deleted and not content.exists()):
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            recovered_info = _info_from_json(read_json(metadata), self._root)
            recovered_version = _info_from_json(read_json(history), self._root)
            if (
                content != asset_path(self._root, recovered_info.key)
                or metadata != _metadata_path(asset_path(self._root, recovered_info.key))
                or history != self._history_path(recovered_info.key, recovered_info.entry_revision)
                or recovered_version != recovered_info
                or recovered_info.store_revision.value != str(revision)
            ):
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            if not _journal_matches_info(payload, recovered_info, deleted):
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            if recovered_info.entry_revision.value:
                previous_history = self._history_path(
                    recovered_info.key,
                    AssetRevision(recovered_info.entry_revision.value - 1),
                )
                if not previous_history.exists():
                    raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
                previous_info = _info_from_json(read_json(previous_history), self._root)
                if (
                    payload.get("old_content_digest") != previous_info.etag
                    or payload.get("old_store_revision") != previous_info.store_revision.value
                ):
                    raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            if deleted:
                if content.exists():
                    raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            elif _etag(content.read_bytes()) != recovered_info.etag:
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
            marker_revision = 0
            if self._marker.exists():
                marker_revision = _positive_int(
                    read_json(self._marker).get("revision", 0),
                    ErrorCode.ASSET_RECOVERY_REQUIRED,
                )
            write_json_atomic(self._marker, {"revision": max(marker_revision, revision)}, fsync=True)
            temporary_content.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            temporary_history.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            self._transaction_directory_fdatasync()

    def _load_entries(self) -> None:
        self._entries.clear()
        self._revision = 0
        if self._marker.exists():
            marker = read_json(self._marker)
            self._revision = _positive_int(marker.get("revision", 0), ErrorCode.STORAGE_INTEGRITY_ERROR)
        for metadata in self._directory.rglob("*.meta"):
            if self._transaction_directory in metadata.parents:
                continue
            if self._history_directory in metadata.parents:
                continue
            raw = read_json(metadata)
            info = _info_from_json(raw, self._root)
            expected_metadata = _metadata_path(asset_path(self._root, info.key))
            if metadata.resolve() != expected_metadata.resolve():
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            content_path = asset_path(self._root, info.key)
            try:
                content = b"" if info.deleted else read_bytes(content_path)
            except FileNotFoundError as exc:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
            if not info.deleted and (_etag(content) != info.etag or len(content) != info.size):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if info.root_id != self._root.root_id or info.root_digest != self._root.digest:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if info.key in self._entries:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._entries[info.key] = (info, content)
            self._revision = max(self._revision, _revision_number(info.store_revision))
        self._load_legacy_entries()

    def _load_legacy_entries(self) -> None:
        root = self._directory.resolve()
        for path in sorted(self._directory.rglob("*")):
            if not path.is_file() or path.name.endswith(".meta"):
                continue
            if self._transaction_directory in path.parents or self._history_directory in path.parents:
                continue
            if path == self._marker:
                continue
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                continue
            try:
                key = AssetKey(relative.parts[0], "/".join(relative.parts[1:]))
                asset_path(self._root, key)
            except (ValueError, LinktoolsAIError):
                continue
            if key in self._entries:
                continue
            content = read_bytes(path)
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            info = AssetInfo(
                key,
                AssetRevision(0),
                AssetStoreRevision(str(self._revision)),
                _etag(content),
                len(content),
                False,
                self._root.root_id,
                self._root.digest,
                timestamp,
            )
            self._entries[key] = (info, content)
            self._versions.setdefault(key, []).append((info, content))

    def _load_history(self) -> None:
        self._versions.clear()
        if not self._history_directory.exists():
            return
        for history in sorted(self._history_directory.rglob("*.json")):
            raw = read_json(history)
            info = _info_from_json(raw, self._root)
            expected = self._history_path(info.key, info.entry_revision)
            if history.resolve() != expected.resolve():
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            encoded = raw.get("content")
            if not isinstance(encoded, str):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
            if info.deleted:
                content = b""
            elif _etag(content) != info.etag or len(content) != info.size:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            versions = self._versions.setdefault(info.key, [])
            if any(existing.entry_revision == info.entry_revision for existing, _ in versions):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            versions.append((info, content))
        for versions in self._versions.values():
            versions.sort(key=lambda item: item[0].entry_revision.value)

    def _history_path(self, key: AssetKey, revision: AssetRevision) -> Path:
        asset_path(self._root, key)
        return self._history_directory / key.kind / key.id / f"{revision.value}.json"

    def _journal_path(self, payload: 'dict[str, str | int | bool | None]', name: str, journal: Path) -> Path:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
        candidate = Path(value).resolve()
        root = self._directory.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED) from exc
        if name.startswith("temporary_"):
            if candidate.parent == self._transaction_directory.resolve() or not candidate.name.startswith(".") or not candidate.name.endswith(".tmp"):
                raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
        elif name in {"content", "metadata"} and candidate.parent == self._transaction_directory.resolve():
            raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
        if journal.resolve().parent != self._transaction_directory.resolve():
            raise LinktoolsAIError(ErrorCode.ASSET_RECOVERY_REQUIRED)
        return candidate

    def _transaction_directory_fdatasync(self) -> None:
        descriptor = os.open(self._transaction_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _require_writable(self) -> None:
        if not self._writable:
            raise LinktoolsAIError(ErrorCode.STORAGE_READ_ONLY)


def _check_entry_revision(previous: 'tuple[AssetInfo, bytes] | None', expected: 'AssetRevision | None') -> None:
    if expected is not None and (previous is None or previous[0].entry_revision != expected):
        raise LinktoolsAIError(ErrorCode.ASSET_REVISION_CONFLICT)


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _revision_number(revision: AssetStoreRevision) -> int:
    try:
        return int(revision.value)
    except ValueError as exc:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc


def _info_json(info: AssetInfo) -> 'dict[str, str | int | bool | None]':
    return {
        "kind": info.key.kind,
        "id": info.key.id,
        "entry_revision": info.entry_revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "deleted": info.deleted,
        "root_id": info.root_id,
        "root_digest": info.root_digest,
        "modified_at": info.modified_at.isoformat(),
    }


def _info_from_json(value: 'dict[str, str | int | bool | None]', root: AssetRoot) -> AssetInfo:
    required = ("kind", "id", "entry_revision", "store_revision", "etag", "size", "deleted", "root_id", "root_digest", "modified_at")
    if any(key not in value for key in required):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    key = AssetKey(str(value["kind"]), str(value["id"]))
    asset_path(root, key)
    entry_revision = _positive_int(value["entry_revision"], ErrorCode.STORAGE_INTEGRITY_ERROR)
    store_revision = _positive_int(value["store_revision"], ErrorCode.STORAGE_INTEGRITY_ERROR)
    etag = str(value["etag"])
    if len(etag) != 64 or any(character not in "0123456789abcdef" for character in etag):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(value["deleted"], bool):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value["deleted"] and (etag != _EMPTY_ETAG or int(str(value["size"])) != 0):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    modified_at = datetime.fromisoformat(str(value["modified_at"]))
    if modified_at.tzinfo is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AssetInfo(
        key,
        AssetRevision(entry_revision),
        AssetStoreRevision(str(store_revision)),
        etag,
        _positive_int(value["size"], ErrorCode.STORAGE_INTEGRITY_ERROR),
        bool(value["deleted"]),
        str(value["root_id"]),
        str(value["root_digest"]),
        modified_at,
    )


def _version_json(info: AssetInfo, value: bytes) -> 'dict[str, str | int | bool | None]':
    result = _info_json(info)
    result["content"] = base64.b64encode(value).decode("ascii")
    return result


def _journal_matches_info(
    payload: 'dict[str, str | int | bool | None]',
    info: AssetInfo,
    deleted: bool,
) -> bool:
    asset_key = payload.get("asset_key")
    if asset_key != f"{info.key.kind}/{info.key.id}":
        return False
    if payload.get("kind") != info.key.kind or payload.get("id") != info.key.id:
        return False
    if payload.get("deleted") is not deleted:
        return False
    if payload.get("new_content_digest") != info.etag or payload.get("content_etag") != info.etag:
        return False
    if payload.get("entry_revision") != info.entry_revision.value:
        return False
    if payload.get("store_revision") != info.store_revision.value or payload.get("revision") != info.store_revision.value:
        return False
    old_content_digest = payload.get("old_content_digest")
    old_store_revision = payload.get("old_store_revision")
    if info.entry_revision.value == 0:
        return old_content_digest is None and old_store_revision is None
    if not isinstance(old_content_digest, str) or not isinstance(old_store_revision, str):
        return False
    try:
        return int(old_store_revision) < int(info.store_revision.value)
    except ValueError:
        return False


def _positive_int(value: 'str | int | bool | None', code: ErrorCode) -> int:
    if isinstance(value, bool):
        raise LinktoolsAIError(code)
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise LinktoolsAIError(code) from exc
    if result < 0:
        raise LinktoolsAIError(code)
    return result


def _metadata_path(content: Path) -> Path:
    return content.with_name(f"{content.name}.meta")


__all__ = ["FileAssetBackend", "MemoryAssetBackend"]
