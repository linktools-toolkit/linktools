"""Generation-two filesystem Asset backend."""

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from linktools.core import environ

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..storage import (
    FilesystemJournal,
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
    normalize_storage_metadata,
    read_object,
    sync_directory,
    write_json_atomic,
)
from ._domain import AssetInfo, AssetKey, AssetRoot
from ._object import AssetObjectKeyFactory

_logger = environ.get_logger("ai.asset.filesystem")
_GENERATION = 1
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
        self._generation_path = self._directory / "generation"
        self._head_path = self._directory / "head.json"
        self._entries_path = self._directory / "entries"
        self._history_path = self._directory / "history"
        self._lock_path = self._directory / "asset.lock"
        self._journal = FilesystemJournal(
            self._directory,
            error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED,
        )
        self._writable = writable
        self._object_store = (
            object_store if object_store is not None else FilesystemObjectStore(self._directory / "objects")
        )
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
            async with self._process_lock:
                if not self._directory.exists():
                    self._set_empty_state()
                else:
                    if not self._directory.is_dir():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    entries = tuple(self._directory.iterdir())
                    if not entries:
                        self._set_empty_state()
                    elif all(path.name == "asset.lock" for path in entries):
                        async with FilesystemMutationLock(self._lock_path):
                            entries = tuple(self._directory.iterdir())
                            if all(path.name == "asset.lock" for path in entries):
                                self._set_empty_state()
                            else:
                                await self._load_existing_root()
                    else:
                        await self._load_existing_root()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        self._ready = True
        _logger.debug("filesystem Asset backend initialized: root=%s revision=%s", self._directory, self._revision)

    async def _load_existing_root(self) -> None:
        self._validate_existing_root()
        if (self._directory / ".txn").exists() and not self._writable:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if (self._directory / ".txn").exists():
            async with FilesystemMutationLock(self._lock_path):
                await self._recover()
        await self._load_state()

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
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StoragePutResult[AssetInfo]:
        return await self._mutate(
            StorageOperation.PUT,
            key,
            bytes(value),
            expected_revision=expected_revision,
            metadata=metadata,
        )

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        return await self._mutate(
            StorageOperation.DELETE,
            key,
            None,
            expected_revision=expected_revision,
            metadata=metadata,
        )

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageResetResult[AssetKey]:
        return await self._mutate(
            StorageOperation.RESET,
            key,
            None,
            expected_revision=expected_revision,
            metadata=metadata,
        )

    async def apply_batch(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        *,
        expected_revision: StorageRevision | None = None,
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        await self._ensure_ready()
        async with self._process_lock:
            await self._reload()
            self._require_writable()
            if expected_revision is not None and expected_revision != StorageRevision(str(self._revision)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._validate_batch(changes)
            prepared = {
                change.key: bytes(change.value or b"") for change in changes if change.operation is StorageOperation.PUT
            }
            previous = {change.key: self._entries.get(change.key) for change in changes}
            for change in changes:
                _check_entry_revision(previous[change.key], change.expected_revision)
            mutates = tuple(
                _mutates(change.operation, previous[change.key], prepared.get(change.key, b"")) for change in changes
            )
            if not any(mutates):
                results = tuple(
                    _result(
                        change.operation,
                        change.key,
                        previous[change.key],
                        StorageRevision(str(self._revision)),
                        changed=False,
                    )
                    for change in changes
                )
                return StorageBatchResult(StorageRevision(str(self._revision)), True, results)
            async with FilesystemMutationLock(self._lock_path):
                self._provision()
                await self._recover()
                await self._load_state()
                self._require_writable()
                if (
                    expected_revision is not None
                    and expected_revision != StorageRevision(str(self._revision))
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                previous = {change.key: self._entries.get(change.key) for change in changes}
                for change in changes:
                    _check_entry_revision(previous[change.key], change.expected_revision)
                mutates = tuple(
                    _mutates(change.operation, previous[change.key], prepared.get(change.key, b""))
                    for change in changes
                )
                if not any(mutates):
                    results = tuple(
                        _result(
                            change.operation,
                            change.key,
                            previous[change.key],
                            StorageRevision(str(self._revision)),
                            changed=False,
                        )
                        for change in changes
                    )
                    return StorageBatchResult(StorageRevision(str(self._revision)), True, results)
                put_changes = (
                    (change, prepared[change.key])
                    for change in changes
                    if change.operation is StorageOperation.PUT
                )
                for change, value in put_changes:
                    if _mutates(change.operation, previous[change.key], value):
                        await self._put_content(value)
                next_revision = self._revision + 1
                before = self._snapshot()
                results: list[
                    StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]
                ] = []
                for change, mutates_change in zip(changes, mutates, strict=True):
                    current = previous[change.key]
                    if not mutates_change:
                        results.append(
                            _result(
                                change.operation,
                                change.key,
                                current,
                                StorageRevision(str(next_revision)),
                                changed=False,
                            )
                        )
                        continue
                    info = _next_info(
                        self._root,
                        change.key,
                        prepared.get(change.key, b""),
                        current,
                        change.operation,
                        next_revision,
                        change.metadata,
                    )
                    self._record(info)
                    results.append(
                        _result(
                            change.operation,
                            change.key,
                            info,
                            StorageRevision(str(next_revision)),
                            changed=True,
                        )
                    )
                self._revision = next_revision
                try:
                    await self._write_state()
                except BaseException as error:  # noqa: BLE001
                    self._restore(before)
                    _raise_filesystem_error(error)
                _logger.debug(
                    "filesystem Asset batch committed: root=%s revision=%s changes=%s",
                    self._directory,
                    self._revision,
                    len(changes),
                )
                return StorageBatchResult(StorageRevision(str(self._revision)), True, tuple(results))

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        await self._ensure_ready()
        await self._reload()
        return tuple(
            VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status, info.metadata)
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
        metadata: Mapping[str, JsonValue] | None,
    ) -> object:
        await self._ensure_ready()
        async with self._process_lock:
            await self._reload()
            self._require_writable()
            current = self._entries.get(key)
            _check_entry_revision(current, expected_revision)
            content = value or b""
            if not _mutates(operation, current, content):
                return _result(operation, key, current, StorageRevision(str(self._revision)), changed=False)
            async with FilesystemMutationLock(self._lock_path):
                self._provision()
                await self._recover()
                await self._load_state()
                current = self._entries.get(key)
                _check_entry_revision(current, expected_revision)
                content = value or b""
                if not _mutates(operation, current, content):
                    return _result(
                        operation,
                        key,
                        current,
                        StorageRevision(str(self._revision)),
                        changed=False,
                    )
                if operation is StorageOperation.PUT:
                    await self._put_content(content)
                before = self._snapshot()
                next_revision = self._revision + 1
                info = _next_info(
                    self._root,
                    key,
                    content,
                    current,
                    operation,
                    next_revision,
                    metadata,
                )
                self._record(info)
                self._revision = next_revision
                try:
                    await self._write_state()
                except BaseException as error:  # noqa: BLE001
                    self._restore(before)
                    _raise_filesystem_error(error)
                _logger.debug(
                    "filesystem Asset mutation committed: root=%s key=%s revision=%s",
                    self._directory,
                    key.id,
                    self._revision,
                )
                return _result(operation, key, info, StorageRevision(str(self._revision)), changed=True)

    async def _put_content(self, value: bytes) -> None:
        digest = _etag(value)
        await self._object_store.put(
            self._object_keys.key(digest), _one(value), expected_size=len(value), expected_digest=digest
        )

    async def _read_info(self, info: AssetInfo) -> bytes:
        object_key = self._object_keys.key(info.etag)
        return await read_object(self._object_store, object_key, expected_digest=info.etag, expected_size=info.size)

    async def _reload(self) -> None:
        if not self._directory.exists() or not any(self._directory.iterdir()):
            self._set_empty_state()
            return
        if (self._directory / ".txn").exists():
            if not self._writable:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            async with FilesystemMutationLock(self._lock_path):
                await self._recover()
        await self._load_state()

    async def validate_integrity(self) -> None:
        await self._ensure_ready()
        await self._reload()
        _logger.info(
            "filesystem Asset integrity validated: root=%s revision=%s",
            self._root.digest[:16],
            self._revision,
        )

    async def _load_state(self) -> None:
        if not self._directory.exists() or not any(self._directory.iterdir()):
            self._set_empty_state()
            return
        if not self._head_path.is_file():
            if any(self._entries_path.rglob("*.json")) or any(self._history_path.rglob("*.json")):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if _read_generation(self._generation_path) != 0:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            self._revision = 0
            self._entries.clear()
            self._versions.clear()
            return
        try:
            raw = json.loads(await asyncio.to_thread(self._head_path.read_text, encoding="utf-8"))
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if not isinstance(raw, dict):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        revision = raw.get("store_revision")
        if not isinstance(revision, int) or revision < 0:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if revision != _read_generation(self._generation_path):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        loaded_versions: dict[AssetKey, list[AssetInfo]] = {}
        for path in self._history_path.glob("*/*/*.json"):
            info = _info_from_json(
                await _read_asset_json(path),
                self._root,
                self._object_store.store_id,
            )
            if path != self._history_file(info):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            loaded_versions.setdefault(info.key, []).append(info)
        loaded_entries: dict[AssetKey, AssetInfo] = {}
        for path in self._entries_path.glob("*/*.json"):
            info = _info_from_json(
                await _read_asset_json(path),
                self._root,
                self._object_store.store_id,
            )
            if path != self._entry_file(info.key):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if info.key in loaded_entries:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            loaded_entries[info.key] = info
        for key, history in loaded_versions.items():
            history.sort(key=lambda item: item.revision.value)
            if [item.revision.value for item in history] != list(range(1, len(history) + 1)):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if key not in loaded_entries or loaded_entries[key] != history[-1]:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            history_revisions = [int(item.store_revision.value) for item in history]
            if (
                any(value < 1 or value > revision for value in history_revisions)
                or history_revisions != sorted(history_revisions)
                or len(set(history_revisions)) != len(history_revisions)
            ):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if set(loaded_entries) != set(loaded_versions):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if revision == 0 and loaded_versions:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if revision > 0 and (
            not loaded_versions
            or max(int(info.store_revision.value) for history in loaded_versions.values() for info in history)
            != revision
        ):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if loaded_entries and any(
            int(info.store_revision.value) < 1 or int(info.store_revision.value) > revision
            for info in loaded_entries.values()
        ):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if loaded_entries and max(int(info.store_revision.value) for info in loaded_entries.values()) > revision:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        self._revision = revision
        self._entries = loaded_entries
        self._versions = loaded_versions
        expected_files = set(self._serialized_state())
        actual_files = {
            path.relative_to(self._directory).as_posix()
            for directory in (self._entries_path, self._history_path)
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files - {"head.json"}:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        checked: set[str] = set()
        for info in (*loaded_entries.values(), *(item for history in loaded_versions.values() for item in history)):
            object_key = self._object_keys.key(info.etag)
            if info.status is not StorageEntryStatus.NORMAL or object_key in checked:
                continue
            checked.add(object_key)
            stat = await self._object_store.stat(object_key)
            if stat is None or stat.digest != info.etag or stat.size != info.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _write_state(self) -> None:
        desired = self._serialized_state()
        previous = self._existing_state()
        if desired == previous:
            return
        base = _read_generation(self._generation_path)
        if self._revision != base + 1:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        plan = self._journal.stage(
            {path: value for path, value in desired.items() if previous.get(path) != value},
            previous.keys() - desired.keys(),
            base_generation=base,
            target_generation=self._revision,
        )
        self._journal.publish(plan)
        _write_text(self._generation_path, str(self._revision))
        sync_directory(self._directory)
        self._journal.complete()

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
            "metadata": dict(info.metadata),
        }

    def _set_empty_state(self) -> None:
        self._revision = 0
        self._entries.clear()
        self._versions.clear()

    def _expected_manifest(self) -> dict[str, object]:
        return {
            "format": "linktools-ai-asset",
            "generation": _GENERATION,
            "root_id": self._root.root_id,
            "root_digest": self._root.digest,
        }

    def _validate_existing_root(self) -> None:
        if not self._manifest_path.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_manifest()
        if not self._generation_path.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _read_generation(self._generation_path)

    def _provision(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        if not self._directory.is_dir():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._manifest_path.exists():
            self._validate_existing_root()
            return
        unexpected = [path for path in self._directory.iterdir() if path.name != "asset.lock"]
        if unexpected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        write_json_atomic(self._manifest_path, self._expected_manifest(), fsync=True)
        _write_text(self._generation_path, "0")
        sync_directory(self._directory)

    def _validate_manifest(self) -> None:
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
        if (
            not isinstance(value, dict)
            or value.get("format") != "linktools-ai-asset"
            or value.get("generation") != _GENERATION
        ):
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

    def _asset_key_digest(self, key: AssetKey) -> str:
        return hashlib.sha256(
            json.dumps(
                {"kind": key.kind, "id": key.id},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _entry_file(self, key: AssetKey) -> Path:
        digest = self._asset_key_digest(key)
        return self._entries_path / digest[:2] / f"{digest}.json"

    def _history_file(self, info: AssetInfo) -> Path:
        digest = self._asset_key_digest(info.key)
        return self._history_path / digest[:2] / digest / f"{info.revision.value:020d}.json"

    def _serialized_state(self) -> dict[str, bytes]:
        values = {
            "head.json": _json_bytes({"store_revision": self._revision}),
        }
        for info in self._entries.values():
            values[self._entry_file(info.key).relative_to(self._directory).as_posix()] = _json_bytes(
                self._info_to_json(info)
            )
        for history in self._versions.values():
            for info in history:
                values[self._history_file(info).relative_to(self._directory).as_posix()] = _json_bytes(
                    self._info_to_json(info)
                )
        return values

    def _existing_state(self) -> dict[str, bytes]:
        values: dict[str, bytes] = {}
        for path in (self._head_path, *self._entries_path.glob("*/*.json"), *self._history_path.glob("*/*/*.json")):
            if path.is_file():
                values[path.relative_to(self._directory).as_posix()] = path.read_bytes()
        return values

    async def _recover(self) -> None:
        self._journal.recover(
            lambda: _read_generation(self._generation_path),
            lambda target: _write_text(self._generation_path, str(target)),
        )

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


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


async def _read_asset_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return value


def _read_generation(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if value < 0:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return value


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    _sync_file(path)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _check_entry_revision(current: AssetInfo | None, expected: StorageEntryRevision | None) -> None:
    if expected is not None and (current is None or current.revision != expected):
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _mutates(operation: StorageOperation, current: AssetInfo | None, value: bytes) -> bool:
    if operation is StorageOperation.DELETE:
        return current is not None and current.status is not StorageEntryStatus.DELETED
    if operation is StorageOperation.RESET:
        return current is not None and current.status is not StorageEntryStatus.RESET
    return current is None or current.status is not StorageEntryStatus.NORMAL or current.etag != _etag(value)


def _next_info(
    root: AssetRoot,
    key: AssetKey,
    value: bytes,
    previous: AssetInfo | None,
    operation: StorageOperation,
    store_revision: int,
    metadata: Mapping[str, JsonValue] | None = None,
) -> AssetInfo:
    status = (
        StorageEntryStatus.NORMAL
        if operation is StorageOperation.PUT
        else StorageEntryStatus.DELETED
        if operation is StorageOperation.DELETE
        else StorageEntryStatus.RESET
    )
    content = value if status is StorageEntryStatus.NORMAL else b""
    entry_revision = 1 if previous is None else previous.revision.value + 1
    return AssetInfo(
        key,
        StorageEntryRevision(entry_revision),
        StorageRevision(str(store_revision)),
        _etag(content),
        len(content),
        status,
        root.root_id,
        root.digest,
        datetime.now(timezone.utc),
        normalize_storage_metadata(metadata),
    )


def _result(
    operation: StorageOperation, key: AssetKey, info: AssetInfo | None, revision: StorageRevision, *, changed: bool
) -> object:
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
            normalize_storage_metadata(raw.get("metadata")),
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
