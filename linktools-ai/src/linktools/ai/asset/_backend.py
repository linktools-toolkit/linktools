#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified container/file-tree AssetStore backends."""

import asyncio
import base64
import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ._domain import (
    AssetDeleteOrigin,
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryInfo,
    AssetEntryKey,
    AssetEntryOrigin,
    AssetEntryRevision,
    AssetEntrySnapshot,
    AssetEntryVersion,
    AssetInfo,
    AssetKey,
    AssetRevision,
    AssetRoot,
    AssetStoreRevision,
    AssetVersion,
    empty_etag,
    validate_rel_path,
)

_logger = environ.get_logger("ai.asset.backend")


@dataclass(frozen=True, slots=True)
class _DesiredEntry:
    content: bytes
    deleted: bool
    origin: AssetEntryOrigin
    source_digest: str | None


@dataclass(frozen=True, slots=True)
class _Manifest:
    info: AssetInfo
    files: tuple[AssetEntrySnapshot, ...]


class InMemoryAssetBackend:
    """Atomic in-process container and file history backend."""

    def __init__(self, root: "AssetRoot | None" = None, *, writable: bool = True) -> None:
        self._root = root or AssetRoot("memory:default", "memory", "memory", "memory")
        self._writable = writable
        self._store_revision = 0
        self._assets: dict[AssetKey, AssetInfo] = {}
        self._asset_history: dict[AssetKey, dict[int, _Manifest]] = {}
        self._entries: dict[AssetEntryKey, AssetEntryInfo] = {}
        self._contents: dict[AssetEntryKey, bytes] = {}
        self._entry_history: dict[AssetEntryKey, dict[int, tuple[AssetEntryVersion, bytes]]] = {}
        self._source_files: dict[AssetKey, dict[str, tuple[bytes, str]]] = {}
        self._overrides: dict[AssetKey, dict[str, bytes]] = {}
        self._tombstones: dict[AssetKey, set[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return self._writable

    async def initialize(self) -> None:
        return

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def current_revision(self) -> AssetStoreRevision:
        async with self._lock:
            return self._store_token()

    async def stat_asset(self, key: AssetKey) -> "AssetInfo | None":
        async with self._lock:
            return self._assets.get(key)

    async def list_assets(self) -> "tuple[AssetInfo, ...]":
        async with self._lock:
            return tuple(self._assets[key] for key in sorted(self._assets, key=lambda item: (item.kind, item.id)) if not self._assets[key].deleted)

    async def list_asset_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        async with self._lock:
            history = self._asset_history.get(key, {})
            return tuple(_asset_version(manifest.info) for _, manifest in sorted(history.items(), reverse=True))

    async def asset_revision_files(self, key: AssetKey, revision: AssetRevision) -> "tuple[AssetEntrySnapshot, ...]":
        async with self._lock:
            try:
                return self._asset_history[key][revision.value].files
            except KeyError as error:
                raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND) from error

    async def current_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "tuple[AssetEntryInfo, bytes] | None":
        async with self._lock:
            info = self._entries.get(key)
            if info is None or (info.deleted and not include_deleted):
                return None
            asset = self._assets.get(key.asset)
            if asset is None or asset.deleted:
                return None
            return info, self._contents.get(key, b"")

    async def list_current_files(self, asset: AssetKey, *, prefix: "str | None", include_deleted: bool) -> "tuple[tuple[AssetEntryInfo, bytes], ...]":
        async with self._lock:
            container = self._assets.get(asset)
            if container is None or container.deleted:
                return ()
            values = []
            for key, info in self._entries.items():
                if key.asset != asset or (info.deleted and not include_deleted) or (prefix is not None and not key.rel_path.startswith(prefix)):
                    continue
                values.append((info, self._contents.get(key, b"")))
            return tuple(sorted(values, key=lambda item: item[0].key.rel_path))

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        async with self._lock:
            return tuple(version for version, _ in sorted(self._entry_history.get(key, {}).values(), key=lambda item: item[0].entry_revision.value, reverse=True))

    async def get_file_at_revision(self, key: AssetEntryKey, revision: AssetEntryRevision) -> "bytes | None":
        async with self._lock:
            history = self._entry_history.get(key)
            if history is None or revision.value not in history:
                raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
            version, content = history[revision.value]
            return None if version.deleted else content

    async def snapshot_files(self, asset: AssetKey, revision: "AssetRevision | None", include_deleted: bool) -> "tuple[AssetEntrySnapshot, ...]":
        async with self._lock:
            if revision is None:
                info = self._assets.get(asset)
                if info is None or info.deleted:
                    return ()
                manifest = self._asset_history[asset][info.revision.value]
            else:
                try:
                    manifest = self._asset_history[asset][revision.value]
                except KeyError as error:
                    raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND) from error
            return tuple(item for item in manifest.files if include_deleted or not item.deleted)

    async def put_file(
        self,
        key: AssetEntryKey,
        value: bytes,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryInfo:
        result = await self.apply_file_batch(
            key.asset,
            (AssetEntryChange("PUT", key.rel_path, bytes(value), expected_entry_revision),),
            primary_path=primary_path,
            expected_revision=expected_revision,
            expected_store_revision=None,
        )
        item = result.results[0]
        if not isinstance(item, AssetEntryInfo):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return item

    async def delete_file(
        self,
        key: AssetEntryKey,
        *,
        primary_path: str,
        expected_entry_revision: AssetEntryRevision | None,
        expected_revision: AssetRevision | None,
    ) -> AssetEntryDeleteResult:
        if key.rel_path == primary_path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "primary asset file cannot be deleted")
        async with self._lock:
            self._require_writable()
            self._check_asset_cas(key.asset, expected_revision)
            container = self._assets.get(key.asset)
            if container is not None and container.deleted:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = self._entries.get(key)
            self._check_entry_cas(current, expected_entry_revision)
            if current is None or current.deleted and current.origin == "TOMBSTONE":
                info = self._assets.get(key.asset)
                return AssetEntryDeleteResult(key, False, None if current is None else current.entry_revision, info.revision if info else AssetRevision(1), self._store_token())
            desired = self._desired_for_asset(key.asset)
            desired[key.rel_path] = _DesiredEntry(b"", True, "TOMBSTONE", None)
            results = self._commit_locked({key.asset: desired}, {key.asset: None})
            info = self._entries[key]
            return AssetEntryDeleteResult(key, True, info.entry_revision, results[key.asset].revision, results[key.asset].store_revision)

    async def apply_file_batch(
        self,
        asset: AssetKey,
        changes: "Sequence[AssetEntryChange]",
        *,
        primary_path: str,
        expected_revision: "AssetRevision | None",
        expected_store_revision: "AssetStoreRevision | None",
    ) -> AssetEntryBatchResult:
        async with self._lock:
            self._require_writable()
            if expected_store_revision is not None and expected_store_revision != self._store_token():
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._check_asset_cas(asset, expected_revision)
            container = self._assets.get(asset)
            if container is not None and container.deleted:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if len({item.rel_path for item in changes}) != len(changes):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "duplicate asset file path")
            desired = self._desired_for_asset(asset)
            no_op_deletes: set[str] = set()
            for change in changes:
                key = AssetEntryKey(asset, change.rel_path)
                self._check_entry_cas(self._entries.get(key), change.expected_entry_revision)
                if change.operation == "PUT":
                    if change.value is None:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    desired[change.rel_path] = _DesiredEntry(bytes(change.value), False, "OVERRIDE", None)
                elif change.operation == "DELETE":
                    if change.rel_path == primary_path:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "primary asset file cannot be deleted")
                    current = self._entries.get(key)
                    if current is not None and current.deleted and current.origin == "TOMBSTONE":
                        no_op_deletes.add(change.rel_path)
                        continue
                    if current is None:
                        no_op_deletes.add(change.rel_path)
                        continue
                    desired[change.rel_path] = _DesiredEntry(b"", True, "TOMBSTONE", None)
                else:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            results = self._commit_locked({asset: desired}, {asset: None})
            current_revision = results.get(asset, self._assets.get(asset))
            if current_revision is None:
                store_revision = self._store_token()
                return AssetEntryBatchResult(
                    asset,
                    AssetRevision(1),
                    store_revision,
                    True,
                    tuple(
                        AssetEntryDeleteResult(AssetEntryKey(asset, change.rel_path), False, None, AssetRevision(1), store_revision)
                        for change in changes
                    ),
                )
            result_items: list[AssetEntryInfo | AssetEntryDeleteResult] = []
            for change in changes:
                key = AssetEntryKey(asset, change.rel_path)
                info = self._entries.get(key)
                if info is None:
                    result_items.append(AssetEntryDeleteResult(key, False, None, current_revision.revision, current_revision.store_revision))
                elif info.deleted:
                    result_items.append(AssetEntryDeleteResult(key, change.rel_path not in no_op_deletes, info.entry_revision, current_revision.revision, current_revision.store_revision))
                else:
                    result_items.append(info)
            return AssetEntryBatchResult(asset, current_revision.revision, current_revision.store_revision, True, tuple(result_items))

    async def replace_tree(
        self,
        asset: AssetKey,
        files: "Mapping[str, bytes]",
        *,
        deleted_rel_paths: "Collection[str]",
        primary_path: str,
        expected_revision: "AssetRevision | None",
    ) -> AssetInfo:
        async with self._lock:
            self._require_writable()
            self._check_asset_cas(asset, expected_revision)
            normalized = {validate_rel_path(path): bytes(value) for path, value in files.items()}
            deleted = {validate_rel_path(path) for path in deleted_rel_paths}
            if primary_path not in normalized or set(normalized).intersection(deleted):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            desired = {path: _DesiredEntry(value, False, "OVERRIDE", None) for path, value in normalized.items()}
            for path in set(self._desired_for_asset(asset)).union(deleted):
                if path not in desired:
                    desired[path] = _DesiredEntry(b"", True, "TOMBSTONE", None)
            result = self._commit_locked({asset: desired}, {asset: None})
            return result[asset]

    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]":
        async with self._lock:
            self._require_writable()
            self._check_asset_cas(key, expected_revision)
            current = self._assets.get(key)
            if current is None or current.deleted and current.deleted_by == "OVERRIDE":
                return False, current
            desired = self._desired_for_asset(key)
            desired = {path: _DesiredEntry(b"", True, "TOMBSTONE", None) for path in desired}
            result = self._commit_locked({key: desired}, {key: "OVERRIDE"})
            return True, result[key]

    async def apply_asset_batch(self, changes: "Sequence[tuple[AssetKey, bytes | None, str, Literal['PUT', 'DELETE'], AssetRevision | None]]", *, expected_store_revision: "AssetStoreRevision | None") -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        async with self._lock:
            self._require_writable()
            if expected_store_revision is not None and expected_store_revision != self._store_token():
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if len({asset for asset, *_rest in changes}) != len(changes):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "duplicate asset key")
            desired_by_asset: dict[AssetKey, dict[str, _DesiredEntry]] = {}
            delete_origins: dict[AssetKey, AssetDeleteOrigin | None] = {}
            for asset, value, primary_path, operation, expected_revision in changes:
                self._check_asset_cas(asset, expected_revision)
                current = self._assets.get(asset)
                if current is not None and current.deleted:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if asset not in desired_by_asset:
                    desired_by_asset[asset] = self._desired_for_asset(asset)
                if operation == "PUT":
                    if value is None:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    desired_by_asset[asset][primary_path] = _DesiredEntry(value, False, "OVERRIDE", None)
                else:
                    desired_by_asset[asset] = self._desired_for_asset(asset)
                    delete_origins[asset] = "OVERRIDE"
            committed = self._commit_locked(desired_by_asset, delete_origins)
            result: list[AssetInfo | AssetDeleteResult] = []
            for asset, value, _primary_path, operation, _expected_revision in changes:
                info = committed.get(asset, self._assets.get(asset))
                if operation == "DELETE":
                    result.append(AssetDeleteResult(asset, info is not None and info.deleted, None if info is None else info.revision, self._store_token()))
                elif info is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                else:
                    result.append(info)
            return tuple(result)

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        async with self._lock:
            self._require_writable()
            self._check_asset_cas(key, expected_revision)
            try:
                manifest = self._asset_history[key][revision.value]
            except KeyError as error:
                raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND) from error
            if manifest.info.deleted:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "deleted asset revision cannot be restored")
            desired = {item.key.rel_path: _DesiredEntry(item.content or b"", item.deleted, "OVERRIDE" if not item.deleted else "TOMBSTONE", None) for item in manifest.files}
            for path in set(self._desired_for_asset(key)).difference(desired):
                desired[path] = _DesiredEntry(b"", True, "TOMBSTONE", None)
            return self._commit_locked({key: desired}, {key: None})[key]

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        async with self._lock:
            self._require_writable()
            self._check_asset_cas(source, expected_source_revision)
            if target in self._assets or target in self._asset_history:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            source_info = self._assets.get(source)
            if source_info is None or source_info.deleted:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            source_desired = self._desired_for_asset(source)
            target_desired = {path: _DesiredEntry(item.content, item.deleted, "OVERRIDE" if not item.deleted else "TOMBSTONE", None) for path, item in source_desired.items()}
            result = self._commit_locked({target: target_desired, source: source_desired}, {source: "OVERRIDE", target: None})
            return result[target]

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: "Mapping[str, str]") -> AssetStoreRevision:
        async with self._lock:
            self._require_writable()
            self._source_files = {asset: dict(files) for asset, files in source_files.items()}
            desired_by_asset: dict[AssetKey, dict[str, _DesiredEntry]] = {}
            delete_origins: dict[AssetKey, AssetDeleteOrigin | None] = {}
            all_assets = set(source_files).union(self._assets)
            for asset in all_assets:
                desired = self._desired_for_asset(asset, source_override=source_files.get(asset, {}))
                primary = primary_paths.get(asset.kind)
                has_primary = primary is not None and primary in desired and not desired[primary].deleted
                current = self._assets.get(asset)
                if current is not None and current.deleted_by == "OVERRIDE":
                    delete_origins[asset] = "OVERRIDE"
                elif not has_primary and (current is not None or desired):
                    delete_origins[asset] = "SOURCE"
                else:
                    delete_origins[asset] = None
                desired_by_asset[asset] = desired
            self._commit_locked(desired_by_asset, delete_origins)
            _logger.info("asset source refresh committed: assets=%s store_revision=%s", len(source_files), self._store_revision)
            return self._store_token()

    def _desired_for_asset(self, asset: AssetKey, *, source_override: "Mapping[str, tuple[bytes, str]] | None" = None) -> "dict[str, _DesiredEntry]":
        source = source_override if source_override is not None else self._source_files.get(asset, {})
        paths = set(source).union(self._overrides.get(asset, {})).union(self._tombstones.get(asset, set())).union(key.rel_path for key in self._entries if key.asset == asset)
        if self._assets.get(asset) is not None and self._assets[asset].deleted_by == "OVERRIDE":
            return {path: _DesiredEntry(b"", True, "TOMBSTONE", None) for path in paths}
        result: dict[str, _DesiredEntry] = {}
        for path in paths:
            if path in self._tombstones.get(asset, set()):
                result[path] = _DesiredEntry(b"", True, "TOMBSTONE", None)
            elif path in self._overrides.get(asset, {}):
                result[path] = _DesiredEntry(self._overrides[asset][path], False, "OVERRIDE", None)
            elif path in source:
                content, digest = source[path]
                result[path] = _DesiredEntry(content, False, "SOURCE", digest)
            else:
                result[path] = _DesiredEntry(b"", True, "SOURCE", None)
        return result

    def _commit_locked(self, desired_by_asset: Mapping[AssetKey, Mapping[str, _DesiredEntry]], delete_origins: "Mapping[AssetKey, AssetDeleteOrigin | None]") -> "dict[AssetKey, AssetInfo]":
        mutations = {asset: desired for asset, desired in desired_by_asset.items() if self._asset_changed(asset, desired, delete_origins.get(asset))}
        if not mutations:
            return {asset: info for asset, info in self._assets.items() if asset in desired_by_asset}
        self._store_revision += 1
        store_revision = self._store_token()
        results: dict[AssetKey, AssetInfo] = {}
        for asset, desired in mutations.items():
            previous = self._assets.get(asset)
            asset_revision = AssetRevision(1 if previous is None else previous.revision.value + 1)
            deleted_by = delete_origins.get(asset)
            active_entries = [item for path, item in desired.items() if not item.deleted]
            container_deleted = deleted_by is not None or not active_entries
            if deleted_by is None and not active_entries and previous is not None and previous.deleted_by is not None:
                deleted_by = previous.deleted_by
                container_deleted = True
            entries = self._materialize_entries(asset, desired, asset_revision, store_revision)
            tree_etag = _tree_etag(entries)
            source_layers = tuple(sorted({"source" if item.origin == "SOURCE" else "override" for item in entries}))
            composition_digest = canonical_sha256(
                {
                    "source": sorted((path, digest) for path, (_content, digest) in self._source_files.get(asset, {}).items()),
                    "overrides": sorted((path, _etag(content)) for path, content in self._overrides.get(asset, {}).items()),
                    "tombstones": sorted(self._tombstones.get(asset, set())),
                }
            )
            info = AssetInfo(asset, asset_revision, store_revision, empty_etag() if container_deleted else tree_etag, 0 if container_deleted else sum(item.size for item in entries if not item.deleted), 0 if container_deleted else sum(not item.deleted for item in entries), container_deleted, deleted_by if container_deleted else None, composition_digest, source_layers, datetime.now(timezone.utc))
            self._assets[asset] = info
            for item in entries:
                self._entries[item.key] = item
            manifest = _Manifest(info, tuple(_snapshot(item, self._contents.get(item.key, b"")) for item in sorted(entries, key=lambda value: value.key.rel_path)))
            self._asset_history.setdefault(asset, {})[asset_revision.value] = manifest
            results[asset] = info
        return results

    def _materialize_entries(self, asset: AssetKey, desired: Mapping[str, _DesiredEntry], asset_revision: AssetRevision, store_revision: AssetStoreRevision) -> tuple[AssetEntryInfo, ...]:
        result: list[AssetEntryInfo] = []
        for path, state in sorted(desired.items()):
            key = AssetEntryKey(asset, path)
            previous = self._entries.get(key)
            if previous is None or _entry_state(previous, self._contents.get(key, b"")) != state:
                history = self._entry_history.setdefault(key, {})
                next_revision = AssetEntryRevision(max((value for value in history), default=0) + 1)
                content = b"" if state.deleted else state.content
                version = AssetEntryVersion(next_revision, _etag(content), len(content), datetime.now(timezone.utc), state.deleted, state.origin, state.source_digest)
                history[next_revision.value] = (version, content)
                self._contents[key] = content
                self._overrides.setdefault(asset, {}).pop(path, None)
                self._tombstones.setdefault(asset, set()).discard(path)
                if state.origin == "OVERRIDE" and not state.deleted:
                    self._overrides.setdefault(asset, {})[path] = state.content
                elif state.origin == "TOMBSTONE":
                    self._tombstones.setdefault(asset, set()).add(path)
                info = _entry_info(key, next_revision, asset_revision, store_revision, state, content)
            else:
                info = AssetEntryInfo(key, key.file_id, previous.entry_revision, asset_revision, store_revision, previous.etag, previous.size, previous.deleted, previous.origin, previous.source_digest, previous.layer, previous.writable, previous.modified_at)
            result.append(info)
        return tuple(result)

    def _asset_changed(self, asset: AssetKey, desired: Mapping[str, _DesiredEntry], deleted_by: "AssetDeleteOrigin | None") -> bool:
        current = self._assets.get(asset)
        if current is None:
            return bool(desired)
        if current.deleted_by == "OVERRIDE" and deleted_by == "OVERRIDE":
            return False
        if current.deleted_by != deleted_by and (deleted_by is not None or current.deleted_by is not None):
            return True
        current_states = {key.rel_path: _entry_state(info, self._contents.get(key, b"")) for key, info in self._entries.items() if key.asset == asset}
        return current_states != dict(desired)

    def _check_asset_cas(self, asset: AssetKey, expected: "AssetRevision | None") -> None:
        if expected is not None and (self._assets.get(asset) is None or self._assets[asset].revision != expected):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    @staticmethod
    def _check_entry_cas(current: "AssetEntryInfo | None", expected: "AssetEntryRevision | None") -> None:
        if expected is not None and (current is None or current.entry_revision != expected):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _store_token(self) -> AssetStoreRevision:
        return AssetStoreRevision(str(self._store_revision))

    def _require_writable(self) -> None:
        if not self._writable:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)

    def _dump_state(self) -> "dict[str, object]":
        return {
            "store_revision": self._store_revision,
            "assets": [
                _encode_manifest(manifest)
                for histories in self._asset_history.values()
                for manifest in histories.values()
            ],
        }

    def _load_state(self, raw: object) -> None:
        if not isinstance(raw, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._store_revision = int(raw.get("store_revision", 0))
        for item in raw.get("assets", []):
            if not isinstance(item, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            manifest = _decode_manifest(item)
            self._assets[manifest.info.key] = manifest.info
            self._asset_history.setdefault(manifest.info.key, {})[manifest.info.revision.value] = manifest
            for snapshot in manifest.files:
                info = _entry_info_from_snapshot(snapshot, manifest.info)
                self._entries[snapshot.key] = info
                self._contents[snapshot.key] = snapshot.content or b""
                version = AssetEntryVersion(snapshot.entry_revision, snapshot.etag, len(snapshot.content or b""), manifest.info.modified_at, snapshot.deleted, snapshot.origin, snapshot.source_digest)
                self._entry_history.setdefault(snapshot.key, {})[snapshot.entry_revision.value] = (version, snapshot.content or b"")
                if snapshot.origin == "OVERRIDE" and not snapshot.deleted:
                    self._overrides.setdefault(snapshot.key.asset, {})[snapshot.key.rel_path] = snapshot.content or b""
                if snapshot.origin == "TOMBSTONE":
                    self._tombstones.setdefault(snapshot.key.asset, set()).add(snapshot.key.rel_path)


def _entry_state(info: AssetEntryInfo, content: bytes) -> _DesiredEntry:
    return _DesiredEntry(content, info.deleted, info.origin, info.source_digest)


def _entry_info(key: AssetEntryKey, revision: AssetEntryRevision, asset_revision: AssetRevision, store_revision: AssetStoreRevision, state: _DesiredEntry, content: bytes) -> AssetEntryInfo:
    return AssetEntryInfo(key, key.file_id, revision, asset_revision, store_revision, _etag(content), len(content), state.deleted, state.origin, state.source_digest, "source" if state.origin == "SOURCE" else "primary", state.origin != "SOURCE", datetime.now(timezone.utc))


def _entry_info_from_snapshot(snapshot: AssetEntrySnapshot, asset: AssetInfo) -> AssetEntryInfo:
    return AssetEntryInfo(snapshot.key, snapshot.key.file_id, snapshot.entry_revision, asset.revision, asset.store_revision, snapshot.etag, len(snapshot.content or b""), snapshot.deleted, snapshot.origin, snapshot.source_digest, "source" if snapshot.origin == "SOURCE" else "primary", snapshot.origin != "SOURCE", asset.modified_at)


def _snapshot(info: AssetEntryInfo, content: bytes) -> AssetEntrySnapshot:
    return AssetEntrySnapshot(info.key, info.entry_revision, info.etag, info.deleted, info.origin, info.source_digest, None if info.deleted else content)


def _tree_etag(entries: Sequence[AssetEntryInfo]) -> str:
    return canonical_sha256([{"path": item.key.rel_path, "etag": item.etag, "deleted": item.deleted} for item in sorted(entries, key=lambda value: value.key.rel_path)])


def _asset_version(info: AssetInfo) -> AssetVersion:
    return AssetVersion(info.revision, info.etag, info.size, info.file_count, info.composition_digest, info.modified_at, info.deleted, info.deleted_by)


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode_manifest(manifest: _Manifest) -> "dict[str, object]":
    info = manifest.info
    return {"key": [info.key.kind, info.key.id], "revision": info.revision.value, "store_revision": info.store_revision.value, "etag": info.etag, "size": info.size, "file_count": info.file_count, "deleted": info.deleted, "deleted_by": info.deleted_by, "composition_digest": info.composition_digest, "source_layers": list(info.source_layers), "modified_at": info.modified_at.isoformat(), "files": [{"path": item.key.rel_path, "entry_revision": item.entry_revision.value, "etag": item.etag, "deleted": item.deleted, "origin": item.origin, "source_digest": item.source_digest, "content": None if item.content is None else base64.b64encode(item.content).decode("ascii")} for item in manifest.files]}


def _decode_manifest(raw: "dict[str, object]") -> _Manifest:
    key_value = raw.get("key")
    if not isinstance(key_value, list) or len(key_value) != 2:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    key = AssetKey(str(key_value[0]), str(key_value[1]))
    info = AssetInfo(key, AssetRevision(int(raw["revision"])), AssetStoreRevision(str(raw["store_revision"])), str(raw["etag"]), int(raw["size"]), int(raw["file_count"]), bool(raw["deleted"]), raw.get("deleted_by"), str(raw["composition_digest"]), tuple(str(item) for item in raw.get("source_layers", [])), datetime.fromisoformat(str(raw["modified_at"])))
    files = []
    for value in raw.get("files", []):
        if not isinstance(value, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = value.get("content")
        decoded = None if content is None else base64.b64decode(str(content), validate=True)
        files.append(AssetEntrySnapshot(AssetEntryKey(key, str(value["path"])), AssetEntryRevision(int(value["entry_revision"])), str(value["etag"]), bool(value["deleted"]), value["origin"], value.get("source_digest"), decoded))
    return _Manifest(info, tuple(files))


__all__ = ["InMemoryAssetBackend"]
