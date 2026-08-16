#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed logical asset discovery, scoped resources, and writes."""

import asyncio
import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath

from linktools.core import environ

from ..core import JsonValue, Page, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import (
    StorageDeleteResult,
    StorageChange,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOperation,
    StorageResetResult,
)
from ._domain import AssetInfo, AssetKey
from ._logical import (
    AssetDiscoveryStatus,
    AssetEntry,
    AssetRef,
    AssetResource,
    AssetTypeBinding,
    AssetTypeRegistrySnapshot,
    AssetVariantBinding,
    DirectoryLayout,
    ResolvedAsset,
    SingleFileLayout,
)
from ._store import AssetStore

_logger = environ.get_logger("ai.asset.repository")


@dataclass(frozen=True, slots=True)
class _Candidate:
    variant: AssetVariantBinding[object]
    ref: AssetRef
    info: AssetInfo


@dataclass(frozen=True, slots=True)
class _Probe:
    owner_id: str | None
    owner_variants: tuple[str, ...]
    owners: tuple[_Candidate, ...]
    candidates: tuple[_Candidate, ...]


@dataclass(slots=True)
class _LockState:
    lock: asyncio.Lock
    references: int = 0


class _RepositoryKeyedLock:
    """Keep per-asset mutation locks bounded by active holders and waiters."""

    def __init__(self) -> None:
        self._states: dict[str, _LockState] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, key: str):
        async with self.hold_many((key,)):
            yield

    @asynccontextmanager
    async def hold_many(self, keys: Sequence[str]):
        ordered = tuple(sorted(set(keys)))
        async with AsyncExitStack() as stack:
            for key in ordered:
                await stack.enter_async_context(self._hold_one(key))
            yield

    @asynccontextmanager
    async def _hold_one(self, key: str):
        async with self._guard:
            state = self._states.get(key)
            if state is None:
                state = _LockState(asyncio.Lock())
                self._states[key] = state
            state.references += 1
        try:
            await state.lock.acquire()
        except BaseException:
            await self._remove_reference(key, state)
            raise
        try:
            yield
        finally:
            state.lock.release()
            await self._remove_reference(key, state)

    async def _remove_reference(self, key: str, state: _LockState) -> None:
        async with self._guard:
            state.references -= 1
            if state.references == 0 and not state.lock.locked():
                self._states.pop(key, None)


class AssetScope:
    """Expose raw files inside one resolved logical asset boundary."""

    def __init__(
        self,
        store: AssetStore,
        ref: AssetRef,
        variant: AssetVariantBinding[object],
        reserved_entry_paths: tuple[str, ...],
    ) -> None:
        self._store = store
        self._ref = ref
        self._variant = variant
        self._reserved_entry_paths = frozenset(reserved_entry_paths)
        layout = variant.layout
        self._directory = isinstance(layout, DirectoryLayout)
        self._entry_key = layout.entry_key(ref)
        self._entry_path = layout.scope_entry_path(self._entry_key)

    @property
    def entry_path(self) -> str:
        """Return the entry name relative to this scope."""
        return self._entry_path

    async def stat(self, path: str) -> "AssetInfo | None":
        """Return metadata for one scope-relative raw file."""
        return await self._store.stat(self._key_for(path))

    async def get(self, path: str) -> "bytes | None":
        """Read one scope-relative raw file."""
        return await self._store.get(self._key_for(path))

    async def list(
        self,
        *,
        prefix: "str | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[AssetResource]":
        """List raw files owned by this scope at any supported depth."""
        _validate_limit(limit)
        normalized_prefix = _normalize_scope_prefix(prefix)
        infos = await _load_all(self._store, kind=self._ref.kind)
        resources: list[AssetResource] = []
        for info in infos:
            path = self._relative_path(info.key)
            if path is None or (
                normalized_prefix is not None
                and path != normalized_prefix
                and not path.startswith(normalized_prefix + "/")
            ):
                continue
            resources.append(AssetResource(path, info, path in self._reserved_entry_paths))
        resources.sort(key=lambda item: item.path)
        scope_digest = canonical_sha256({"kind": self._ref.kind, "id": self._ref.id})
        start = _scope_cursor_start(cursor, scope_digest, self._entry_path, normalized_prefix, resources)
        selected = tuple(resources[start : start + limit])
        next_cursor = None
        if selected and start + len(selected) < len(resources):
            next_cursor = _scope_cursor(scope_digest, self._entry_path, normalized_prefix, selected[-1].path)
        return Page(selected, next_cursor)

    async def put(
        self,
        path: str,
        value: bytes,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> AssetInfo:
        """Write one non-entry raw resource through the underlying store."""
        key = self._mutation_key_for(path)
        result = await self._store.put(
            key,
            value,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        return result

    async def delete(
        self,
        path: str,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        """Delete one non-entry raw resource through the underlying store."""
        key = self._mutation_key_for(path)
        return await self._store.delete(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )

    async def reset(
        self,
        path: str,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        """Reset one non-entry raw resource so lower layers can reappear."""
        key = self._mutation_key_for(path)
        return await self._store.reset(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )

    def _relative_path(self, key: AssetKey) -> str | None:
        if self._directory:
            prefix = self._ref.id + "/"
            if key.kind != self._ref.kind or not key.id.startswith(prefix):
                return None
            path = key.id[len(prefix) :]
        elif key != self._entry_key:
            return None
        else:
            path = self._entry_path
        try:
            _validate_scope_path(path)
        except AIError:
            return None
        return path

    def _key_for(self, path: str) -> AssetKey:
        normalized = _validate_scope_path(path)
        if not self._directory:
            if normalized != self._entry_path:
                raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT)
            return self._entry_key
        try:
            return AssetKey(self._ref.kind, f"{self._ref.id}/{normalized}")
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    def _mutation_key_for(self, path: str) -> AssetKey:
        normalized = _validate_scope_path(path)
        if normalized in self._reserved_entry_paths:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"path": normalized, "reason": "entry_managed_by_repository"},
            )
        return self._key_for(normalized)


class AssetRepository:
    """Resolve typed logical assets over the raw AssetStore API."""

    def __init__(self, store: AssetStore, registry: AssetTypeRegistrySnapshot) -> None:
        if store is None or registry is None or not registry.frozen:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._store = store
        self._registry = registry
        self._locks = _RepositoryKeyedLock()

    @property
    def ready(self) -> bool:
        """Return whether the underlying raw asset store is initialized."""
        return self._store.ready

    async def resolve(self, ref: AssetRef) -> "ResolvedAsset[object]":
        """Discover, stabilize, decode, and return one logical asset."""
        binding = self._binding_for(ref)
        first = await self._probe(ref, binding)
        self._raise_for_owner(first, ref)
        if len(first.candidates) == 0:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if len(first.candidates) > 1:
            self._raise_layout_conflict(ref, first.candidates)
        candidate = first.candidates[0]
        data = await self._store.get(candidate.info.key)
        if data is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        second = await self._probe(ref, binding)
        if not _probe_equal(first, second):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if hashlib.sha256(data).hexdigest() != candidate.info.etag:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _decode_value(candidate.variant, ref, data, binding.value_type)
        scope = self._scope(ref, candidate.variant, binding)
        _validate_identity(binding, ref, value)
        _logger.debug(
            "asset resolved: kind=%s id_digest=%s variant=%s revision=%s",
            ref.kind,
            canonical_sha256(ref.id),
            candidate.variant.name,
            candidate.info.revision,
        )
        return ResolvedAsset(ref, candidate.variant.name, value, candidate.info, scope)

    async def list(
        self,
        *,
        kind: str,
        prefix: "str | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[AssetEntry]":
        """List logical assets by layout without decoding content."""
        _validate_limit(limit)
        binding = self._binding_for_kind(kind)
        normalized_prefix = _normalize_logical_prefix(kind, prefix, binding)
        infos = await _load_all(self._store, kind=kind)
        entries = _discover(binding, infos)
        visible = tuple(
            entry
            for entry in sorted(entries.values(), key=lambda item: item.ref.id)
            if normalized_prefix is None
            or entry.ref.id == normalized_prefix
            or entry.ref.id.startswith(normalized_prefix + "/")
        )
        discovery_digest = canonical_sha256(
            [
                {
                    "kind": item.ref.kind,
                    "id": item.ref.id,
                    "status": item.status.value,
                    "variants": list(item.variants),
                }
                for item in visible
            ]
        )
        start = _typed_cursor_start(
            cursor,
            kind,
            normalized_prefix,
            self._registry.layout_digest,
            discovery_digest,
            visible,
        )
        selected = visible[start : start + limit]
        next_cursor = None
        if selected and start + len(selected) < len(visible):
            next_cursor = _typed_cursor(
                kind,
                normalized_prefix,
                self._registry.layout_digest,
                discovery_digest,
                selected[-1].ref.id,
            )
        conflict_count = sum(item.status is AssetDiscoveryStatus.CONFLICT for item in visible)
        _logger.debug(
            "asset logical list: kind=%s count=%s conflicts=%s discovery_digest=%s",
            kind,
            len(visible),
            conflict_count,
            discovery_digest,
        )
        return Page(selected, next_cursor)

    async def put(
        self,
        ref: AssetRef,
        value: "object",
        *,
        variant: "str | None" = None,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "ResolvedAsset[object]":
        """Encode and write one logical asset while preserving its layout."""
        binding = self._binding_for(ref)
        async with self._locks.hold(f"{ref.kind}:{ref.id}"):
            first = await self._probe(ref, binding)
            self._raise_for_owner(first, ref, for_write=True)
            if len(first.candidates) > 1:
                self._raise_layout_conflict(ref, first.candidates)
            selected = self._select_write_variant(binding, first, variant)
            old_info: AssetInfo | None = first.candidates[0].info if first.candidates else None
            old_bytes: bytes | None = None
            if old_info is not None:
                old_bytes = await self._stable_old_bytes(ref, binding, first, old_info)
            is_new_directory = old_info is None and isinstance(selected.layout, DirectoryLayout)
            key = selected.layout.entry_key(ref)
            if is_new_directory:
                await self._ensure_no_descendant_candidates(ref, binding, exclude_key=key)
            _validate_value_type(binding, value)
            _validate_identity(binding, ref, value)
            encoded = _encode_value(selected, ref, value, binding.value_type)
            new_info = await self._store.put(
                key,
                encoded,
                expected_revision=expected_revision,
                metadata=metadata,
            )
            post = await self._probe(ref, binding)
            takeover = is_new_directory and await self._has_descendant_candidates(ref, binding, exclude_key=key)
            if (
                post.owner_id is not None
                or len(post.candidates) != 1
                or post.candidates[0].variant.name != selected.name
                or not _info_matches(post.candidates[0].info, new_info)
                or takeover
            ):
                await self._compensate(ref, key, new_info, old_info, old_bytes)
                _logger.warning(
                    "asset typed write conflicted after write: kind=%s id_digest=%s variant=%s",
                    ref.kind,
                    canonical_sha256(ref.id),
                    selected.name,
                )
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            scope = self._scope(ref, selected, binding)
            _logger.debug(
                "asset typed put: kind=%s id_digest=%s variant=%s revision=%s existing=%s",
                ref.kind,
                canonical_sha256(ref.id),
                selected.name,
                new_info.revision,
                old_info is not None,
            )
            return ResolvedAsset(ref, selected.name, value, new_info, scope)

    async def rename(
        self,
        source: AssetRef,
        target: AssetRef,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "ResolvedAsset[object]":
        """Atomically move one logical asset and its owned raw resources."""
        if source.kind != target.kind:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"reason": "rename_kind_mismatch"},
            )
        if source == target:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if not self._store.atomic_batch:
            raise AIError(
                ErrorCode.STORAGE_CAPABILITY_MISSING,
                safe_details={"capability": "atomic_batch"},
            )
        binding = self._binding_for(source)
        async with self._locks.hold_many(
            (f"{source.kind}:{source.id}", f"{target.kind}:{target.id}")
        ):
            batch_revision = await self._store.current_revision()
            source_probe = await self._probe(source, binding)
            self._raise_for_owner(source_probe, source)
            if len(source_probe.candidates) == 0:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if len(source_probe.candidates) > 1:
                self._raise_layout_conflict(source, source_probe.candidates)
            source_candidate = source_probe.candidates[0]
            if expected_revision is not None and source_candidate.info.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            target_probe = await self._probe(target, binding)
            if target_probe.candidates:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if target_probe.owner_id is not None or await self._has_descendant_candidates(target, binding):
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)

            infos = await self._rename_source_infos(source, source_candidate.variant)
            changes = await self._rename_changes(
                source,
                target,
                source_candidate.variant,
                binding,
                infos,
                metadata,
            )
            result = await self._store.apply_batch(changes, expected_revision=batch_revision)
            if not result.atomic:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info(
                "asset logical rename committed: kind=%s source_id_digest=%s "
                "target_id_digest=%s changes=%s revision=%s",
                source.kind,
                canonical_sha256(source.id),
                canonical_sha256(target.id),
                len(changes),
                result.store_revision,
            )
        return await self.resolve(target)

    async def _rename_source_infos(
        self,
        source: AssetRef,
        variant: AssetVariantBinding[object],
    ) -> tuple[AssetInfo, ...]:
        if isinstance(variant.layout, SingleFileLayout):
            info = await self._store.stat(variant.layout.entry_key(source))
            if info is None or info.status is not StorageEntryStatus.NORMAL:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return (info,)
        infos = await _load_all(self._store, kind=source.kind)
        prefix = source.id + "/"
        selected = tuple(
            info for info in infos if info.key.id == variant.layout.entry_key(source).id
            or info.key.id.startswith(prefix)
        )
        if not selected:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return tuple(sorted(selected, key=lambda info: info.key.id))

    async def _rename_changes(
        self,
        source: AssetRef,
        target: AssetRef,
        variant: AssetVariantBinding[object],
        binding: AssetTypeBinding[object],
        infos: Sequence[AssetInfo],
        metadata: "Mapping[str, JsonValue] | None",
    ) -> tuple[StorageChange[AssetKey, bytes], ...]:
        entry_key = variant.layout.entry_key(source)
        source_entry = next(info for info in infos if info.key == entry_key)
        source_bytes = await self._store.get(source_entry.key)
        if source_bytes is None or hashlib.sha256(source_bytes).hexdigest() != source_entry.etag:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _decode_value(variant, source, source_bytes, binding.value_type)
        if not _identity_matches(binding, target, value):
            if binding.retargeter is None:
                raise AIError(
                    ErrorCode.STORAGE_CAPABILITY_MISSING,
                    safe_details={"capability": "asset_retarget"},
                )
            try:
                value = binding.retargeter(source, target, variant.name, value)
            except AIError:
                raise
            except Exception as error:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
            _validate_value_type(binding, value)
            _validate_identity(binding, target, value)
        target_entry_key = variant.layout.entry_key(target)
        target_existing = await self._store.stat(target_entry_key)
        changes: list[StorageChange[AssetKey, bytes]] = []
        for info in infos:
            data = await self._store.get(info.key)
            if data is None or hashlib.sha256(data).hexdigest() != info.etag:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            suffix = _rename_suffix(source, variant, info.key)
            target_key = AssetKey(target.kind, target.id + suffix)
            if info.key == source_entry.key:
                data = _encode_value(variant, target, value, binding.value_type)
            existing = target_existing if target_key == target_entry_key else await self._store.stat(target_key)
            changes.append(
                StorageChange(
                    StorageOperation.PUT,
                    target_key,
                    data,
                    None if existing is None else existing.revision,
                    metadata or {},
                )
            )
        changes.extend(
            StorageChange(
                StorageOperation.DELETE,
                info.key,
                None,
                info.revision,
                metadata or {},
            )
            for info in infos
        )
        return tuple(changes)

    def _binding_for_kind(self, kind: str) -> AssetTypeBinding[object]:
        return self._registry.binding(kind)

    def _binding_for(self, ref: AssetRef) -> AssetTypeBinding[object]:
        binding = self._binding_for_kind(ref.kind)
        if not binding.allow_nested_id and "/" in ref.id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            for variant in binding.variants:
                variant.layout.entry_key(ref)
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        return binding

    async def _probe(self, ref: AssetRef, binding: AssetTypeBinding[object]) -> _Probe:
        owner_id: str | None = None
        owner_variants: list[str] = []
        owners: tuple[_Candidate, ...] = ()
        parts = ref.id.split("/")
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            ancestor_ref = AssetRef(ref.kind, ancestor)
            directory_matches: dict[str, _Candidate] = {}
            for variant in binding.variants:
                if not isinstance(variant.layout, DirectoryLayout):
                    continue
                info = await self._normal_stat(variant.layout.entry_key(ancestor_ref))
                if info is not None:
                    directory_matches[variant.name] = _Candidate(variant, ancestor_ref, info)
            if not directory_matches:
                continue
            matches = list(directory_matches.values())
            for variant in binding.variants:
                if variant.name in directory_matches:
                    continue
                info = await self._normal_stat(variant.layout.entry_key(ancestor_ref))
                if info is not None:
                    matches.append(_Candidate(variant, ancestor_ref, info))
            matches.sort(key=lambda item: item.variant.name)
            if matches:
                owner_id = ancestor
                owner_variants = sorted(item.variant.name for item in matches)
                owners = tuple(matches)
                break
        candidates: list[_Candidate] = []
        if owner_id is None:
            for variant in binding.variants:
                info = await self._normal_stat(variant.layout.entry_key(ref))
                if info is not None:
                    candidates.append(_Candidate(variant, ref, info))
        return _Probe(owner_id, tuple(owner_variants), owners, tuple(candidates))

    async def _normal_stat(self, key: AssetKey) -> AssetInfo | None:
        info = await self._store.stat(key)
        return info if info is not None and info.status is StorageEntryStatus.NORMAL else None

    async def _stable_old_bytes(
        self,
        ref: AssetRef,
        binding: AssetTypeBinding[object],
        first: _Probe,
        info: AssetInfo,
    ) -> bytes:
        data = await self._store.get(info.key)
        if data is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        second = await self._probe(ref, binding)
        if not _probe_equal(first, second):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if hashlib.sha256(data).hexdigest() != info.etag:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return data

    async def _ensure_no_descendant_candidates(
        self,
        ref: AssetRef,
        binding: AssetTypeBinding[object],
        *,
        exclude_key: AssetKey,
    ) -> None:
        if await self._has_descendant_candidates(ref, binding, exclude_key=exclude_key):
            raise AIError(
                ErrorCode.ASSET_LAYOUT_CONFLICT,
                safe_details={"kind": ref.kind, "id_digest": canonical_sha256(ref.id)},
            )

    async def _has_descendant_candidates(
        self,
        ref: AssetRef,
        binding: AssetTypeBinding[object],
        *,
        exclude_key: AssetKey | None = None,
    ) -> bool:
        infos = await _load_all(self._store, kind=ref.kind)
        for candidate in _raw_candidates(binding, infos):
            if exclude_key is not None and candidate.info.key == exclude_key:
                continue
            if candidate.ref.id != ref.id and candidate.ref.id.startswith(ref.id + "/"):
                return True
        return False

    async def _compensate(
        self,
        ref: AssetRef,
        key: AssetKey,
        new_info: AssetInfo,
        old_info: AssetInfo | None,
        old_bytes: bytes | None,
    ) -> None:
        try:
            if old_info is None:
                await self._store.reset(key, expected_revision=new_info.revision)
            elif old_bytes is not None and (
                old_info.root_id == new_info.root_id and old_info.root_digest == new_info.root_digest
            ):
                await self._store.put(key, old_bytes, expected_revision=new_info.revision)
            else:
                await self._store.reset(key, expected_revision=new_info.revision)
        except Exception as error:
            _logger.exception(
                "asset typed write recovery failed: kind=%s id_digest=%s",
                ref.kind,
                canonical_sha256(ref.id),
            )
            raise AIError(ErrorCode.ASSET_RECOVERY_REQUIRED) from error
        _logger.debug("asset typed write recovery completed: kind=%s id_digest=%s", ref.kind, canonical_sha256(ref.id))

    def _select_write_variant(
        self,
        binding: AssetTypeBinding[object],
        probe: _Probe,
        requested: str | None,
    ) -> AssetVariantBinding[object]:
        if requested is not None:
            selected = binding.variant(requested)
        else:
            selected = None
        if not probe.candidates:
            return selected or binding.variant(binding.default_write_variant)
        existing = probe.candidates[0].variant
        if selected is not None and selected.name != existing.name:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        return existing

    def _scope(
        self,
        ref: AssetRef,
        variant: AssetVariantBinding[object],
        binding: AssetTypeBinding[object],
    ) -> AssetScope:
        reserved = tuple(
            item.layout.entry
            for item in binding.variants
            if isinstance(item.layout, DirectoryLayout)
        )
        if isinstance(variant.layout, SingleFileLayout):
            reserved = (variant.layout.scope_entry_path(variant.layout.entry_key(ref)),)
        return AssetScope(self._store, ref, variant, reserved)

    def _raise_for_owner(self, probe: _Probe, ref: AssetRef, *, for_write: bool = False) -> None:
        if not probe.owner_variants:
            return
        if len(probe.owner_variants) > 1:
            self._raise_layout_conflict(ref, tuple(_CandidatePlaceholder(name) for name in probe.owner_variants))
        raise AIError(
            ErrorCode.ASSET_LAYOUT_CONFLICT if for_write else ErrorCode.STORAGE_NOT_FOUND,
            safe_details={"owner_id_digest": canonical_sha256(probe.owner_id or "")},
        )

    def _raise_layout_conflict(self, ref: AssetRef, candidates: Sequence[object]) -> None:
        names = tuple(sorted(item.variant.name if isinstance(item, _Candidate) else item.name for item in candidates))
        _logger.warning(
            "asset layout conflict: kind=%s id_digest=%s variants=%s",
            ref.kind,
            canonical_sha256(ref.id),
            names,
        )
        raise AIError(
            ErrorCode.ASSET_LAYOUT_CONFLICT,
            safe_details={"kind": ref.kind, "id_digest": canonical_sha256(ref.id), "variants": list(names)},
        )


@dataclass(frozen=True, slots=True)
class _CandidatePlaceholder:
    name: str


async def _load_all(store: AssetStore, *, kind: str) -> tuple[AssetInfo, ...]:
    values: list[AssetInfo] = []
    cursor: str | None = None
    while True:
        page = await store.list_info(kind=kind, cursor=cursor, limit=200)
        values.extend(page.items)
        if page.next_cursor is None:
            return tuple(values)
        if page.next_cursor == cursor:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        cursor = page.next_cursor


def _raw_candidates(
    binding: AssetTypeBinding[object],
    infos: Sequence[AssetInfo],
) -> tuple[_Candidate, ...]:
    values: list[_Candidate] = []
    for info in infos:
        for variant in binding.variants:
            ref = variant.layout.match(info.key)
            if ref is None or (not binding.allow_nested_id and "/" in ref.id):
                continue
            values.append(_Candidate(variant, ref, info))
    return tuple(values)


def _discover(
    binding: AssetTypeBinding[object],
    infos: Sequence[AssetInfo],
) -> dict[str, AssetEntry]:
    candidates = _raw_candidates(binding, infos)
    directory_groups: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        if isinstance(candidate.variant.layout, DirectoryLayout):
            directory_groups.setdefault(candidate.ref.id, []).append(candidate)
    owned_roots: list[str] = []
    selected: dict[str, list[_Candidate]] = {}
    for root in sorted(directory_groups, key=lambda value: (len(value.split("/")), value)):
        if any(root == outer or root.startswith(outer + "/") for outer in owned_roots):
            continue
        owned_roots.append(root)
        selected.setdefault(root, []).extend(directory_groups[root])
    for candidate in candidates:
        if isinstance(candidate.variant.layout, DirectoryLayout):
            continue
        if any(candidate.info.key.id.startswith(root + "/") for root in owned_roots):
            continue
        selected.setdefault(candidate.ref.id, []).append(candidate)
    result: dict[str, AssetEntry] = {}
    for identifier, values in selected.items():
        variants = tuple(sorted({item.variant.name for item in values}))
        status = AssetDiscoveryStatus.RESOLVABLE if len(variants) == 1 else AssetDiscoveryStatus.CONFLICT
        result[identifier] = AssetEntry(AssetRef(binding.kind, identifier), status, variants)
    return result


def _probe_equal(left: _Probe, right: _Probe) -> bool:
    if left.owner_id != right.owner_id or left.owner_variants != right.owner_variants:
        return False
    return (
        _candidate_signatures(left.owners) == _candidate_signatures(right.owners)
        and _candidate_signatures(left.candidates) == _candidate_signatures(right.candidates)
    )


def _candidate_signatures(candidates: Sequence[_Candidate]) -> tuple[tuple[str, str, str, int, str], ...]:
    return tuple(
        sorted(
            (
                candidate.variant.name,
                candidate.info.root_id,
                candidate.info.root_digest,
                candidate.info.revision.value,
                candidate.info.etag,
            )
            for candidate in candidates
        )
    )


def _info_matches(left: AssetInfo, right: AssetInfo) -> bool:
    return (
        left.root_id == right.root_id
        and left.root_digest == right.root_digest
        and left.revision == right.revision
        and left.etag == right.etag
    )


def _decode_value(
    variant: AssetVariantBinding[object],
    ref: AssetRef,
    data: bytes,
    value_type: type[object],
) -> object:
    try:
        value = variant.codec.decode(data)
        _validate_exact_type(value_type, value)
        if variant.value_adapter is not None:
            value = variant.value_adapter.to_logical(ref.id, value)
            _validate_exact_type(value_type, value)
        return value
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _encode_value(
    variant: AssetVariantBinding[object],
    ref: AssetRef,
    value: object,
    value_type: type[object],
) -> bytes:
    try:
        storage_value = value
        if variant.value_adapter is not None:
            storage_value = variant.value_adapter.to_storage(ref.id, value)
            _validate_exact_type(value_type, storage_value)
        encoded = variant.codec.encode(storage_value)
        if not isinstance(encoded, bytes):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return encoded
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _validate_value_type(binding: AssetTypeBinding[object], value: object) -> None:
    _validate_exact_type(binding.value_type, value)


def _validate_exact_type(value_type: type[object], value: object) -> None:
    if type(value) is not value_type:
        raise AIError(ErrorCode.ASSET_CONFIG_TYPE_INVALID)


def _validate_identity(binding: AssetTypeBinding[object], ref: AssetRef, value: object) -> None:
    if binding.identity_validator is None:
        return
    try:
        valid = binding.identity_validator(ref, value)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error
    if not valid:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)


def _identity_matches(binding: AssetTypeBinding[object], ref: AssetRef, value: object) -> bool:
    if binding.identity_validator is None:
        return True
    try:
        return binding.identity_validator(ref, value)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH) from error


def _rename_suffix(source: AssetRef, variant: AssetVariantBinding[object], key: AssetKey) -> str:
    del variant
    if not key.id.startswith(source.id):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return key.id[len(source.id) :]


def _validate_limit(limit: int) -> None:
    if limit < 1 or limit > 200:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _validate_scope_path(path: str) -> str:
    if not path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if path.startswith("/") or PurePosixPath(path).is_absolute():
        raise AIError(ErrorCode.ASSET_PATH_ABSOLUTE)
    if ".." in path.split("/"):
        raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT)
    if "\\" in path or "\x00" in path or any(part in {"", "."} for part in path.split("/")):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return path


def _normalize_scope_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    normalized = prefix if prefix == "/" else prefix.removesuffix("/")
    return _validate_scope_path(normalized)


def _normalize_logical_prefix(
    kind: str,
    prefix: str | None,
    binding: AssetTypeBinding[object],
) -> str | None:
    if prefix is None:
        return None
    normalized = prefix.removesuffix("/")
    try:
        ref = AssetRef(kind, normalized)
    except ValueError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not binding.allow_nested_id and "/" in ref.id:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return ref.id


def _typed_cursor(
    kind: str,
    prefix: str | None,
    layout_digest: str,
    discovery_digest: str,
    last_id: str,
) -> str:
    raw = canonical_json_bytes(
        {
            "version": 1,
            "kind": kind,
            "prefix": prefix,
            "layout_digest": layout_digest,
            "discovery_digest": discovery_digest,
            "last_id": last_id,
        }
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _typed_cursor_start(
    cursor: str | None,
    kind: str,
    prefix: str | None,
    layout_digest: str,
    discovery_digest: str,
    values: Sequence[AssetEntry],
) -> int:
    if cursor is None:
        return 0
    try:
        payload = json.loads(base64.urlsafe_b64decode((cursor + "=" * (-len(cursor) % 4)).encode("ascii")))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("kind") != kind
            or payload.get("prefix") != prefix
            or payload.get("layout_digest") != layout_digest
            or payload.get("discovery_digest") != discovery_digest
            or not isinstance(payload.get("last_id"), str)
        ):
            raise ValueError
        last_id = payload["last_id"]
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    return next((index for index, item in enumerate(values) if item.ref.id > last_id), len(values))


def _scope_cursor(scope_digest: str, entry_path: str, prefix: str | None, last_path: str) -> str:
    raw = canonical_json_bytes(
        {
            "version": 1,
            "scope_digest": scope_digest,
            "entry_path": entry_path,
            "prefix": prefix,
            "last_path": last_path,
        }
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _scope_cursor_start(
    cursor: str | None,
    scope_digest: str,
    entry_path: str,
    prefix: str | None,
    values: Sequence[AssetResource],
) -> int:
    if cursor is None:
        return 0
    try:
        payload = json.loads(base64.urlsafe_b64decode((cursor + "=" * (-len(cursor) % 4)).encode("ascii")))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("scope_digest") != scope_digest
            or payload.get("entry_path") != entry_path
            or payload.get("prefix") != prefix
            or not isinstance(payload.get("last_path"), str)
        ):
            raise ValueError
        last_path = payload["last_path"]
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError):
        raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from None
    return next((index for index, item in enumerate(values) if item.path > last_path), len(values))


__all__ = ["AssetRepository", "AssetScope"]
