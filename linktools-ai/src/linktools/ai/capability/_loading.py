#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Asset capture and loader context for capability groups."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from ..asset import (
    AssetInfo,
    AssetKey,
    AssetStore,
    AssetStoreReader,
    AssetVersionRef,
)
from ..core import ImmutableJsonMapping, JsonValue
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    MCPServerSpec,
    RepositoryInstructionDocument,
)
from ..storage import StorageRevision
from ._contribution import CapabilityContribution
from ._skill import SkillDefinition

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class _CapabilityAssetReader:
    _store: AssetStore = field(repr=False, compare=False)
    _revision: StorageRevision
    _versions: Mapping[AssetKey, AssetVersionRef] = field(repr=False, compare=False)
    _metadata: tuple[AssetInfo, ...] = field(repr=False, compare=False)

    async def current_revision(self) -> StorageRevision:
        if await self._store.current_revision() != self._revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        return self._revision

    async def get(self, key: AssetKey) -> "bytes | None":
        ref = self._versions.get(key)
        if ref is None:
            return None
        return (await self._store.read_versions((ref,)))[0]

    async def get_many(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[bytes | None, ...]":
        refs = tuple(self._versions.get(key) for key in keys)
        selected = tuple(ref for ref in refs if ref is not None)
        values = iter(await self._store.read_versions(selected))
        return tuple(next(values) if ref is not None else None for ref in refs)

    async def local_paths(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[Path | None, ...]":
        requested = tuple(dict.fromkeys(key for key in keys if key in self._versions))
        if not requested:
            return tuple(None for _ in keys)
        paths = await self._store.local_paths(requested)
        visible = tuple(
            key for key, path in zip(requested, paths, strict=True)
            if path is not None
        )
        if visible:
            try:
                current = await self._store.resolve_versions(visible)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND:
                    raise AIError(ErrorCode.SNAPSHOT_CONFLICT) from error
                raise
            if any(
                ref != self._versions[key]
                for key, ref in zip(visible, current, strict=True)
            ):
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        by_key = dict(zip(requested, paths, strict=True))
        return tuple(
            by_key.get(key) if key in self._versions else None
            for key in keys
        )

    async def capture_metadata(self) -> "tuple[AssetInfo, ...]":
        return self._metadata

    async def resolve_versions(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[AssetVersionRef, ...]":
        result: list[AssetVersionRef] = []
        for key in keys:
            ref = self._versions.get(key)
            if ref is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            result.append(ref)
        return tuple(result)

    async def read_versions(
        self,
        refs: Sequence[AssetVersionRef],
    ) -> "tuple[bytes, ...]":
        return await self._store.read_versions(refs)


@dataclass(frozen=True, slots=True)
class CapabilityLoadEntry:
    """Asset metadata captured at the start of one group capture."""

    key: AssetKey
    etag: str
    size: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, AssetKey)
            or not isinstance(self.etag, str)
            or len(self.etag) != 64
            or any(character not in "0123456789abcdef" for character in self.etag)
            or isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            metadata = ImmutableJsonMapping(self.metadata)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "metadata", metadata)


class CapabilityLoadContext:
    """One immutable Asset capture shared by every loader in a group."""

    def __init__(
        self,
        group_id: str,
        store: AssetStore,
        source_revision: StorageRevision,
        entries: Sequence[CapabilityLoadEntry],
        versions: Mapping[AssetKey, AssetVersionRef],
        asset_reader: AssetStoreReader,
    ) -> None:
        self._group_id = group_id
        self._store = store
        self._source_revision = source_revision
        self._entries = tuple(entries)
        self._by_key = {entry.key: entry for entry in self._entries}
        self._versions = dict(versions)
        self._asset_reader = asset_reader
        self._cache: dict[AssetKey, bytes] = {}
        if (
            not isinstance(group_id, str)
            or not group_id.strip()
            or len(self._by_key) != len(self._entries)
            or set(self._by_key) != set(self._versions)
            or any(
                self._versions[entry.key].key != entry.key
                or self._versions[entry.key].etag != entry.etag
                or self._versions[entry.key].size != entry.size
                for entry in self._entries
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @classmethod
    async def capture(
        cls,
        group_id: str,
        store: AssetStore,
    ) -> "CapabilityLoadContext":
        if not store.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        source_revision = await store.current_revision()
        if not isinstance(source_revision, StorageRevision):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata = await store.capture_metadata()
        versions = await store.resolve_versions(
            tuple(info.key for info in metadata)
        )
        if len(versions) != len(metadata) or any(
            not ref.matches_info(info)
            for info, ref in zip(metadata, versions, strict=True)
        ):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        if await store.current_revision() != source_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        version_by_key = {ref.key: ref for ref in versions}
        entries = tuple(
            CapabilityLoadEntry(
                info.key,
                info.etag,
                info.size,
                info.metadata,
            )
            for info in metadata
        )
        reader = _CapabilityAssetReader(
            store,
            source_revision,
            version_by_key,
            tuple(metadata),
        )
        return cls(
            group_id,
            store,
            source_revision,
            entries,
            version_by_key,
            reader,
        )

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def source_revision(self) -> StorageRevision:
        return self._source_revision

    @property
    def asset_reader(self) -> AssetStoreReader:
        return self._asset_reader

    def list(
        self,
        *,
        kind: "str | None" = None,
        prefix: "str | None" = None,
    ) -> "tuple[CapabilityLoadEntry, ...]":
        return tuple(
            entry
            for entry in self._entries
            if (kind is None or entry.key.kind == kind)
            and (prefix is None or entry.key.id.startswith(prefix))
        )

    def bind_versions(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[AssetVersionRef, ...]":
        result: list[AssetVersionRef] = []
        for key in keys:
            ref = self._versions.get(key)
            if ref is None:
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            result.append(ref)
        return tuple(result)

    async def read(self, key: AssetKey) -> bytes:
        return (await self.read_many((key,)))[0]

    async def read_many(self, keys: Sequence[AssetKey]) -> "tuple[bytes, ...]":
        requested = tuple(dict.fromkeys(keys))
        if any(key not in self._by_key for key in requested):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        pending = tuple(key for key in requested if key not in self._cache)
        if pending:
            refs = self.bind_versions(pending)
            try:
                values = await self._store.read_versions(refs)
            except AIError as error:
                if error.code in {
                    ErrorCode.ASSET_VERSION_NOT_FOUND,
                    ErrorCode.ASSET_VERSION_OWNER_UNKNOWN,
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                }:
                    raise AIError(ErrorCode.SNAPSHOT_CONFLICT) from error
                raise
            for key, data in zip(pending, values, strict=True):
                self._cache[key] = data
        return tuple(self._cache[key] for key in keys)

    async def verify(self) -> None:
        if await self._store.current_revision() != self._source_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)


class CapabilityLoader(Protocol[AppT]):
    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> (
        "Sequence[CapabilityContribution[AppT] | AgentSpec | SkillDefinition "
        "| MCPServerSpec | RepositoryInstructionDocument]"
    ): ...


__all__ = [
    "CapabilityLoadContext",
    "CapabilityLoadEntry",
    "CapabilityLoader",
]
