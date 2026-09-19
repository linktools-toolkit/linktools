#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill package resource sources."""

import asyncio
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from ..asset import AssetKey, AssetStore
from ..core import DEFAULT_DISCOVERY_POLICY, JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, StorageRevision, read_object


@dataclass(frozen=True, slots=True)
class SkillSourceRef:
    source_id: str
    root: str
    snapshot: "ObjectRef | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.snapshot is not None and not isinstance(self.snapshot, ObjectRef):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        object.__setattr__(
            self,
            "root",
            _normalize_relative_path(self.root, field_name="skill root"),
        )

    def with_snapshot(self, snapshot: ObjectRef) -> "SkillSourceRef":
        if not isinstance(snapshot, ObjectRef):
            raise TypeError("snapshot must be ObjectRef")
        return SkillSourceRef(self.source_id, self.root, snapshot)


@dataclass(frozen=True, slots=True)
class SkillLocation:
    kind: Literal["local", "virtual"]
    path: str

    def __post_init__(self) -> None:
        if self.kind not in {"local", "virtual"} or not isinstance(self.path, str) or not self.path:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "local" and not Path(self.path).is_absolute():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "virtual" and self.path.startswith("virtual:"):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    def display(self) -> str:
        return self.path if self.kind == "local" else f"virtual:{self.path}"


@dataclass(frozen=True, slots=True)
class SkillResourceView:
    location: SkillLocation
    resources: tuple[str, ...]

    def __post_init__(self) -> None:
        resources = tuple(sorted(self.resources))
        if resources != self.resources or len(resources) != len(set(resources)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for path in resources:
            _normalize_resource_path(path)


@runtime_checkable
class SkillResourceSource(Protocol):
    @property
    def id(self) -> str: ...

    async def inspect(self, root: str) -> SkillResourceView: ...

    async def read(self, root: str, path: str) -> bytes: ...


@runtime_checkable
class SnapshotSkillResourceSource(SkillResourceSource, Protocol):
    async def current_revision(self, root: str) -> StorageRevision: ...

    async def snapshot(
        self,
        root: str,
        *,
        expected_revision: StorageRevision,
        object_store: ObjectStore,
    ) -> ObjectRef: ...


class LocalSkillResourceSource:
    def __init__(self, source_id: str, root: "str | Path") -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        self._id = source_id
        self._root = Path(root).expanduser().resolve()

    @property
    def id(self) -> str:
        return self._id

    async def inspect(self, root: str) -> SkillResourceView:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        return await asyncio.to_thread(self._inspect_sync, logical_root)

    async def read(self, root: str, path: str) -> bytes:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        relative = _normalize_resource_path(path)
        return await asyncio.to_thread(self._read_sync, logical_root, relative)

    async def current_revision(self, root: str) -> StorageRevision:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        return await _skill_source_revision(self, logical_root)

    async def snapshot(
        self,
        root: str,
        *,
        expected_revision: StorageRevision,
        object_store: ObjectStore,
    ) -> ObjectRef:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        return await _snapshot_skill_source(
            self,
            logical_root,
            expected_revision=expected_revision,
            object_store=object_store,
        )

    def _inspect_sync(self, root: str) -> SkillResourceView:
        package = self._package_path(root)
        resources: list[str] = []
        for directory, directory_names, file_names in os.walk(package, followlinks=True):
            base = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if _skill_directory_is_discoverable(package, base, name)
            ]
            for name in file_names:
                path = base / name
                relative = path.relative_to(package).as_posix()
                if relative == "SKILL.md" or DEFAULT_DISCOVERY_POLICY.ignores(relative):
                    continue
                try:
                    _resolve_contained_file(package, path)
                except AIError as error:
                    if error.code in {
                        ErrorCode.ASSET_NOT_FOUND,
                        ErrorCode.ASSET_PATH_OUTSIDE_ROOT,
                    }:
                        continue
                    raise
                resources.append(_normalize_resource_path(relative))
        return SkillResourceView(
            SkillLocation("local", str(package)),
            tuple(sorted(resources)),
        )

    def _read_sync(self, root: str, path: str) -> bytes:
        package = self._package_path(root)
        candidate = package.joinpath(*PurePosixPath(path).parts)
        return _resolve_contained_file(package, candidate).read_bytes()

    def _package_path(self, root: str) -> Path:
        candidate = self._root.joinpath(*PurePosixPath(root).parts)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AIError(ErrorCode.ASSET_NOT_FOUND) from error
        if not resolved.is_dir():
            raise AIError(ErrorCode.ASSET_NOT_FOUND)
        return resolved


class AssetSkillResourceSource:
    def __init__(self, source_id: str, store: AssetStore) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        if not isinstance(store, AssetStore):
            raise TypeError("store must be AssetStore")
        self._id = source_id
        self._store = store

    @property
    def id(self) -> str:
        return self._id

    async def inspect(self, root: str) -> SkillResourceView:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        prefix = f"{logical_root}/"
        resources: list[str] = []
        cursor: str | None = None
        while True:
            page = await self._store.list_info(
                kind="skill",
                prefix=prefix,
                cursor=cursor,
                limit=200,
            )
            for info in page.items:
                relative = info.key.id[len(prefix) :]
                if relative == "SKILL.md" or DEFAULT_DISCOVERY_POLICY.ignores(relative):
                    continue
                try:
                    resources.append(_normalize_resource_path(relative))
                except AIError as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return SkillResourceView(
            SkillLocation("virtual", f"{self._id}/skills/{logical_root}"),
            tuple(sorted(resources)),
        )

    async def read(self, root: str, path: str) -> bytes:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        relative = _normalize_resource_path(path)
        value = await self._store.get(AssetKey("skill", f"{logical_root}/{relative}"))
        if value is None:
            raise AIError(ErrorCode.ASSET_NOT_FOUND)
        return bytes(value)


    async def current_revision(self, root: str) -> StorageRevision:
        _normalize_relative_path(root, field_name="skill root")
        return await self._store.current_revision()

    async def snapshot(
        self,
        root: str,
        *,
        expected_revision: StorageRevision,
        object_store: ObjectStore,
    ) -> ObjectRef:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        current = await self.current_revision(logical_root)
        if current != expected_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        return await _snapshot_skill_source(
            self,
            logical_root,
            expected_revision=expected_revision,
            object_store=object_store,
        )


async def _skill_source_revision(
    source: SkillResourceSource,
    root: str,
) -> StorageRevision:
    view = await source.inspect(root)
    if not isinstance(view, SkillResourceView):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    entries: list[dict[str, JsonValue]] = []
    for relative in view.resources:
        value = await source.read(root, relative)
        entries.append(
            {
                "path": relative,
                "digest": hashlib.sha256(value).hexdigest(),
                "size": len(value),
            }
        )
    return StorageRevision(
        hashlib.sha256(
            canonical_json_bytes(
                {
                    "version": 1,
                    "resources": entries,
                }
            )
        ).hexdigest()
    )


async def _snapshot_skill_source(
    source: SnapshotSkillResourceSource,
    root: str,
    *,
    expected_revision: StorageRevision,
    object_store: ObjectStore,
) -> ObjectRef:
    if not isinstance(expected_revision, StorageRevision):
        raise TypeError("expected_revision must be StorageRevision")
    before = await source.current_revision(root)
    if before != expected_revision:
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    view = await source.inspect(root)
    if not isinstance(view, SkillResourceView):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    entries: list[dict[str, JsonValue]] = []
    for relative in view.resources:
        value = await source.read(root, relative)
        digest = hashlib.sha256(value).hexdigest()
        key = f"v1/skill-source-content/{digest}"
        await _put_skill_snapshot_object(
            object_store,
            key,
            value,
            digest=digest,
        )
        entries.append(
            {
                "path": relative,
                "content": {
                    "key": key,
                    "digest": digest,
                    "size": len(value),
                },
            }
        )
    after = await source.current_revision(root)
    if after != expected_revision:
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    manifest: dict[str, JsonValue] = {
        "kind": "skill-source-snapshot",
        "format_version": 1,
        "source_id": source.id,
        "root": root,
        "revision": expected_revision.value,
        "sandbox_materialize": view.location.kind == "local",
        "resources": entries,
    }
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    key = f"v1/skill-source-snapshot/{digest}"
    await _put_skill_snapshot_object(
        object_store,
        key,
        payload,
        digest=digest,
    )
    return ObjectRef(object_store.store_id, key, digest, len(payload))


async def _put_skill_snapshot_object(
    object_store: ObjectStore,
    key: str,
    value: bytes,
    *,
    digest: str,
) -> None:
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

    async def chunks():
        yield value

    await object_store.put(
        key,
        chunks(),
        expected_size=len(value),
        expected_digest=digest,
    )


class FrozenSkillResourceSource:
    """Read one or more immutable Skill resource roots from snapshot objects."""

    def __init__(
        self,
        source_id: str,
        snapshots: Mapping[str, ObjectRef],
        object_store: ObjectStore,
    ) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        roots = {
            _normalize_relative_path(root, field_name="skill root"): ref
            for root, ref in snapshots.items()
        }
        if not roots or any(not isinstance(ref, ObjectRef) for ref in roots.values()):
            raise ValueError("skill snapshots must contain ObjectRef values")
        self._id = source_id
        self._snapshots = MappingProxyType(dict(sorted(roots.items())))
        self._object_store = object_store
        self._manifests: dict[str, Mapping[str, object]] = {}

    @property
    def id(self) -> str:
        return self._id

    async def inspect(self, root: str) -> SkillResourceView:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        manifest = await self._manifest(logical_root)
        entries = manifest["resources"]
        assert isinstance(entries, list)
        resources = tuple(
            cast(str, entry["path"])
            for entry in entries
            if isinstance(entry, Mapping)
        )
        if len(resources) != len(entries):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return SkillResourceView(
            SkillLocation(
                "virtual",
                f"{self._id}/skills/{logical_root}",
            ),
            resources,
        )

    async def sandbox_materialize(self, root: str) -> bool:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        manifest = await self._manifest(logical_root)
        return bool(manifest.get("sandbox_materialize", False))

    async def read(self, root: str, path: str) -> bytes:
        logical_root = _normalize_relative_path(root, field_name="skill root")
        relative = _normalize_resource_path(path)
        manifest = await self._manifest(logical_root)
        entries = manifest["resources"]
        assert isinstance(entries, list)
        for raw in entries:
            if not isinstance(raw, Mapping) or raw.get("path") != relative:
                continue
            content = raw.get("content")
            if not isinstance(content, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                key = str(content["key"])
                digest = str(content["digest"])
                size = int(content["size"])
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return await read_object(
                self._object_store,
                key,
                expected_digest=digest,
                expected_size=size,
            )
        raise AIError(ErrorCode.ASSET_NOT_FOUND)

    async def _manifest(self, root: str) -> Mapping[str, object]:
        cached = self._manifests.get(root)
        if cached is not None:
            return cached
        ref = self._snapshots.get(root)
        if ref is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                safe_details={
                    "source_id": self._id,
                    "root": root,
                },
            )
        payload = await read_object(
            self._object_store,
            ref.key,
            expected_digest=ref.digest,
            expected_size=ref.size,
        )
        try:
            manifest = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("kind") != "skill-source-snapshot"
            or manifest.get("format_version") != 1
            or manifest.get("source_id") != self._id
            or manifest.get("root") != root
            or not isinstance(manifest.get("revision"), str)
            or not isinstance(manifest.get("sandbox_materialize", False), bool)
            or not isinstance(manifest.get("resources"), list)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen: set[str] = set()
        previous: str | None = None
        for raw in cast(list[object], manifest["resources"]):
            if not isinstance(raw, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            relative = raw.get("path")
            content = raw.get("content")
            if (
                not isinstance(relative, str)
                or not isinstance(content, Mapping)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            relative = _normalize_resource_path(relative)
            if relative in seen or (
                previous is not None and relative < previous
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                key = content["key"]
                digest = content["digest"]
                size = content["size"]
            except KeyError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(relative)
            previous = relative
        frozen = MappingProxyType(dict(manifest))
        self._manifests[root] = frozen
        return frozen


class SkillSourceRegistry:
    def __init__(self, sources: Sequence[SkillResourceSource] = ()) -> None:
        values: dict[str, SkillResourceSource] = {}
        for source in sources:
            if not isinstance(source, SkillResourceSource):
                raise TypeError("sources must implement SkillResourceSource")
            if source.id in values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            values[source.id] = source
        self._sources: Mapping[str, SkillResourceSource] = MappingProxyType(values)

    def resolve(self, source_id: str) -> SkillResourceSource:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                safe_details={"source_id": source_id},
            ) from error

    def with_overrides(
        self,
        sources: Sequence[SkillResourceSource],
    ) -> "SkillSourceRegistry":
        values = dict(self._sources)
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, SkillResourceSource):
                raise TypeError("sources must implement SkillResourceSource")
            if source.id in seen:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            seen.add(source.id)
            values[source.id] = source
        return SkillSourceRegistry(tuple(values.values()))


def normalize_skill_resource_path(path: str) -> str:
    return _normalize_resource_path(path)


def _normalize_resource_path(path: str) -> str:
    return _normalize_relative_path(path, field_name="skill resource path")


def _normalize_relative_path(path: str, *, field_name: str) -> str:
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    if path.startswith("virtual:") or path.startswith("file:"):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    pure = PurePosixPath(path)
    if pure.is_absolute() or path.startswith("./") or path.endswith("/") or "//" in path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    if len(parts[0]) == 2 and parts[0][1] == ":":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    normalized = "/".join(parts)
    if normalized != path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field_name} is invalid")
    return normalized


def _skill_directory_is_discoverable(package: Path, base: Path, name: str) -> bool:
    path = base / name
    relative = path.relative_to(package).as_posix()
    if DEFAULT_DISCOVERY_POLICY.ignores(relative):
        return False
    try:
        target = path.resolve(strict=True)
        target.relative_to(package)
    except (OSError, RuntimeError, ValueError):
        return False
    if not target.is_dir():
        return False
    current = base
    while True:
        try:
            if current.resolve(strict=True) == target:
                return False
        except (OSError, RuntimeError):
            return False
        if current == package:
            return True
        current = current.parent


def _resolve_contained_file(root: Path, candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AIError(ErrorCode.ASSET_NOT_FOUND) from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT) from error
    if not resolved.is_file():
        raise AIError(ErrorCode.ASSET_NOT_FOUND)
    return resolved


__all__ = [
    "AssetSkillResourceSource",
    "FrozenSkillResourceSource",
    "LocalSkillResourceSource",
    "SkillLocation",
    "SkillResourceSource",
    "SnapshotSkillResourceSource",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "normalize_skill_resource_path",
]
