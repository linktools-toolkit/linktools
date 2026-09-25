#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill package resource sources."""

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol, cast, runtime_checkable

from ..asset import AssetStoreReader, AssetVersionRef
from ..core import (
    DEFAULT_DISCOVERY_POLICY,
    validate_logical_id,
)
from ..errors import AIError, ErrorCode
from ._resource_path import normalize_resource_path


@dataclass(frozen=True, slots=True)
class SkillResourceVersion:
    path: str
    asset: AssetVersionRef
    executable_bits: int = 0

    def __post_init__(self) -> None:
        _normalize_resource_path(self.path)
        if not isinstance(self.asset, AssetVersionRef):
            raise TypeError("skill resource asset must be AssetVersionRef")
        _validate_resource_mode(self.executable_bits)


@dataclass(frozen=True, slots=True)
class SkillSourceRef:
    source_id: str
    root: str
    resource_versions: tuple[SkillResourceVersion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            validate_logical_id(self.root)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        versions = tuple(self.resource_versions)
        if any(not isinstance(item, SkillResourceVersion) for item in versions):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        ordered = tuple(sorted(versions, key=lambda item: item.path))
        if ordered != versions or len({item.path for item in versions}) != len(versions):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    def with_asset_versions(
        self,
        resource_versions: Sequence[SkillResourceVersion],
    ) -> "SkillSourceRef":
        return SkillSourceRef(
            self.source_id,
            self.root,
            tuple(sorted(resource_versions, key=lambda item: item.path)),
        )


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

    async def inspect(self, source: SkillSourceRef) -> SkillResourceView: ...

    async def read(self, source: SkillSourceRef, path: str) -> bytes: ...


class LocalSkillResourceSource:
    def __init__(self, source_id: str, root: "str | Path") -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        self._id = source_id
        self._root = Path(root).expanduser().resolve()

    @property
    def id(self) -> str:
        return self._id

    def _root_ref(self, source: SkillSourceRef) -> str:
        if not isinstance(source, SkillSourceRef) or source.source_id != self._id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return _normalize_relative_path(source.root, field_name="skill root")

    async def inspect(self, source: SkillSourceRef) -> SkillResourceView:
        logical_root = self._root_ref(source)
        return await asyncio.to_thread(self._inspect_sync, logical_root)

    async def read(self, source: SkillSourceRef, path: str) -> bytes:
        logical_root = self._root_ref(source)
        relative = _normalize_resource_path(path)
        return await asyncio.to_thread(self._read_sync, logical_root, relative)

    async def resource_mode(self, source: SkillSourceRef, path: str) -> int:
        logical_root = self._root_ref(source)
        relative = _normalize_resource_path(path)
        package = await asyncio.to_thread(self._package_path, logical_root)
        resolved = await asyncio.to_thread(
            _resolve_contained_file,
            package,
            package / relative,
        )
        try:
            return (await asyncio.to_thread(resolved.stat)).st_mode & 0o111
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

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
    """Read Skill resources through version references captured in SkillSourceRef."""

    def __init__(self, source_id: str, asset_reader: AssetStoreReader) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        if not isinstance(asset_reader, AssetStoreReader):
            raise TypeError("asset_reader must provide AssetStoreReader operations")
        self._id = source_id
        self._asset_reader = asset_reader

    @property
    def id(self) -> str:
        return self._id

    @property
    def asset_reader(self) -> AssetStoreReader:
        return self._asset_reader

    def _binding(self, source: SkillSourceRef) -> SkillSourceRef:
        if not isinstance(source, SkillSourceRef) or source.source_id != self._id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return source

    async def inspect(self, source: SkillSourceRef) -> SkillResourceView:
        binding = self._binding(source)
        return SkillResourceView(
            SkillLocation("virtual", f"{self._id}/resources/{binding.root}"),
            tuple(item.path for item in binding.resource_versions),
        )

    async def resource_mode(self, source: SkillSourceRef, path: str) -> int:
        binding = self._binding(source)
        relative = _normalize_resource_path(path)
        for item in binding.resource_versions:
            if item.path == relative:
                return item.executable_bits
        raise AIError(ErrorCode.ASSET_NOT_FOUND)

    async def read(self, source: SkillSourceRef, path: str) -> bytes:
        binding = self._binding(source)
        relative = _normalize_resource_path(path)
        for item in binding.resource_versions:
            if item.path == relative:
                return (await self._asset_reader.read_versions((item.asset,)))[0]
        raise AIError(ErrorCode.ASSET_NOT_FOUND)


def _validate_resource_mode(mode: object) -> None:
    if (
        isinstance(mode, bool)
        or not isinstance(mode, int)
        or mode < 0
        or mode > 0o111
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


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
    try:
        return normalize_resource_path(path)
    except ValueError as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            "skill resource path is invalid",
        ) from error


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
    "LocalSkillResourceSource",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceVersion",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "normalize_skill_resource_path",
]
