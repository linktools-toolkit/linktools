#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill package resource sources."""

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from ..asset import AssetStoreReader, AssetVersionRef
from ..core import (
    DEFAULT_DISCOVERY_POLICY,
    validate_logical_id,
)
from ..errors import AIError, ErrorCode
from ._resource_path import require_resource_path


@dataclass(frozen=True, slots=True)
class SkillResourceVersion:
    path: str
    asset: AssetVersionRef
    executable_bits: int = 0

    def __post_init__(self) -> None:
        _require_resource_path(self.path)
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
            _require_resource_path(path)


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
        return source.root

    async def inspect(self, source: SkillSourceRef) -> SkillResourceView:
        logical_root = self._root_ref(source)
        return await asyncio.to_thread(self._inspect_sync, logical_root)

    async def read(self, source: SkillSourceRef, path: str) -> bytes:
        logical_root = self._root_ref(source)
        relative = _require_resource_path(path)
        return await asyncio.to_thread(self._read_sync, logical_root, relative)

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
                resources.append(_require_resource_path(relative))
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
        resources = tuple(item.path for item in binding.resource_versions)
        location = SkillLocation(
            "virtual",
            f"{self._id}/resources/{binding.root}",
        )
        if binding.resource_versions:
            paths = await self._asset_reader.local_paths(
                tuple(item.asset.key for item in binding.resource_versions)
            )
            if all(path is not None for path in paths):
                package = _resolve_local_skill_package(
                    resources,
                    tuple(path for path in paths if path is not None),
                )
                if package is not None:
                    location = SkillLocation("local", str(package))
        return SkillResourceView(location, resources)

    async def read(self, source: SkillSourceRef, path: str) -> bytes:
        binding = self._binding(source)
        relative = _require_resource_path(path)
        for item in binding.resource_versions:
            if item.path == relative:
                return (await self._asset_reader.read_versions((item.asset,)))[0]
        raise AIError(ErrorCode.ASSET_NOT_FOUND)



def _resolve_local_skill_package(
    relatives: Sequence[str],
    paths: Sequence[Path],
) -> "Path | None":
    if not relatives or len(relatives) != len(paths):
        return None
    package = paths[0]
    for _part in PurePosixPath(relatives[0]).parts:
        package = package.parent
    for relative, path in zip(relatives, paths, strict=True):
        expected = package.joinpath(*PurePosixPath(relative).parts)
        if path != expected:
            return None
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(package.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        if not resolved.is_file():
            return None
    try:
        resolved_package = package.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved_package if resolved_package.is_dir() else None

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

def require_skill_resource_path(path: str) -> str:
    return _require_resource_path(path)


def _require_resource_path(path: str) -> str:
    try:
        return require_resource_path(path)
    except ValueError as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            "skill resource path is invalid",
        ) from error


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
    "require_skill_resource_path",
]
