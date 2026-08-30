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

from ..asset import AssetKey, AssetStore
from ..core import DEFAULT_DISCOVERY_POLICY
from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class SkillSourceRef:
    source_id: str
    root: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        object.__setattr__(self, "root", _normalize_relative_path(self.root, field_name="skill root"))


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
    "LocalSkillResourceSource",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "normalize_skill_resource_path",
]
