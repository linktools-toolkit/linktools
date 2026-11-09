#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill package resource sources."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from ..asset import AssetStoreReader, AssetVersionRef, RetainedAssetReader
from ..core import (
    DEFAULT_DISCOVERY_POLICY,
    JsonValue,
    validate_logical_id,
)
from ..errors import AIError, ErrorCode
from ..spec import CapabilityRef
from ._resource_path import SKILL_DECLARATION_FILES, require_resource_path

@dataclass(frozen=True, slots=True)
class SkillResource:
    path: str
    asset: AssetVersionRef
    executable_bits: int = 0

    def __post_init__(self) -> None:
        _require_resource_path(self.path)
        if not isinstance(self.asset, AssetVersionRef):
            raise TypeError("skill resource asset must be AssetVersionRef")
        _validate_resource_mode(self.executable_bits)

    def to_payload(self) -> dict[str, JsonValue]:
        return {"path": self.path, "asset": self.asset.to_payload(), "executable_bits": self.executable_bits}

    @classmethod
    def from_payload(cls, value: object) -> "SkillResource":
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return cls(value["path"], AssetVersionRef.from_payload(value["asset"]), value.get("executable_bits", 0))
        except (KeyError, TypeError, ValueError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(frozen=True, slots=True)
class SkillSourceRef:
    source_id: str
    root: str
    resources: tuple[SkillResource, ...] = ()
    declaration: SkillResource | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or not self.source_id.strip()
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            validate_logical_id(self.root)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if self.declaration is not None and not isinstance(self.declaration, SkillResource):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        resources = tuple(self.resources)
        if any(not isinstance(item, SkillResource) for item in resources):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        ordered = tuple(sorted(resources, key=lambda item: item.path))
        if ordered != resources or len({item.path for item in resources}) != len(resources):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    def to_payload(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"source_id": self.source_id, "root": self.root}
        if self.resources:
            result["resources"] = [item.to_payload() for item in self.resources]
        if self.declaration is not None:
            result["declaration"] = self.declaration.to_payload()
        return result

    @classmethod
    def from_payload(cls, value: object) -> "SkillSourceRef":
        if not isinstance(value, Mapping) or not isinstance(value.get("resources", []), list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return cls(value["source_id"], value["root"],
                       tuple(sorted((SkillResource.from_payload(item) for item in value.get("resources", [])), key=lambda item: item.path)),
                       None if value.get("declaration") is None else SkillResource.from_payload(value["declaration"]))
        except (KeyError, TypeError, ValueError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(frozen=True, slots=True)
class SkillInput:
    """An admitted required Skill root; its bytes remain with the Asset owner."""

    skill: CapabilityRef
    source: SkillSourceRef

    def __post_init__(self) -> None:
        if (not isinstance(self.skill, CapabilityRef) or self.skill.kind != "skill"
                or not isinstance(self.source, SkillSourceRef) or self.source.declaration is None
                or self.source.resources):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                          safe_details={"reason": "required_skill_identity_unavailable"})

        declaration = self.source.declaration
        if (self.source.root != self.skill.id or declaration.path not in SKILL_DECLARATION_FILES
                or declaration.asset.key.kind != "skill"
                or declaration.asset.key.id != f"{self.source.root}/{declaration.path}"):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                          safe_details={"reason": "required_skill_source_mismatch"})

    def to_payload(self) -> dict[str, JsonValue]:
        return {"version": 1, "skill": self.skill.to_payload(), "source": self.source.to_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "SkillInput":
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if type(value.get("version")) is not int or value["version"] != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        try:
            return cls(CapabilityRef.from_payload(value["skill"]), SkillSourceRef.from_payload(value["source"]))
        except (KeyError, TypeError, ValueError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def read(self, sources: "SkillSourceRegistry") -> bytes:
        source = sources.resolve(self.source.source_id)
        if not isinstance(source, AssetSkillSource) or not isinstance(source.asset_reader, RetainedAssetReader):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                          safe_details={"reason": "required_skill_history_unavailable"})
        declaration = self.source.declaration
        assert declaration is not None
        data, = await source.asset_reader.require_retained_versions((declaration.asset,))
        return data


SKILL_READ_METADATA_KEY = "linktools.skill_read"
REQUIRED_SKILL_READS_METADATA_KEY = "linktools.required_skill_reads"


@dataclass(frozen=True, slots=True)
class SkillReadReceipt:
    """Successful delivery and its observed resource identity, not an allowlist."""

    skill: CapabilityRef
    content_digest: str
    source_id: str | None = None
    path: str | None = None
    asset: AssetVersionRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill, CapabilityRef) or self.skill.kind != "skill":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (not isinstance(self.content_digest, str) or len(self.content_digest) != 64
                or any(char not in "0123456789abcdef" for char in self.content_digest)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.source_id is not None and (not isinstance(self.source_id, str) or not self.source_id.strip()):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.path is not None:
            _require_resource_path(self.path)
        if self.asset is not None and (not isinstance(self.asset, AssetVersionRef) or self.source_id is None or self.path is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def identity_verified(self) -> bool:
        return self.asset is not None

    def to_payload(self) -> dict[str, JsonValue]:
        return {"version": 1, "skill": self.skill.to_payload(), "content_digest": self.content_digest,
                "source_id": self.source_id, "path": self.path,
                "asset": None if self.asset is None else self.asset.to_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "SkillReadReceipt":
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if type(value.get("version")) is not int or value["version"] != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        try:
            return cls(CapabilityRef.from_payload(value["skill"]), value["content_digest"],
                       value["source_id"], value["path"],
                       None if value["asset"] is None else AssetVersionRef.from_payload(value["asset"]))
        except (KeyError, TypeError, ValueError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


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
class SkillSourceView:
    location: SkillLocation
    resources: tuple[str, ...]

    def __post_init__(self) -> None:
        resources = tuple(sorted(self.resources))
        if resources != self.resources or len(resources) != len(set(resources)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for path in resources:
            _require_resource_path(path)


@runtime_checkable
class SkillSource(Protocol):
    @property
    def source_id(self) -> str: ...

    async def inspect(self, source: SkillSourceRef) -> SkillSourceView: ...

    async def read(self, source: SkillSourceRef, path: str) -> bytes: ...


class AssetSkillSource:
    """Read Skill resources using explicit Asset version references."""

    def __init__(
        self, source_id: str, asset_reader: AssetStoreReader,
    ) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("skill source id must be non-empty")
        if not isinstance(asset_reader, AssetStoreReader):
            raise TypeError("asset_reader must provide AssetStoreReader operations")
        self._source_id = source_id
        self._asset_reader = asset_reader

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def asset_reader(self) -> AssetStoreReader:
        return self._asset_reader

    def _binding(self, source: SkillSourceRef) -> SkillSourceRef:
        if (
            not isinstance(source, SkillSourceRef)
            or source.source_id != self._source_id
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return source

    async def inspect(self, source: SkillSourceRef) -> SkillSourceView:
        binding = self._binding(source)
        versioned_resources = tuple(
            item.path for item in binding.resources
        )
        resources = tuple(
            path
            for path in versioned_resources
            if not DEFAULT_DISCOVERY_POLICY.ignores(path)
        )
        location = SkillLocation(
            "virtual",
            f"{self._source_id}/resources/{binding.root}",
        )
        if binding.resources:
            paths = await self._asset_reader.local_paths(
                tuple(item.asset.key for item in binding.resources)
            )
            if all(path is not None for path in paths):
                package = await asyncio.to_thread(
                    _resolve_local_skill_package,
                    versioned_resources,
                    tuple(path for path in paths if path is not None),
                )
                if package is not None:
                    location = SkillLocation("local", str(package))
        return SkillSourceView(location, resources)

    async def read(self, source: SkillSourceRef, path: str) -> bytes:
        binding = self._binding(source)
        relative = _require_resource_path(path)
        selected = (*binding.resources, *((binding.declaration,) if binding.declaration is not None else ()))
        for item in selected:
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
        or mode & ~0o111
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class SkillSourceRegistry:
    def __init__(self, sources: Sequence[SkillSource] = ()) -> None:
        values: dict[str, SkillSource] = {}
        for source in sources:
            if not isinstance(source, SkillSource):
                raise TypeError("sources must implement SkillSource")
            if source.source_id in values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            values[source.source_id] = source
        self._sources: Mapping[str, SkillSource] = MappingProxyType(values)

    def resolve(self, source_id: str) -> SkillSource:
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


__all__ = [
    "AssetSkillSource",
    "SkillLocation",
    "SkillSource",
    "SkillResource",
    "SkillInput",
    "SkillReadReceipt",
    "SKILL_READ_METADATA_KEY",
    "REQUIRED_SKILL_READS_METADATA_KEY",
    "SkillSourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "require_skill_resource_path",
]
