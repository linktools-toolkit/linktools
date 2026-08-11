#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical asset bindings and immutable discovery registry."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._domain import AssetInfo, AssetKey

if TYPE_CHECKING:
    from ._repository import AssetScope

LogicalT = TypeVar("LogicalT")
_SNAPSHOT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AssetRef:
    """Identify one logical asset without selecting its physical layout."""

    kind: str
    id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or not isinstance(self.id, str)
            or not self.kind
            or not self.id
            or "\x00" in self.kind
            or "\x00" in self.id
        ):
            raise ValueError("asset reference is invalid")
        if "\\" in self.id or self.id.startswith("/") or self.id.endswith("/"):
            raise ValueError("asset reference is invalid")
        parts = self.id.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise ValueError("asset reference is invalid")
        try:
            AssetKey(self.kind, self.id)
        except ValueError as error:
            raise ValueError("asset reference is invalid") from error


@dataclass(frozen=True, slots=True)
class SingleFileLayout:
    """Map a logical id to one file by appending a suffix."""

    suffix: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.suffix, str)
            or "\x00" in self.suffix
            or "/" in self.suffix
            or "\\" in self.suffix
        ):
            raise ValueError("single-file suffix is invalid")

    def entry_key(self, ref: AssetRef) -> AssetKey:
        return AssetKey(ref.kind, ref.id + self.suffix)

    def match(self, key: AssetKey) -> AssetRef | None:
        if not key.id.endswith(self.suffix):
            return None
        identifier = key.id if not self.suffix else key.id[: -len(self.suffix)]
        try:
            return AssetRef(key.kind, identifier)
        except ValueError:
            return None

    def descriptor(self) -> dict[str, JsonValue]:
        return {"type": "single-file", "suffix": self.suffix}

    def scope_entry_path(self, key: AssetKey) -> str:
        return key.id.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class DirectoryLayout:
    """Map a logical id to a named entry file and its resource subtree."""

    entry: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.entry, str)
            or not self.entry
            or self.entry in {".", ".."}
            or "\x00" in self.entry
            or "/" in self.entry
            or "\\" in self.entry
        ):
            raise ValueError("directory entry is invalid")

    def entry_key(self, ref: AssetRef) -> AssetKey:
        return AssetKey(ref.kind, f"{ref.id}/{self.entry}")

    def match(self, key: AssetKey) -> AssetRef | None:
        marker = f"/{self.entry}"
        if not key.id.endswith(marker):
            return None
        identifier = key.id[: -len(marker)]
        try:
            return AssetRef(key.kind, identifier)
        except ValueError:
            return None

    def descriptor(self) -> dict[str, JsonValue]:
        return {"type": "directory", "entry": self.entry}

    def scope_entry_path(self, key: AssetKey) -> str:
        del key
        return self.entry


@runtime_checkable
class AssetCodec(Protocol[LogicalT]):
    def encode(self, value: LogicalT) -> bytes: ...

    def decode(self, data: bytes) -> LogicalT: ...


@runtime_checkable
class AssetValueAdapter(Protocol[LogicalT]):
    def to_logical(self, logical_id: str, value: LogicalT) -> LogicalT: ...

    def to_storage(self, logical_id: str, value: LogicalT) -> LogicalT: ...


@dataclass(frozen=True, slots=True)
class AssetVariantBinding(Generic[LogicalT]):
    """Bind one physical layout to a codec and optional representation adapter."""

    name: str
    layout: SingleFileLayout | DirectoryLayout
    codec: AssetCodec[LogicalT]
    fingerprint: str
    codec_version: int
    value_adapter: AssetValueAdapter[LogicalT] | None = None
    adapter_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.layout) not in (SingleFileLayout, DirectoryLayout):
            raise TypeError("asset variant layout must be a supported concrete layout")
        if (
            not isinstance(self.name, str)
            or not isinstance(self.fingerprint, str)
            or not self.name
            or not self.fingerprint
            or not isinstance(self.codec_version, int)
            or isinstance(self.codec_version, bool)
            or self.codec_version < 1
        ):
            raise ValueError("asset variant binding is invalid")
        if (self.value_adapter is None) != (self.adapter_fingerprint is None):
            raise ValueError("asset variant adapter fingerprint is incomplete")
        if self.adapter_fingerprint is not None and (
            not isinstance(self.adapter_fingerprint, str) or not self.adapter_fingerprint
        ):
            raise ValueError("asset variant adapter fingerprint is invalid")


@dataclass(frozen=True, slots=True)
class AssetTypeBinding(Generic[LogicalT]):
    """Declare every supported representation of one logical asset kind."""

    kind: str
    value_type: type[LogicalT]
    variants: tuple[AssetVariantBinding[LogicalT], ...]
    default_write_variant: str
    identity_validator: Callable[[AssetRef, LogicalT], bool] | None = None
    identity_fingerprint: str | None = None
    allow_nested_id: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or not isinstance(self.default_write_variant, str)
            or not isinstance(self.allow_nested_id, bool)
        ):
            raise TypeError("asset type binding is invalid")
        try:
            AssetKey(self.kind, "asset")
        except ValueError as error:
            raise ValueError("asset type binding kind is invalid") from error
        variants = tuple(self.variants)
        object.__setattr__(self, "variants", variants)
        if not _is_concrete_value_type(self.value_type) or not variants or not self.default_write_variant:
            raise ValueError("asset type binding is incomplete")
        names = tuple(variant.name for variant in variants)
        if len(set(names)) != len(names) or self.default_write_variant not in names:
            raise ValueError("asset type binding variants are invalid")
        if (self.identity_validator is None) != (self.identity_fingerprint is None):
            raise ValueError("asset identity fingerprint is incomplete")
        if self.identity_fingerprint is not None and (
            not isinstance(self.identity_fingerprint, str) or not self.identity_fingerprint
        ):
            raise ValueError("asset identity fingerprint is invalid")

    def variant(self, name: str) -> AssetVariantBinding[LogicalT]:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise AIError(ErrorCode.ASSET_LAYOUT_UNKNOWN)


class AssetDiscoveryStatus(StrEnum):
    RESOLVABLE = "RESOLVABLE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class AssetEntry:
    """Describe a discovered logical asset without decoding its content."""

    ref: AssetRef
    status: AssetDiscoveryStatus
    variants: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status is AssetDiscoveryStatus.RESOLVABLE and len(self.variants) != 1:
            raise ValueError("resolvable asset must have one variant")
        if self.status is AssetDiscoveryStatus.CONFLICT and len(self.variants) < 2:
            raise ValueError("conflicting asset must have multiple variants")
        if tuple(sorted(self.variants)) != self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("asset variants must be unique and sorted")


@dataclass(frozen=True, slots=True)
class AssetResource:
    """Expose one raw file through a resolved asset scope."""

    path: str
    info: AssetInfo
    is_entry: bool


@dataclass(frozen=True, slots=True)
class ResolvedAsset(Generic[LogicalT]):
    """Return a decoded logical value together with its raw entry and scope."""

    ref: AssetRef
    variant: str
    spec: LogicalT
    entry: AssetInfo
    scope: "AssetScope"


class AssetTypeRegistry:
    """Register logical asset bindings before producing an immutable snapshot."""

    def __init__(self) -> None:
        self._bindings: dict[str, AssetTypeBinding[object]] = {}
        self._snapshot: AssetTypeRegistrySnapshot | None = None

    @property
    def frozen(self) -> bool:
        return self._snapshot is not None

    def register(self, binding: "AssetTypeBinding[object]") -> None:
        if self._snapshot is not None:
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT, "asset type registry is frozen")
        if binding.kind in self._bindings:
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT, "asset kind is already registered")
        _validate_layouts(binding)
        self._bindings[binding.kind] = binding

    def manifest_entries(self) -> tuple[dict[str, JsonValue], ...]:
        return _manifest_entries(tuple(self._bindings.values()))

    def freeze(self) -> "AssetTypeRegistrySnapshot":
        if self._snapshot is not None:
            return self._snapshot
        entries = self.manifest_entries()
        layout_digest = canonical_sha256(
            [
                {
                    "kind": entry["kind"],
                    "variants": [
                        {"name": variant["name"], "layout": variant["layout"]}
                        for variant in cast(list[dict[str, JsonValue]], entry["variants"])
                    ],
                    "default_write_variant": entry["default_write_variant"],
                    "allow_nested_id": entry["allow_nested_id"],
                }
                for entry in entries
            ]
        )
        binding_digest = canonical_sha256(entries)
        self._snapshot = AssetTypeRegistrySnapshot._create(
            MappingProxyType(dict(self._bindings)),
            layout_digest,
            binding_digest,
        )
        return self._snapshot


@dataclass(frozen=True, slots=True, init=False)
class AssetTypeRegistrySnapshot:
    """Immutable registry state consumed by an AssetRepository."""

    _bindings: "Mapping[str, AssetTypeBinding[object]]"
    layout_digest: str
    binding_digest: str

    def __init__(
        self,
        bindings: "Mapping[str, AssetTypeBinding[object]]",
        layout_digest: str,
        binding_digest: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise TypeError("asset registry snapshots are factory-issued")
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "layout_digest", layout_digest)
        object.__setattr__(self, "binding_digest", binding_digest)

    @classmethod
    def _create(
        cls,
        bindings: "Mapping[str, AssetTypeBinding[object]]",
        layout_digest: str,
        binding_digest: str,
    ) -> "AssetTypeRegistrySnapshot":
        return cls(bindings, layout_digest, binding_digest, _token=_SNAPSHOT_TOKEN)

    @property
    def frozen(self) -> bool:
        return True

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def binding(self, kind: str) -> "AssetTypeBinding[object]":
        try:
            return self._bindings[kind]
        except KeyError as error:
            raise AIError(ErrorCode.ASSET_CODEC_UNKNOWN) from error

    def manifest_entries(self) -> tuple[dict[str, JsonValue], ...]:
        return _manifest_entries(tuple(self._bindings.values()))


def _validate_layouts(binding: AssetTypeBinding[object]) -> None:
    variants = binding.variants
    descriptors = [canonical_sha256(variant.layout.descriptor()) for variant in variants]
    if len(set(descriptors)) != len(descriptors):
        raise AIError(ErrorCode.ASSET_CODEC_CONFLICT, "asset layout is registered more than once")
    single = [variant.layout.suffix for variant in variants if isinstance(variant.layout, SingleFileLayout)]
    for index, suffix in enumerate(single):
        if any(
            index != other_index and (suffix.endswith(other) or other.endswith(suffix))
            for other_index, other in enumerate(single)
        ):
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT, "single-file layouts overlap")


def _is_concrete_value_type(value_type: object) -> bool:
    return (
        value_type is not Any
        and isinstance(value_type, type)
        and not inspect.isabstract(value_type)
        and not _is_protocol_class(value_type)
    )


def _is_protocol_class(value_type: type[object]) -> bool:
    return value_type is Protocol or Protocol in value_type.__bases__


def _manifest_entries(bindings: tuple[AssetTypeBinding[object], ...]) -> tuple[dict[str, JsonValue], ...]:
    entries: list[dict[str, JsonValue]] = []
    for binding in sorted(bindings, key=lambda item: item.kind):
        variants: list[dict[str, JsonValue]] = []
        for variant in sorted(binding.variants, key=lambda item: item.name):
            variants.append(
                {
                    "name": variant.name,
                    "layout": variant.layout.descriptor(),
                    "codec_fingerprint": variant.fingerprint,
                    "codec_version": variant.codec_version,
                    "adapter_fingerprint": variant.adapter_fingerprint,
                }
            )
        qualified = f"{binding.value_type.__module__}.{binding.value_type.__qualname__}"
        entries.append(
            {
                "kind": binding.kind,
                "value_type": qualified,
                "allow_nested_id": binding.allow_nested_id,
                "identity_fingerprint": binding.identity_fingerprint,
                "default_write_variant": binding.default_write_variant,
                "variants": variants,
            }
        )
    return tuple(entries)


__all__ = [
    "AssetCodec",
    "AssetDiscoveryStatus",
    "AssetEntry",
    "AssetRef",
    "AssetResource",
    "AssetTypeBinding",
    "AssetTypeRegistry",
    "AssetTypeRegistrySnapshot",
    "AssetValueAdapter",
    "AssetVariantBinding",
    "DirectoryLayout",
    "ResolvedAsset",
    "SingleFileLayout",
]
