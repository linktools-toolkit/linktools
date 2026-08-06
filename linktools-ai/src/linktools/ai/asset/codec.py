#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen codecs for typed Asset values."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
from .model import AssetKey, AssetValue

TAsset = TypeVar("TAsset", bound=AssetValue)


@runtime_checkable
class AssetCodec(Protocol[TAsset]):
    @property
    def kind(self) -> str:
        ...

    @property
    def value_type(self) -> 'type[TAsset]':
        ...

    @property
    def fingerprint(self) -> str:
        ...

    def encode(self, value: TAsset) -> bytes:
        ...

    def decode(self, data: bytes) -> TAsset:
        ...

    def validate_key(self, key: AssetKey, value: TAsset) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise LinktoolsAIError(ErrorCode.ASSET_CONTENT_MISMATCH)


@dataclass(frozen=True, slots=True)
class AssetCodecManifestEntry:
    kind: str
    value_type: str
    fingerprint: str
    codec_version: int


@dataclass(frozen=True, slots=True)
class AssetCodecManifest:
    entries: "tuple[AssetCodecManifestEntry, ...]"
    digest: str


class AssetCodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[str, AssetCodec[AssetValue]] = {}
        self._manifest: AssetCodecManifest | None = None

    def register(self, codec: 'AssetCodec[TAsset]') -> None:
        if self._manifest is not None:
            raise LinktoolsAIError(ErrorCode.ASSET_CODEC_CONFLICT, "codec registry is frozen")
        if not codec.kind.strip() or not codec.fingerprint.strip():
            raise LinktoolsAIError(ErrorCode.ASSET_CODEC_CONFLICT)
        existing = self._codecs.get(codec.kind)
        if existing is not None:
            if existing.value_type is codec.value_type and existing.fingerprint == codec.fingerprint:
                return
            raise LinktoolsAIError(ErrorCode.ASSET_CODEC_CONFLICT)
        self._codecs[codec.kind] = codec

    def freeze(self) -> AssetCodecManifest:
        entries = tuple(
            AssetCodecManifestEntry(
                kind=kind,
                value_type=f"{codec.value_type.__module__}.{codec.value_type.__qualname__}",
                fingerprint=codec.fingerprint,
                codec_version=1,
            )
            for kind, codec in sorted(self._codecs.items())
        )
        digest = canonical_sha256(
            {
                "entries": [
                    {
                        "kind": entry.kind,
                        "value_type": entry.value_type,
                        "fingerprint": entry.fingerprint,
                        "codec_version": entry.codec_version,
                    }
                    for entry in entries
                ]
            }
        )
        self._manifest = AssetCodecManifest(entries, digest)
        return self._manifest

    def resolve(self, kind: str, expected: 'type[TAsset]') -> 'AssetCodec[TAsset]':
        codec = self._codecs.get(kind)
        if codec is None:
            raise LinktoolsAIError(ErrorCode.ASSET_CODEC_UNKNOWN)
        if codec.value_type is not expected:
            raise LinktoolsAIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return codec

    def manifest(self) -> AssetCodecManifest:
        if self._manifest is None:
            return self.freeze()
        return self._manifest


__all__ = ["AssetCodec", "AssetCodecManifest", "AssetCodecManifestEntry", "AssetCodecRegistry"]
