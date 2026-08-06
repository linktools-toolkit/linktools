#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asset source and codec contracts."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..foundation.errors import (
    InvalidAssetError,
    AssetConflictError,
    AssetError,
    AssetNotFoundError,
    AssetParseError,
)

T = TypeVar("T")


@runtime_checkable
class AssetSource(Protocol):
    async def list_ids(self, suffix: str) -> "tuple[str, ...]": ...

    async def read(self, path: str) -> str: ...

    def identity(self, raw: str) -> str: ...


@runtime_checkable
class AssetCodec(Protocol, Generic[T]):
    def decode(self, item_id: str, raw: str) -> T: ...


__all__ = [
    "InvalidAssetError",
    "AssetCodec",
    "AssetConflictError",
    "AssetError",
    "AssetNotFoundError",
    "AssetParseError",
    "AssetSource",
]
