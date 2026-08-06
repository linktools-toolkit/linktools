#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text asset source and decoder contracts."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

TAsset = TypeVar("TAsset")


@runtime_checkable
class AssetSource(Protocol):
    async def list_ids(self, suffix: str) -> "tuple[str, ...]": ...

    async def read(self, path: str) -> str: ...

    def identity(self, raw: str) -> str: ...


@runtime_checkable
class AssetCodec(Generic[TAsset], Protocol):
    def decode(self, item_id: str, raw: str) -> TAsset: ...


__all__ = ["AssetCodec", "AssetSource", "TAsset"]
