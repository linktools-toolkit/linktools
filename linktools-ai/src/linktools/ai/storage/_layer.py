#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered read layers and write visibility."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic

from ._contracts import InfoT, KeyT, ReadableStorageBackend, ValueT


class LayerRefreshPolicy(StrEnum):
    STATIC = "STATIC"
    REVISIONED = "REVISIONED"
    ALWAYS = "ALWAYS"


class StorageWriteVisibility(StrEnum):
    READABLE = "READABLE"
    EXTERNAL = "EXTERNAL"


@dataclass(frozen=True, slots=True)
class StorageLayer(Generic[KeyT, ValueT, InfoT]):
    id: str
    backend: "ReadableStorageBackend[KeyT, ValueT, InfoT]"
    refresh: LayerRefreshPolicy = LayerRefreshPolicy.STATIC

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("layer id must not be empty")


__all__ = ["LayerRefreshPolicy", "StorageLayer", "StorageWriteVisibility"]
