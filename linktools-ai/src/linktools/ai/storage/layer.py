#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered read layers and write visibility."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic

from .contracts import InfoT, KeyT, ReadableMetadataBackend, StoreRevisionT, ValueT


class LayerRefreshPolicy(StrEnum):
    STATIC = "STATIC"
    REVISIONED = "REVISIONED"
    ALWAYS = "ALWAYS"


class StorageWriteVisibility(StrEnum):
    READABLE = "READABLE"
    EXTERNAL = "EXTERNAL"


@dataclass(frozen=True, slots=True)
class StorageLayer(Generic[KeyT, ValueT, InfoT, StoreRevisionT]):
    id: str
    backend: "ReadableMetadataBackend[KeyT, ValueT, InfoT, StoreRevisionT]"
    refresh: LayerRefreshPolicy = LayerRefreshPolicy.STATIC

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("layer id must not be empty")


__all__ = ["LayerRefreshPolicy", "StorageLayer", "StorageWriteVisibility"]
