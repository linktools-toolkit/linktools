#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy content value and identity metadata for asset loaders."""

import hashlib
from dataclasses import dataclass
from typing import Protocol

from ..core import AssetConflictError


@dataclass(frozen=True, slots=True)
class AssetContentInfo:
    path: str
    kind: str
    version: int
    etag: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class AssetContent:
    info: AssetContentInfo
    content: bytes

    def validate_etag(self) -> None:
        if self.info.etag != compute_asset_etag(self.content):
            raise AssetConflictError(
                f"asset etag mismatch for {self.info.path!r}: etag must equal sha256(content)"
            )


class AssetContentSource(Protocol):
    async def get(self, path: str) -> "AssetContent | None": ...

    async def list_info(self) -> "tuple[AssetContentInfo, ...]": ...


def compute_asset_etag(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = ["AssetContent", "AssetContentInfo", "AssetContentSource", "compute_asset_etag"]
