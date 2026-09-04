#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw Asset byte storage and backend contracts."""

from ..errors import AssetError
from ._backend import InMemoryAssetBackend
from ._config import StrictConfigReader, resolved_name
from ._directory import (
    AssetPathAdapter,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
    directory_root,
)
from ._domain import AssetBackend, AssetInfo, AssetKey, AssetRoot, WritableAssetBackend
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._object import AssetObjectKeyFactory
from ._sql import SqlAssetBackend, build_asset_sql_metadata
from ._store import AssetCacheAdapter, AssetStore

__all__ = [
    "AssetBackend",
    "AssetCacheAdapter",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetObjectKeyFactory",
    "AssetPathAdapter",
    "AssetRoot",
    "AssetStore",
    "DirectoryAssetBackend",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "PrefixAssetPathAdapter",
    "SqlAssetBackend",
    "StrictConfigReader",
    "WritableAssetBackend",
    "build_asset_sql_metadata",
    "directory_root",
    "filesystem_root",
    "resolved_name",
]
