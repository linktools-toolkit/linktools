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
from ._domain import (
    AssetBackend,
    AssetInfo,
    AssetKey,
    AssetRoot,
    AssetVersionRef,
    WritableAssetBackend,
)
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._materialization import AssetMaterializer, MaterializedAssets, validate_materialized_path
from ._object import AssetObjectKeyFactory
from ._sql import SqlAssetBackend, build_asset_sql_metadata
from ._store import AssetCacheAdapter, AssetStore, AssetStoreReader

__all__ = [
    "AssetBackend",
    "AssetCacheAdapter",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetMaterializer",
    "AssetObjectKeyFactory",
    "AssetPathAdapter",
    "AssetRoot",
    "AssetStore",
    "AssetVersionRef",
    "AssetStoreReader",
    "DirectoryAssetBackend",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "MaterializedAssets",
    "PrefixAssetPathAdapter",
    "SqlAssetBackend",
    "StrictConfigReader",
    "WritableAssetBackend",
    "build_asset_sql_metadata",
    "directory_root",
    "filesystem_root",
    "resolved_name",
    "validate_materialized_path",
]
