#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw Asset file storage."""

from ..errors import (
    AssetConflictError,
    AssetError,
    AssetNotFoundError,
    AssetParseError,
    InvalidAssetError,
)
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
    WritableAssetBackend,
)
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._logical import (
    AssetCodec,
    AssetDiscoveryStatus,
    AssetEntry,
    AssetRef,
    AssetResource,
    AssetRetargeter,
    AssetTypeBinding,
    AssetValueAdapter,
    AssetVariantBinding,
    DirectoryLayout,
    ResolvedAsset,
    SingleFileLayout,
)
from ._object import AssetObjectKeyFactory
from ._repository import AssetRepository, AssetScope
from ._sql import (
    SqlAssetBackend,
    build_asset_sql_metadata,
)
from ._store import AssetCacheAdapter, AssetStore

__all__ = [
    "AssetBackend",
    "AssetCacheAdapter",
    "AssetCodec",
    "AssetConflictError",
    "AssetDiscoveryStatus",
    "AssetEntry",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetObjectKeyFactory",
    "AssetNotFoundError",
    "AssetParseError",
    "AssetPathAdapter",
    "AssetRef",
    "AssetRepository",
    "AssetResource",
    "AssetRetargeter",
    "AssetScope",
    "AssetRoot",
    "AssetStore",
    "AssetTypeBinding",
    "AssetValueAdapter",
    "AssetVariantBinding",
    "DirectoryLayout",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "InvalidAssetError",
    "DirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "ResolvedAsset",
    "SingleFileLayout",
    "SqlAssetBackend",
    "WritableAssetBackend",
    "build_asset_sql_metadata",
    "StrictConfigReader",
    "filesystem_root",
    "directory_root",
    "resolved_name",
]
