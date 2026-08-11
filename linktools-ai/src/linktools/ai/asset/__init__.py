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
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
    local_directory_root,
)
from ._domain import AssetBackend, AssetInfo, AssetKey, AssetRoot
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._logical import (
    AssetCodec,
    AssetDiscoveryStatus,
    AssetEntry,
    AssetRef,
    AssetResource,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetTypeRegistrySnapshot,
    AssetValueAdapter,
    AssetVariantBinding,
    DirectoryLayout,
    ResolvedAsset,
    SingleFileLayout,
)
from ._repository import AssetRepository, AssetScope
from ._sql import SqlAssetBackend, SqlAssetTables
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
    "AssetNotFoundError",
    "AssetParseError",
    "AssetPathAdapter",
    "AssetRef",
    "AssetRepository",
    "AssetResource",
    "AssetScope",
    "AssetRoot",
    "AssetStore",
    "AssetTypeBinding",
    "AssetTypeRegistry",
    "AssetTypeRegistrySnapshot",
    "AssetValueAdapter",
    "AssetVariantBinding",
    "DirectoryLayout",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "InvalidAssetError",
    "LocalDirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "ResolvedAsset",
    "SingleFileLayout",
    "SqlAssetBackend",
    "SqlAssetTables",
    "StrictConfigReader",
    "filesystem_root",
    "local_directory_root",
    "resolved_name",
]
