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
from ._sql import SqlAssetBackend, SqlAssetTables
from ._store import AssetCacheAdapter, AssetStore

__all__ = [
    "AssetBackend",
    "AssetCacheAdapter",
    "AssetConflictError",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetNotFoundError",
    "AssetParseError",
    "AssetPathAdapter",
    "AssetRoot",
    "AssetStore",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "InvalidAssetError",
    "LocalDirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "SqlAssetBackend",
    "SqlAssetTables",
    "StrictConfigReader",
    "filesystem_root",
    "local_directory_root",
    "resolved_name",
]
