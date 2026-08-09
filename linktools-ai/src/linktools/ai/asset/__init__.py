#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed asset values, codecs and the single AssetStore boundary."""

from ..errors import (
    AssetConflictError,
    AssetError,
    AssetNotFoundError,
    AssetParseError,
    InvalidAssetError,
)
from ._backend import InMemoryAssetBackend
from ._codec import (
    AssetCodec,
    AssetCodecManifest,
    AssetCodecManifestEntry,
    AssetCodecRegistry,
)
from ._config import StrictConfigReader, resolved_name
from ._directory import (
    AssetPathAdapter,
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
    local_directory_root,
)
from ._domain import (
    AssetBatchResult,
    AssetChange,
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryInfo,
    AssetEntryKey,
    AssetEntryOrigin,
    AssetEntryRevision,
    AssetEntrySnapshot,
    AssetEntryVersion,
    AssetInfo,
    AssetKey,
    AssetRequest,
    AssetRevision,
    AssetRoot,
    AssetSource,
    AssetStoreRevision,
    AssetValue,
    AssetVersion,
)
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._sql import SqlAssetBackend, SqlAssetTables
from ._store import AssetStore

__all__ = [
    "AssetBatchResult",
    "AssetChange",
    "AssetConflictError",
    "AssetCodec",
    "AssetCodecManifest",
    "AssetCodecManifestEntry",
    "AssetCodecRegistry",
    "AssetDeleteResult",
    "AssetEntryBatchResult",
    "AssetEntryChange",
    "AssetEntryDeleteResult",
    "AssetEntryInfo",
    "AssetEntryKey",
    "AssetEntryOrigin",
    "AssetEntryRevision",
    "AssetEntrySnapshot",
    "AssetEntryVersion",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetNotFoundError",
    "AssetPathAdapter",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "AssetParseError",
    "AssetRequest",
    "AssetRevision",
    "AssetRoot",
    "AssetStore",
    "AssetStoreRevision",
    "AssetValue",
    "AssetVersion",
    "AssetSource",
    "filesystem_root",
    "InvalidAssetError",
    "LocalDirectoryAssetBackend",
    "PrefixAssetPathAdapter",
    "StrictConfigReader",
    "SqlAssetBackend",
    "SqlAssetTables",
    "local_directory_root",
    "resolved_name",
]
