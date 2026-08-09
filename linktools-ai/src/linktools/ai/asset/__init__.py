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
from ._composition import AssetBackend, AssetComposition
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
    "AssetBackend",
    "AssetBatchResult",
    "AssetChange",
    "AssetCodec",
    "AssetCodecManifest",
    "AssetCodecManifestEntry",
    "AssetCodecRegistry",
    "AssetComposition",
    "AssetConflictError",
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
    "AssetParseError",
    "AssetPathAdapter",
    "AssetRequest",
    "AssetRevision",
    "AssetRoot",
    "AssetSource",
    "AssetStore",
    "AssetStoreRevision",
    "AssetValue",
    "AssetVersion",
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
