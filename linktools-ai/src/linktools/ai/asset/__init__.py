#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed asset values, codecs and the single AssetStore boundary."""

from ._codec import AssetCodec, AssetCodecManifest, AssetCodecManifestEntry, AssetCodecRegistry
from ._cache import AssetCacheCodec, AssetObjectCache
from ._config import StrictConfigReader, resolved_name
from ._content import AssetContent, AssetContentInfo, AssetContentSource, compute_asset_etag
from ._domain import AssetSource
from ._index import AssetIndex
from ._domain import (
    AssetBackend,
    AssetBatchPartialError,
    AssetBatchResult,
    AssetChange,
    AssetDeleteResult,
    AssetInfo,
    AssetKey,
    AssetRequest,
    AssetRevision,
    AssetRoot,
    AssetStoreRevision,
    AssetValue,
    AssetVersion,
    BatchAssetReader,
    BatchAssetWriter,
    OwnedAssetInfo,
    VersionedAssetBackend,
)
from ._store import AssetStore
from ._sql import SqlAssetBackend, SqlAssetTables
from ._backend import FilesystemAssetBackend, InMemoryAssetBackend
from ._filesystem import AssetPathAdapter, FilesystemAssetContentStore, PrefixAssetPathAdapter
from ._parsing import AssetLoader, AssetLoaderSource, load_markdown_text, load_yaml_text, parse_json_text, parse_markdown_text, parse_yaml_text
from ..errors import AssetConflictError, AssetError, AssetNotFoundError, AssetParseError, InvalidAssetError

__all__ = [
    "AssetBackend",
    "AssetBatchPartialError",
    "AssetBatchResult",
    "AssetChange",
    "AssetConflictError",
    "AssetCodec",
    "AssetCodecManifest",
    "AssetCodecManifestEntry",
    "AssetCodecRegistry",
    "AssetDeleteResult",
    "AssetError",
    "AssetContent",
    "AssetContentInfo",
    "AssetInfo",
    "AssetIndex",
    "AssetKey",
    "AssetLoader",
    "AssetLoaderSource",
    "AssetNotFoundError",
    "AssetObjectCache",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "AssetPathAdapter",
    "AssetParseError",
    "AssetRequest",
    "AssetRevision",
    "AssetRoot",
    "AssetStore",
    "AssetStoreRevision",
    "AssetValue",
    "AssetVersion",
    "BatchAssetReader",
    "BatchAssetWriter",
    "OwnedAssetInfo",
    "VersionedAssetBackend",
    "AssetCacheCodec",
    "AssetSource",
    "InvalidAssetError",
    "FilesystemAssetContentStore",
    "PrefixAssetPathAdapter",
    "StrictConfigReader",
    "SqlAssetBackend",
    "SqlAssetTables",
    "AssetContentSource",
    "compute_asset_etag",
    "load_markdown_text",
    "load_yaml_text",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
    "resolved_name",
]
