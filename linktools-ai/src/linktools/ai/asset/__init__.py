#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed asset values, codecs and the single AssetStore boundary."""

from ._codec import AssetCodec, AssetCodecManifest, AssetCodecManifestEntry, AssetCodecRegistry
from ._objectcache import AssetCacheCodec, AssetCacheStore, AssetObjectCache
from ._config import StrictConfigReader, resolved_name
from ._content import AssetContent, AssetContentInfo, compute_asset_etag
from .domain import AssetSource
from ._index import AssetIndex
from .domain import (
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
from .store import AssetStore
from ._backend import FileAssetBackend, MemoryAssetBackend
from ._local import AssetPathAdapter, LocalAssetBackend, PrefixAssetPathAdapter
from ._parsing import AssetLoader, AssetLoaderSource, TextAssetStore, load_markdown_text, load_yaml_text, parse_json_text, parse_markdown_text, parse_yaml_text
from ..core.errors import AssetConflictError, AssetError, AssetNotFoundError, AssetParseError, InvalidAssetError

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
    "FileAssetBackend",
    "MemoryAssetBackend",
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
    "AssetCacheStore",
    "AssetSource",
    "InvalidAssetError",
    "LocalAssetBackend",
    "PrefixAssetPathAdapter",
    "StrictConfigReader",
    "TextAssetStore",
    "compute_asset_etag",
    "load_markdown_text",
    "load_yaml_text",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
    "resolved_name",
]
