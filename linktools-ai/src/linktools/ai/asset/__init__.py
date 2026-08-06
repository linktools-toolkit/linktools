#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed asset values, codecs and the single AssetStore boundary."""

from .codec import AssetCodec, AssetCodecManifest, AssetCodecManifestEntry, AssetCodecRegistry
from .cache import AssetCacheCodec, AssetCacheStore, AssetObjectCache
from .config import StrictConfigReader, resolved_name
from .content import AssetContent, AssetContentInfo, compute_asset_etag
from .contracts import AssetSource
from .index import AssetIndex
from .model import (
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
from .parsing import AssetLoader, TextAssetStore, load_markdown_text, load_yaml_text, parse_json_text, parse_markdown_text, parse_yaml_text
from .local import AssetPathAdapter, LocalAssetBackend, PrefixAssetPathAdapter
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
    "AssetNotFoundError",
    "AssetObjectCache",
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
