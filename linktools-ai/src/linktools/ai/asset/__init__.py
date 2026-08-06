#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asset parsing, indexing, caching, and persistence contracts."""

from ..foundation.errors import (
    InvalidAssetError,
    AssetConflictError,
    AssetError,
    AssetNotFoundError,
    AssetParseError,
)
from .cache import AssetObjectCache
from .contracts import AssetCodec, AssetSource
from .content import AssetContent, AssetContentInfo, compute_asset_etag
from .index import AssetIndex
from .parsing import AssetLoader, StrictConfigReader, parse_json_text, parse_markdown_text, parse_yaml_text
from .source import AssetLoaderSource
from .store import AssetStore

__all__ = [
    "InvalidAssetError",
    "AssetCodec",
    "AssetConflictError",
    "AssetContent",
    "AssetContentInfo",
    "AssetError",
    "AssetIndex",
    "AssetLoader",
    "AssetLoaderSource",
    "AssetNotFoundError",
    "AssetObjectCache",
    "AssetParseError",
    "AssetSource",
    "AssetStore",
    "StrictConfigReader",
    "compute_asset_etag",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
]
