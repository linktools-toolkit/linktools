#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static exports for the domain-independent storage kernel."""

from .cache import ContentCache, FilesystemContentCache, MemoryContentCache, TieredContentCache
from .composition import StorageAdapter, StorageCacheAdapter, StorageComposition, StorageLayer
from .database import CoordinationScope, StorageDatabase, build_sqlite_storage, build_storage, scope_for_url
from .initialization import initialize_storage
from .local.files import atomic_write_bytes, atomic_write_json, read_bytes, read_json
from .local.paths import Sha256Digest, StorageId, StoragePath, safe_child
from .multi import BatchStorageReader, BatchStorageWriter, StorageReader, StorageWriter
from .revision import LayerRefreshPolicy, MetadataLoad, MetadataLoadMode, RevisionSource, StorageChange, StorageMetadataBackend
from .versioning import VersionSummary, VersionedStorage

__all__ = [
    "BatchStorageReader", "BatchStorageWriter", "ContentCache", "CoordinationScope",
    "FilesystemContentCache", "LayerRefreshPolicy", "MemoryContentCache", "MetadataLoad",
    "MetadataLoadMode", "RevisionSource", "Sha256Digest", "StorageAdapter",
    "StorageCacheAdapter", "StorageChange", "StorageComposition", "StorageDatabase",
    "StorageLayer", "StorageMetadataBackend", "StoragePath", "StorageId", "StorageReader",
    "StorageWriter", "TieredContentCache", "atomic_write_bytes", "atomic_write_json",
    "build_sqlite_storage", "build_storage", "initialize_storage", "read_bytes", "read_json",
    "safe_child", "scope_for_url", "VersionSummary", "VersionedStorage",
]
