"""Shared storage primitives and SQL conventions for domain stores."""

from .json import JsonScalar, JsonValue, canonical_json_bytes, decode_json, encode_json, normalize_json
from .local.files import atomic_write_bytes, atomic_write_json, read_bytes, read_json
from .local.paths import StoragePath, Sha256Digest, StorageId, safe_child
from .composition import (
    StorageAdapter,
    StorageCacheAdapter,
    StorageComposition,
    StorageInitializer,
)
from .revision import (
    ChangeSource,
    CompositeRevisionSource,
    MetadataSnapshot,
    RevisionCache,
    RevisionCacheCodec,
    RevisionCacheSource,
    RevisionSource,
)
from .multi import (
    BatchStorageReader,
    MultiBackend,
    OverlayRefreshPolicy,
    StorageLayer,
    StorageReader,
    StorageWriter,
)

__all__ = [
    "StoragePath",
    "BatchStorageReader",
    "MultiBackend",
    "JsonScalar",
    "JsonValue",
    "Sha256Digest",
    "StorageId",
    "StorageComposition",
    "StorageAdapter",
    "StorageCacheAdapter",
    "StorageInitializer",
    "StorageLayer",
    "StorageReader",
    "StorageWriter",
    "OverlayRefreshPolicy",
    "ChangeSource",
    "CompositeRevisionSource",
    "MetadataSnapshot",
    "RevisionSource",
    "RevisionCache",
    "RevisionCacheCodec",
    "RevisionCacheSource",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "decode_json",
    "encode_json",
    "normalize_json",
    "read_bytes",
    "read_json",
    "safe_child",
]
