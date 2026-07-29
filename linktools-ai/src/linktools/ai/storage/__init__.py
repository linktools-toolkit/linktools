"""Shared storage primitives and SQL conventions for domain stores.

The generic composition/revision/cache/overlay machinery that used to live here
has moved to ``linktools.ai.spec`` -- it served only the spec domain's
revision-aware source, so it is spec-owned now. This package keeps only the
truly cross-domain database, lease, and file primitives plus the JSON helpers
re-exported for convenience.
"""

from ..json import JsonScalar, JsonValue, canonical_json_bytes, normalize_json
from .local.files import atomic_write_bytes, atomic_write_json, read_bytes, read_json
from .local.paths import StoragePath, Sha256Digest, StorageId, safe_child

__all__ = [
    "StoragePath",
    "JsonScalar",
    "JsonValue",
    "Sha256Digest",
    "StorageId",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "normalize_json",
    "read_bytes",
    "read_json",
    "safe_child",
]
