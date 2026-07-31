#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared storage primitives and SQL conventions for domain stores.

Generic composition/revision/cache/overlay machinery lives here
(``cache``/``composition``/``multi``/``revision``): it is domain-agnostic and
consumed by domain stores (e.g. ``spec``) via the narrow Protocols below. The
truly cross-domain database, lease, and file primitives plus JSON helpers are
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
