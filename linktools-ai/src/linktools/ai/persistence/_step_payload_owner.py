#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed execution payload bytes sharing the immutable Step content relation."""

import hashlib
from collections.abc import Mapping

from ..core import validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ._archive import ContentRecord, archive_scope_key, archive_scope_parts
from ._record_key import decode_key_components, encode_key_components
from .state import StoredRecord, record_key

PAYLOAD_PARTITION = "step_payload"
PAYLOAD_CODEC = "execution-payload"
PAYLOAD_PARTITION_MAX_BYTES = 274


def payload_store_key(namespace: str, tenant_id: str) -> bytes:
    """Keep payload and native message partitions structurally disjoint."""
    return encode_key_components((PAYLOAD_PARTITION, validate_persistence_namespace(namespace),
                                  validate_tenant_id(tenant_id)), PAYLOAD_PARTITION_MAX_BYTES)


def payload_store_parts(value: bytes) -> tuple[str, str]:
    try:
        parts = decode_key_components(value, PAYLOAD_PARTITION_MAX_BYTES)
        if len(parts) != 3 or parts[0] != PAYLOAD_PARTITION:
            raise ValueError("not a payload partition")
        if payload_store_key(parts[1], parts[2]) != value:
            raise ValueError("noncanonical payload partition")
        return parts[1], parts[2]
    except (TypeError, ValueError, UnicodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def payload_store_locator(value: bytes) -> str:
    payload_store_parts(value)
    return hashlib.sha256(value).hexdigest()


def payload_address(key: bytes) -> tuple[bytes, str] | None:
    """Resolve only the canonical fixed-codec archive content address."""
    try:
        parts = decode_key_components(key, 768)
    except (TypeError, ValueError, UnicodeError):
        return None
    if len(parts) != 6 or parts[0] != "record" or parts[3:5] != ("archive", "archive_content"):
        return None
    codec, separator, digest = parts[5].rpartition(":")
    if codec != PAYLOAD_CODEC or not separator:
        return None
    try:
        if len(digest) != 64 or bytes.fromhex(digest).hex() != digest:
            raise ValueError("invalid payload digest")
        return payload_store_key(parts[1], parts[2]), digest
    except (TypeError, ValueError, UnicodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def payload_record_values(record: StoredRecord) -> dict[str, object]:
    """Project validated immutable facts without retaining the State record shell."""
    content = ContentRecord.from_record(record)
    namespace, tenant_id = archive_scope_parts(record.scope_key)
    if (content.codec != PAYLOAD_CODEC or content.payload is not None
            or record.parent_key is not None or record.subject_id is not None
            or record.state is not None or record.lease_owner is not None
            or record.lease_fence != 0 or record.lease_expires_at is not None
            or record.data != content.to_json()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    store = payload_store_key(namespace, tenant_id)
    return {"store_key": store, "store_locator": payload_store_locator(store),
            "sha256": content.raw_digest, "message": content.body,
            "compression": content.compression, "raw_size": content.raw_size,
            "payload_json": None, "gc_revision": record.storage_revision}


def payload_record_from_row(row: Mapping[str, object]) -> StoredRecord:
    """Recover a logical archive record from its actual physical content facts."""
    namespace, tenant_id = payload_store_parts(row["store_key"])
    if row["payload_json"] is not None or row["store_locator"] != payload_store_locator(row["store_key"]):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = ContentRecord(PAYLOAD_CODEC, row["sha256"], row["raw_size"], row["compression"], body=row["message"])
    return StoredRecord(
        key=record_key(namespace, tenant_id, "archive", "archive_content", (content.identity,)),
        scope_key=archive_scope_key(namespace, tenant_id), parent_key=None, kind="archive_content",
        sort_key=content.identity.encode(), state=None, storage_revision=row["gc_revision"],
        lease_owner=None, lease_fence=0, lease_expires_at=None,
        data=content.to_json(), payload_bytes=content.body,
    )


__all__ = ["PAYLOAD_PARTITION", "PAYLOAD_CODEC", "PAYLOAD_PARTITION_MAX_BYTES", "payload_store_key",
           "payload_store_parts", "payload_store_locator", "payload_address", "payload_record_values",
           "payload_record_from_row"]
