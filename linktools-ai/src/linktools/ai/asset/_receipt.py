#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned Asset batch receipt codec shared by built-in writers."""

import hashlib
import re
from collections.abc import Mapping
from datetime import datetime

from ..core import JsonValue, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ..storage import (
    StorageBatchResult,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StoredPayload,
)
from ._domain import AssetInfo, AssetKey

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_batch_receipt_identity(
    idempotency_key: str | None,
    request_digest: str | None,
) -> None:
    if (idempotency_key is None) != (request_digest is None):
        raise ValueError("idempotency_key and request_digest must be provided together")
    if idempotency_key is None:
        return
    validate_idempotency_key(idempotency_key)
    if not isinstance(request_digest, str) or _SHA256.fullmatch(request_digest) is None:
        raise ValueError("request_digest must be a SHA-256 digest")


def batch_receipt_key_digest(idempotency_key: str) -> str:
    validate_idempotency_key(idempotency_key)
    return hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()


def encode_asset_batch_receipt(
    result: StorageBatchResult[AssetInfo, AssetKey],
) -> dict[str, JsonValue]:
    if result.idempotency_key is None or result.request_digest is None:
        raise ValueError("persisted batch receipt requires idempotency identity")
    validate_batch_receipt_identity(result.idempotency_key, result.request_digest)
    return {
        "version": 1,
        "idempotency_key_digest": batch_receipt_key_digest(result.idempotency_key),
        "request_digest": result.request_digest,
        "store_revision": result.store_revision.value,
        "results": [_encode_result(item) for item in result.results],
    }


def decode_asset_batch_receipt(
    payload: object,
    *,
    idempotency_key: str | None = None,
    expected_key_digest: str | None = None,
) -> StorageBatchResult[AssetInfo, AssetKey]:
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    key_digest = _sha256(payload.get("idempotency_key_digest"))
    if expected_key_digest is not None and key_digest != expected_key_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if idempotency_key is not None and batch_receipt_key_digest(idempotency_key) != key_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    request_digest = _sha256(payload.get("request_digest"))
    store_revision = StorageRevision(_string(payload.get("store_revision")))
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    results = tuple(_decode_result(item, store_revision) for item in raw_results)
    return StorageBatchResult(
        store_revision,
        True,
        results,
        request_digest,
        idempotency_key,
    )


def _encode_result(
    result: StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey],
) -> dict[str, JsonValue]:
    if isinstance(result, StoragePutResult):
        return {
            "kind": "put",
            "info": _encode_info(result.info),
            "entry_revision": result.entry_revision.value,
            "store_revision": result.store_revision.value,
            "changed": result.changed,
        }
    if isinstance(result, StorageDeleteResult):
        return {
            "kind": "delete",
            "key": _encode_key(result.key),
            "deleted": result.deleted,
            "entry_revision": (
                None if result.entry_revision is None else result.entry_revision.value
            ),
            "store_revision": result.store_revision.value,
        }
    return {
        "kind": "reset",
        "key": _encode_key(result.key),
        "reset": result.reset,
        "store_revision": result.store_revision.value,
    }


def _decode_result(
    payload: object,
    batch_revision: StorageRevision,
) -> StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]:
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    kind = payload.get("kind")
    store_revision = StorageRevision(_string(payload.get("store_revision")))
    if store_revision != batch_revision:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if kind == "put":
        info = _decode_info(payload.get("info"))
        entry_revision = StorageEntryRevision(_integer(payload.get("entry_revision"), minimum=1))
        changed = _boolean(payload.get("changed"))
        if info.revision != entry_revision or info.store_revision != store_revision:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StoragePutResult(info, entry_revision, store_revision, changed)
    if kind == "delete":
        raw_revision = payload.get("entry_revision")
        entry_revision = (
            None
            if raw_revision is None
            else StorageEntryRevision(_integer(raw_revision, minimum=1))
        )
        return StorageDeleteResult(
            _decode_key(payload.get("key")),
            _boolean(payload.get("deleted")),
            entry_revision,
            store_revision,
        )
    if kind == "reset":
        return StorageResetResult(
            _decode_key(payload.get("key")),
            _boolean(payload.get("reset")),
            store_revision,
        )
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _encode_key(key: AssetKey) -> dict[str, JsonValue]:
    return {"kind": key.kind, "id": key.id}


def _decode_key(payload: object) -> AssetKey:
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AssetKey(_string(payload.get("kind")), _string(payload.get("id")))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _encode_info(info: AssetInfo) -> dict[str, JsonValue]:
    return {
        "key": _encode_key(info.key),
        "revision": info.revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "status": info.status.value,
        "root_id": info.root_id,
        "root_digest": info.root_digest,
        "modified_at": info.modified_at.isoformat(),
        "metadata": dict(info.metadata),
        "content": None if info.content is None else info.content.to_json(),
    }


def _decode_info(payload: object) -> AssetInfo:
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = payload.get("content")
    try:
        return AssetInfo(
            _decode_key(payload.get("key")),
            StorageEntryRevision(_integer(payload.get("revision"), minimum=1)),
            StorageRevision(_string(payload.get("store_revision"))),
            _sha256(payload.get("etag")),
            _integer(payload.get("size"), minimum=0),
            StorageEntryStatus(_string(payload.get("status"))),
            _string(payload.get("root_id")),
            _string(payload.get("root_digest")),
            datetime.fromisoformat(_string(payload.get("modified_at"))),
            dict(metadata),
            None if content is None else StoredPayload.from_json(content),
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _sha256(value: object) -> str:
    text = _string(value)
    if _SHA256.fullmatch(text) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return text


def _integer(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


__all__ = [
    "batch_receipt_key_digest",
    "decode_asset_batch_receipt",
    "encode_asset_batch_receipt",
    "validate_batch_receipt_identity",
]
