#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Runtime snapshot admission limits and object-reference wire contracts."""

import base64
import binascii

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

from ..steps import encode_native_store_key, decode_native_store_key
from ..core import RUNTIME_OBJECT_STORE_ID, JsonValue
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef


class SnapshotExclusiveGuard(Protocol):
    """Stop and drain all users of every participating physical store.

    The caller must exclude reads and writes across processes, adapters and
    routed groups until the context exits. This is an application-level
    maintenance contract, not a database lock. Offline reclamation invalidates
    content evidence before users may reopen the stores.
    """

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    max_entries: int
    max_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or self.max_entries < 1
            or isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise ValueError("snapshot limits must be positive integers")


def snapshot_object_ref_payload(ref: ObjectRef) -> dict[str, JsonValue]:
    """Encode a portable object reference with its logical Runtime owner."""
    return {
        "store_id": RUNTIME_OBJECT_STORE_ID,
        "key": ref.key,
        "digest": ref.digest,
        "size": ref.size,
    }


def snapshot_object_ref_from_payload(value: object) -> ObjectRef:
    """Strictly decode the Runtime snapshot object-reference wire shape."""
    required = {"store_id", "key", "digest", "size"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    store_id = value["store_id"]
    key = value["key"]
    digest = value["digest"]
    size = value["size"]
    if (
        store_id != RUNTIME_OBJECT_STORE_ID
        or not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return ObjectRef(store_id, key, digest, size)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


_NATIVE_MESSAGE_FIELDS = frozenset({"store_key", "sha256", "message", "compression", "raw_size", "payload_json"})


def encode_native_message_row(row: Mapping[str, object]) -> dict[str, JsonValue]:
    """Encode binary bodies only at the portable JSON transport boundary."""
    result = {name: row[name] for name in _NATIVE_MESSAGE_FIELDS}
    result["store_key"] = encode_native_store_key(result["store_key"])
    body = result["message"]
    if body is not None:
        if type(body) is not bytes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result["message"] = base64.b64encode(body).decode("ascii")
    return result


def decode_native_message_row(row: Mapping[str, JsonValue]) -> dict[str, object]:
    """Restore a portable native row to the database's binary representation."""
    if not isinstance(row, Mapping) or set(row) != _NATIVE_MESSAGE_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result = dict(row)
    try:
        result["store_key"] = decode_native_store_key(result["store_key"])
    except (AIError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    body = result["message"]
    if body is not None:
        if not isinstance(body, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            result["message"] = base64.b64decode(body, validate=True)
        except (ValueError, binascii.Error) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return result


def encode_native_sequence_row(row: Mapping[str, object]) -> dict[str, JsonValue]:
    """Encode only the scope field of a portable native sequence row."""
    return dict(row, store_key=encode_native_store_key(row["store_key"]))


def decode_native_sequence_row(row: Mapping[str, JsonValue]) -> dict[str, object]:
    """Restore the exact native partition before SQL or closure validation."""
    try:
        return dict(row, store_key=decode_native_store_key(row["store_key"]))
    except (AIError, TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def validate_snapshot_native_rows(sequences: object, messages: object) -> None:
    """Accept only semantic native columns, never surrogate keys or audit fields."""
    sequence_fields = {"store_key", "sha256", "message_sha256", "tree_count",
                       "left_sha256", "right_sha256", "next_sha256"}
    if not isinstance(sequences, list) or not isinstance(messages, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for rows, fields in ((sequences, sequence_fields), (messages, _NATIVE_MESSAGE_FIELDS)):
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for name in fields - {"store_key", "tree_count", "message", "compression", "raw_size", "payload_json"}:
                value = row[name]
                if value is None and name in {"left_sha256", "right_sha256", "next_sha256"}:
                    continue
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded = decode_native_sequence_row(row)
            identity = decoded["store_key"], row["sha256"]
            if identity in seen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(identity)
            if "tree_count" in row and (type(row["tree_count"]) is not int or not 1 <= row["tree_count"] < 2 ** 63):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if "message" in row:
                from ._archive import ContentRecord
                ContentRecord.from_native_row(decode_native_message_row(row))


__all__ = [
    "encode_native_sequence_row", "decode_native_sequence_row",
    "validate_snapshot_native_rows",
    "encode_native_message_row",
    "decode_native_message_row",
    "SnapshotExclusiveGuard",
    "SnapshotLimits",
    "snapshot_object_ref_payload",
    "snapshot_object_ref_from_payload",
]
