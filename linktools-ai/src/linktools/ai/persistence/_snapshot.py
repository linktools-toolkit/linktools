#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Runtime snapshot admission limits and object-reference wire contracts."""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

from ..core import RUNTIME_OBJECT_STORE_ID, JsonValue
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef


class SnapshotExclusiveGuard(Protocol):
    """Quiesce every mutation source that can affect a portable snapshot."""

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


def validate_snapshot_native_rows(sequences: object, messages: object) -> None:
    """Accept only semantic native columns, never surrogate keys or audit fields."""
    sequence_fields = {"store_digest", "sha256", "message_sha256", "tree_count",
                       "left_sha256", "right_sha256", "next_sha256"}
    message_fields = {"store_digest", "sha256", "message"}
    if not isinstance(sequences, list) or not isinstance(messages, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for rows, fields in ((sequences, sequence_fields), (messages, message_fields)):
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for name in fields - {"tree_count", "message"}:
                value = row[name]
                if value is None and name in {"left_sha256", "right_sha256", "next_sha256"}:
                    continue
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            identity = row["store_digest"], row["sha256"]
            if identity in seen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(identity)
            if "tree_count" in row and (type(row["tree_count"]) is not int or not 1 <= row["tree_count"] < 2 ** 63):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if "message" in row and not isinstance(row["message"], str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = [
    "validate_snapshot_native_rows",
    "SnapshotExclusiveGuard",
    "SnapshotLimits",
    "snapshot_object_ref_payload",
    "snapshot_object_ref_from_payload",
]
