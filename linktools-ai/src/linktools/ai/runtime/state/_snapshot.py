#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Runtime snapshot admission limits and object-reference wire contracts."""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

from ...core import RUNTIME_OBJECT_STORE_ID, JsonValue
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef


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


__all__ = [
    "SnapshotExclusiveGuard",
    "SnapshotLimits",
    "snapshot_object_ref_payload",
    "snapshot_object_ref_from_payload",
]
