#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sticky evidence that portable snapshots cannot represent every partition."""

from typing import Protocol

from ..errors import AIError, ErrorCode
from .state import StateTransaction, StoredOperation

_KEY = b"\x00capability-partitions"
_STREAM = _KEY


class _OperationReader(Protocol):
    async def get_operation(self, key: bytes) -> StoredOperation | None: ...


async def has_capability_partition_marker(transaction: _OperationReader) -> bool:
    """Read the durable presence marker without discovering custom routes."""
    marker = await transaction.get_operation(_KEY)
    if marker is None:
        return False
    if (marker.key != _KEY or marker.stream_key != _STREAM or marker.sequence != 0
            or marker.compactable or marker.state != "present"
            or marker.data != {"capability_partitions": True}):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return True


async def ensure_capability_partition_marker(transaction: StateTransaction) -> None:
    """Establish sticky evidence before optional partition writes can commit."""
    if not await has_capability_partition_marker(transaction):
        await transaction.insert_operation(StoredOperation(
            key=_KEY, stream_key=_STREAM, sequence=0, state="present",
            compactable=False, data={"capability_partitions": True},
        ))


__all__ = ["ensure_capability_partition_marker", "has_capability_partition_marker"]
