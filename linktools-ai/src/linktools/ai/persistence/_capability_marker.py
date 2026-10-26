#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sticky evidence that portable snapshots cannot represent every partition."""

import hashlib

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import StateTransaction, StoredOperation

_KEY = hashlib.sha256(b"linktools.runtime.capability-partitions").digest()


async def has_capability_partition_marker(transaction: StateTransaction) -> bool:
    """Read the durable presence marker without discovering custom routes."""
    marker = await transaction.get_operation(_KEY)
    if marker is None:
        return False
    if marker.state != "present" or marker.data != {"capability_partitions": True}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return True


async def ensure_capability_partition_marker(transaction: StateTransaction) -> None:
    """Establish sticky evidence before optional partition writes can commit."""
    if not await has_capability_partition_marker(transaction):
        await transaction.insert_operation(StoredOperation(
            key_digest=_KEY, stream_digest=_KEY, sequence=0, state="present",
            compactable=False, data={"capability_partitions": True},
        ))


__all__ = ["ensure_capability_partition_marker", "has_capability_partition_marker"]
