#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Runtime snapshot persisted-field validation."""

import hashlib

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeSnapshot
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef


async def _put(store: InMemoryObjectStore, key: str, payload: bytes) -> ObjectRef:
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks():
        yield payload

    await store.put(
        key,
        chunks(),
        expected_size=len(payload),
        expected_digest=digest,
    )
    return ObjectRef(store.store_id, key, digest, len(payload))


@pytest.mark.asyncio
async def test_runtime_snapshot_rejects_coerced_object_ref_fields() -> None:
    store = InMemoryObjectStore("snapshot")
    manifest = {
        "kind": "runtime-snapshot",
        "format_version": 1,
        "namespace": "runtime",
        "tenant_id": "tenant",
        "state": {
            "store_id": "snapshot",
            "key": "state",
            "digest": "a" * 64,
            "size": "1",
        },
        "workspace": {
            "present": False,
            "workspace_id": None,
            "entries": [],
        },
        "metadata": {},
    }
    ref = await _put(
        store,
        "runtime-snapshot",
        canonical_json_bytes(manifest),
    )

    with pytest.raises(AIError) as raised:
        await RuntimeSnapshot.verify(
            ref,
            object_store=store,
            limits=SnapshotLimits(max_entries=10, max_bytes=4096),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
