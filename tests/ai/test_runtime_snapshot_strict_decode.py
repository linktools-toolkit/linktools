#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Runtime snapshot persisted-field validation."""

import hashlib
from datetime import datetime, timezone

import pytest

from linktools.ai.core import SessionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeSnapshot, RuntimeState
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.runtime.state._codec import (
    _encode_persisted_domain,
    encode_envelope,
    encode_record,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import SessionRecord
from linktools.ai.runtime.state._store import (
    StoredRecord,
    partition_digest,
    scope_digest,
    sortable_identity,
)
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
        "format_version": 2,
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


@pytest.mark.asyncio
async def test_runtime_snapshot_rejects_coerced_format_version() -> None:
    store = InMemoryObjectStore("snapshot")
    manifest = {
        "kind": "runtime-snapshot",
        "format_version": 2.0,
        "namespace": "runtime",
        "tenant_id": "tenant",
        "state": {
            "store_id": "snapshot",
            "key": "state",
            "digest": "a" * 64,
            "size": 1,
        },
        "workspace": {
            "present": False,
            "entries": [],
        },
        "metadata": {},
    }
    ref = await _put(
        store,
        "runtime-snapshot-version",
        canonical_json_bytes(manifest),
    )

    with pytest.raises(AIError) as raised:
        await RuntimeSnapshot.verify(
            ref,
            object_store=store,
            limits=SnapshotLimits(max_entries=10, max_bytes=4096),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_runtime_state_restore_rejects_mismatched_logical_record_key_before_target_write(
    tmp_path,
) -> None:
    namespace = "runtime"
    tenant_id = "tenant"
    session = SessionRecord(
        session_id="session",
        owner_principal_id="principal",
        status=SessionStatus.OPEN,
        revision=1,
        cwd=None,
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        closed_at=None,
        active_execution_id=None,
        agent_id="agent",
    )
    record = StoredRecord(
        b"x" * 32,
        partition_digest(namespace, tenant_id, "conversation", "session"),
        scope_digest(
            namespace,
            tenant_id,
            "conversation",
            "session",
            "owner",
            session.owner_principal_id,
        ),
        None,
        "session",
        sortable_identity(session.session_id),
        session.status.value,
        0,
        None,
        0,
        None,
        encode_envelope(
            {
                "type": wire_type_id(session),
                "payload": _encode_persisted_domain(session),
            }
        ),
    )
    manifest = {
        "kind": "runtime-state-snapshot",
        "format_version": 2,
        "namespace": namespace,
        "tenant_id": tenant_id,
        "domains": {
            "conversation": {
                "records": [encode_record(record)],
                "aliases": [],
                "facts": [],
                "operations": [],
                "sequences": [],
            }
        },
        "objects": [],
    }
    store = InMemoryObjectStore("snapshot")
    ref = await _put(
        store,
        "invalid-state-snapshot",
        canonical_json_bytes(manifest),
    )
    target = tmp_path / "restored"

    with pytest.raises(AIError) as raised:
        await RuntimeState.restore_snapshot(
            ref,
            object_store=store,
            root=target,
            limits=SnapshotLimits(max_entries=20, max_bytes=16 * 1024),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not target.exists()
