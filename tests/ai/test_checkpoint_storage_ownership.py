#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint controls and archives share one domain and transaction owner."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.contracts import RuntimePayloadRef
from linktools.ai.storage import InMemoryCacheBackend, ObjectRef, ObjectStore, StorageRetentionMode, StorageRoute, read_object


async def _put(store: ObjectStore, content: bytes) -> ObjectRef:
    digest = hashlib.sha256(content).hexdigest()

    async def chunks():
        yield content

    value = await store.put("shared-key", chunks(), expected_digest=digest, expected_size=len(content))
    return ObjectRef(store.store_id, value.key, value.digest, value.size)

@pytest.mark.asyncio
async def test_checkpoint_controls_and_archives_share_the_same_state_store() -> None:
    async with Persistence.open("checkpoint-sources", storage=PersistenceStorage.in_memory()) as state:
        assert {item["domain"] for item in state.status()} == set(state.plan.resolved_routes)
        assert len(state.status()) == 8
        archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
        assert state.checkpoint.checkpoints.state_store is archive.state_store
        assert state.checkpoint.approvals.state_store is archive.state_store
        assert state.checkpoint.external_calls.state_store is archive.state_store
        assert state.checkpoint.tools.state_store is archive.state_store
        reference = await _put(state.object_store(PersistenceDomain.CHECKPOINT), b"control")
        assert await state.read_object(PersistenceDomain.CHECKPOINT, reference) == b"control"

@pytest.mark.asyncio
async def test_transient_checkpoint_cleanup_does_not_delete_control_objects() -> None:
    storage = PersistenceStorage(replace(PersistencePlan(), checkpoint=StorageRoute.transient()))
    async with Persistence.open("checkpoint-retention", storage=storage) as state:
        status = {item["domain"]: item for item in state.status()}
        assert status[PersistenceDomain.CHECKPOINT]["retention"] is StorageRetentionMode.TRANSIENT
        assert status[PersistenceDomain.EXECUTION]["retention"] is StorageRetentionMode.VOLATILE
        control_ref = await _put(state.object_store(PersistenceDomain.CHECKPOINT), b"control")
        working = state.working_object_store(PersistenceDomain.CHECKPOINT, owner_scope="execution:completed")
        other = state.working_object_store(PersistenceDomain.CHECKPOINT, owner_scope="execution:active")
        archive_ref = await _put(working, b"archive")
        other_ref = await _put(other, b"active")
        assert await read_object(working, archive_ref.key, expected_digest=archive_ref.digest, expected_size=archive_ref.size) == b"archive"
        assert await state.retention.release_execution_handoff("completed", tenant_id="default")
        assert await state.read_object(PersistenceDomain.CHECKPOINT, control_ref) == b"control"
        assert await read_object(other, other_ref.key, expected_digest=other_ref.digest, expected_size=other_ref.size) == b"active"
        with pytest.raises(AIError) as error:
            await read_object(working, archive_ref.key, expected_digest=archive_ref.digest, expected_size=archive_ref.size)
        assert error.value.code is ErrorCode.STORAGE_NOT_FOUND


@pytest.mark.asyncio
async def test_checkpoint_cache_route_keeps_the_execution_transaction_group(tmp_path: Path) -> None:
    execution = StorageRoute.sqlite(tmp_path / "runtime.db", cache=InMemoryCacheBackend(max_bytes=65536))
    checkpoint = execution.with_cache(InMemoryCacheBackend(max_bytes=65536))
    storage = PersistenceStorage(replace(PersistencePlan.from_default(execution), checkpoint=checkpoint))
    async with Persistence.open("checkpoint-routes", storage=storage) as state:
        control_store = state.checkpoint.checkpoints.state_store
        archive_store = state.run_store.read_store(PersistenceDomain.CHECKPOINT).state_store
        assert control_store is archive_store
        assert control_store.storage_group is state.execution.executions.state_store.storage_group
        assert state.plan.route(PersistenceDomain.CHECKPOINT) == checkpoint
        assert state.plan.route(PersistenceDomain.CHECKPOINT).cache is checkpoint.cache


def test_admitted_payload_source_round_trips_checkpoint() -> None:
    wire = {
        "$dataclass": "runtime_payload_ref",
        "fields": {
            "payload": {
                "$dataclass": "stored_payload",
                "fields": {
                    "kind": "inline", "encoding": "base64",
                    "digest": "0682c5f2076f099c34cfdd15a9e063849ed437a49677e6fcc5b4198c76575be5",
                    "size": 8, "value": "b3JpZ2luYWw=", "ref": None,
                },
            },
            "source_domain": {"$enum": "runtime_domain", "value": "checkpoint"},
        },
    }
    restored = decode_domain(wire, RuntimePayloadRef)
    assert restored.source_domain is PersistenceDomain.CHECKPOINT
    assert encode_domain(restored) == wire
