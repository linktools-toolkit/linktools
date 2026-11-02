#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint ownership preserves stable control and archive addresses."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.contracts import RuntimePayloadRef
from linktools.ai.storage import InMemoryCacheBackend, ObjectRef, ObjectStore, StorageRoute


async def _put(store: ObjectStore, content: bytes) -> ObjectRef:
    digest = hashlib.sha256(content).hexdigest()

    async def chunks():
        yield content

    value = await store.put("shared-key", chunks(), expected_digest=digest, expected_size=len(content))
    return ObjectRef(store.store_id, value.key, value.digest, value.size)


@pytest.mark.asyncio
async def test_checkpoint_control_and_archive_keep_distinct_object_sources() -> None:
    async with Persistence.open("checkpoint-sources", storage=PersistenceStorage.in_memory()) as state:
        assert {item["domain"] for item in state.status()} == set(state.plan.resolved_routes)
        assert len(state.status()) == 8
        control = state.object_store(PersistenceDomain.CHECKPOINT_CONTROL)
        archive = state.object_store(PersistenceDomain.CHECKPOINT)
        control_ref = await _put(control, b"control")
        archive_ref = await _put(archive, b"archive")
        assert control_ref.store_id == archive_ref.store_id
        assert control_ref.key == archive_ref.key
        assert await state.read_object(PersistenceDomain.CHECKPOINT_CONTROL, control_ref) == b"control"
        assert await state.read_object(PersistenceDomain.CHECKPOINT, archive_ref) == b"archive"
        with pytest.raises(AIError) as error:
            await state.read_object(PersistenceDomain.CHECKPOINT, control_ref)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_transient_checkpoint_cleanup_does_not_delete_control_objects() -> None:
    storage = PersistenceStorage(replace(PersistencePlan(), checkpoint=StorageRoute.transient()))
    async with Persistence.open("checkpoint-retention", storage=storage) as state:
        control_ref = await _put(state.object_store(PersistenceDomain.CHECKPOINT_CONTROL), b"control")
        archive_ref = await _put(state.working_object_store(
            PersistenceDomain.CHECKPOINT, owner_scope="execution:completed",
        ), b"archive")
        assert await state.retention.release_execution_handoff("completed", tenant_id="default")
        assert await state.read_object(PersistenceDomain.CHECKPOINT_CONTROL, control_ref) == b"control"
        with pytest.raises(AIError) as error:
            await state.read_object(PersistenceDomain.CHECKPOINT, archive_ref)
        assert error.value.code is ErrorCode.STORAGE_NOT_FOUND


@pytest.mark.asyncio
async def test_checkpoint_cache_route_keeps_the_execution_transaction_group(tmp_path: Path) -> None:
    execution = StorageRoute.sqlite(tmp_path / "runtime.db", cache=InMemoryCacheBackend(max_bytes=65536))
    checkpoint = execution.with_cache(InMemoryCacheBackend(max_bytes=65536))
    storage = PersistenceStorage(replace(PersistencePlan.from_default(execution), checkpoint=checkpoint))
    async with Persistence.open("checkpoint-routes", storage=storage) as state:
        control_store = state.checkpoint.checkpoints.state_store
        archive_store = state.run_store.read_store(PersistenceDomain.CHECKPOINT).state_store
        assert control_store.storage_group is archive_store.storage_group
        assert state.plan.route(PersistenceDomain.CHECKPOINT) == checkpoint
        assert state.plan.route(PersistenceDomain.CHECKPOINT_CONTROL) == execution


def test_admitted_payload_source_retains_its_original_wire_address() -> None:
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
            "source_domain": {"$enum": "runtime_domain", "value": "recovery"},
        },
    }
    restored = decode_domain(wire, RuntimePayloadRef)
    assert restored.source_domain is PersistenceDomain.CHECKPOINT_CONTROL
    assert encode_domain(restored) == wire
