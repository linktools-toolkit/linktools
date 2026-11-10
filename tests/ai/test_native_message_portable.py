#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable native roots retain their sole body owner across partial exports."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._snapshot import validate_snapshot_native_rows
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint
from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations, SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, read_object

from .test_step_archive_read_boundaries import _archive


_LIMITS = SnapshotLimits(10000, 10 * 1024 * 1024)


@pytest.mark.asyncio
@pytest.mark.parametrize("object_payload", (False, True))
async def test_partial_trace_snapshot_restores_native_bodies_and_rekeys_group(tmp_path: Path, object_payload: bool) -> None:
    objects = InMemoryObjectStore("portable")
    message = ModelResponse(parts=[TextPart("native portable body" * (2000 if object_payload else 1))])
    (tmp_path / "source").mkdir()
    async with _archive(tmp_path / "source", "sqlite") as (state, archive, run):
        native = SqlStepStore(archive.state_store.context, namespace=state.namespace, tenant_id=state.tenant_id)
        await native.initialize(local=True)
        archive.transcript_repository.bind_native_message_store(native)
        try:
            batch = await archive.prepare_checkpoints(
                run, (AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0),),
            )
            await archive.sync_prepared_projection(run, events=(), checkpoints=batch.checkpoints,
                                                  execution_id="execution", producer_generation=1)
            resources = replace(state.snapshot_resources(), read_only=True,
                                durable_domains=frozenset({PersistenceDomain.TRACE}))
            ref = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
                object_store=objects, limits=_LIMITS)
        finally:
            await native.close()
    manifest = json.loads(await read_object(objects, ref.key, expected_digest=ref.digest, expected_size=ref.size))
    assert set(manifest["domains"]) == {"trace"}
    assert len(manifest["native_messages"]) == 1
    assert len(manifest["native_messages"][0]["messages"]) == 1
    assert not manifest["archives"]
    assert bool(manifest["objects"]) is object_payload
    original_group = manifest["native_messages"][0]["group_id"]
    target = tmp_path / "target"
    await Persistence.restore_snapshot(ref, object_store=objects, root=target, limits=_LIMITS)
    async with Persistence.open("archive-boundaries", tenant_id="tenant",
                                storage=PersistenceStorage.from_root(target), read_only=True) as restored:
        archive = restored.run_store.read_store(PersistenceDomain.TRACE)
        assert [item async for item in archive.iter_messages(agent_run_id="run")] == [message]
        resources = replace(restored.snapshot_resources(), durable_domains=frozenset({PersistenceDomain.TRACE}))
        second = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
            object_store=objects, limits=_LIMITS)
    result = json.loads(await read_object(objects, second.key, expected_digest=second.digest, expected_size=second.size))
    assert result["native_messages"][0]["group_id"] != original_group
    before = json.loads(manifest["native_messages"][0]["messages"][0]["message"])
    after = json.loads(result["native_messages"][0]["messages"][0]["message"])
    assert before["raw_digest"] == after["raw_digest"]
    assert before["raw_size"] == after["raw_size"]
    assert result["native_messages"][0]["sequences"] == manifest["native_messages"][0]["sequences"]


@pytest.mark.parametrize("change", ["audit", "duplicate", "partition", "count", "body"])
def test_native_transport_rejects_nonsemantic_or_malformed_rows(change: str) -> None:
    sequence = {"store_digest": "a" * 64, "sha256": "b" * 64, "message_sha256": "c" * 64,
                "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    message = {"store_digest": "a" * 64, "sha256": "c" * 64, "message": "{}"}
    sequences, messages = [sequence], [message]
    if change == "audit":
        message["id"] = 1
    elif change == "duplicate":
        sequences.append(dict(sequence))
    elif change == "partition":
        message["store_digest"] = "not a digest"
    elif change == "count":
        sequence["tree_count"] = True
    else:
        message["message"] = {}
    with pytest.raises(AIError) as rejected:
        validate_snapshot_native_rows(sequences, messages)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
