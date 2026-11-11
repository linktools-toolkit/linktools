#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable captures preserve business data while retaining real attachments."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, TextContent

from linktools.ai.steps import native_store_key
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import archive_content_scope, ArchiveRoot
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence.snapshots import SnapshotLimits, snapshot_manifest_identity
from linktools.ai.steps import NativeStepSource
from linktools.ai.storage import InMemoryObjectStore, read_object
from linktools.ai.task.engine import TaskEngine
from .test_evaluation_consumers import CONTEXT, PRINCIPAL


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", (False, True))
async def test_capture_snapshot_preserves_opaque_markers_and_attachment_bytes(tmp_path: Path, binary: bool) -> None:
    namespace = "capture-opaque"
    metadata = {
        "invalid_root": {"$dataclass": "archive_root", "fields": {}},
        "valid_root": encode_domain(ArchiveRoot("business-scope", archive_content_scope("business", "tenant"))),
        "native_root": {"$dataclass": "transcript_chunk", "fields": {}},
        "native_source": encode_domain(NativeStepSource("b" * 32, native_store_key("c", "tenant"))),
        "payload": {"kind": "binary", "digest": "a" * 64, "size": 3},
    }
    body = b"\x00\xffopaque attachment" * 500
    prompt = (TextContent("ordinary note", metadata=metadata),)
    if binary:
        prompt += (BinaryContent(body, media_type="application/octet-stream"),)
    source = PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True)
    async with Persistence.open(namespace, storage=source, tenant_id=PRINCIPAL.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        capture = await engine.inputs.create_agent_input(prompt, principal=PRINCIPAL, idempotency_key="note")
        assert (await engine.inputs.read_agent(capture, principal=PRINCIPAL)).prompt[0].metadata == metadata
    snapshots = InMemoryObjectStore()
    limits = SnapshotLimits(1000, 4 * 1024 * 1024)
    async with Persistence.open(namespace, storage=source, tenant_id=PRINCIPAL.tenant_id, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=snapshots, limits=limits)
    manifest = json.loads(await read_object(snapshots, snapshot.key, expected_digest=snapshot.digest, expected_size=snapshot.size))
    identity = snapshot_manifest_identity(manifest)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=snapshots, root=target, limits=limits)
    restored = PersistenceStorage.from_root(target)
    async with Persistence.open(namespace, storage=restored, tenant_id=PRINCIPAL.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        result = await engine.inputs.read_agent(capture, principal=PRINCIPAL)
        assert result.prompt[0].metadata == metadata
        assert [item.data for item in result.prompt if isinstance(item, BinaryContent)] == ([body] if binary else [])
    async with Persistence.open(namespace, storage=restored, tenant_id=PRINCIPAL.tenant_id, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=snapshots, limits=limits)
    manifest = json.loads(await read_object(snapshots, snapshot.key, expected_digest=snapshot.digest, expected_size=snapshot.size))
    assert snapshot_manifest_identity(manifest) == identity


@pytest.mark.asyncio
async def test_opaque_capture_coexists_with_relocated_native_history(tmp_path: Path) -> None:
    from dataclasses import replace
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.core import service_principal
    from linktools.ai.persistence import PersistenceDomain, SqlStepStore
    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint
    from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations
    from .test_step_archive_read_boundaries import _archive

    source = tmp_path / "source"
    source.mkdir()
    snapshots = InMemoryObjectStore()
    limits = SnapshotLimits(10000, 10 * 1024 * 1024)
    message = ModelResponse(parts=[TextPart("native body")])
    principal = service_principal("tenant", "owner")
    async with _archive(source, "sqlite") as (state, archive, run):
        native = SqlStepStore(archive.state_store.context, namespace=state.namespace, tenant_id=state.tenant_id)
        await native.initialize(local=True)
        archive.transcript_repository.bind_native_message_store(native)
        try:
            batch = await archive.prepare_checkpoints(run, (
                AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0),))
            await archive.sync_prepared_projection(run, events=(), checkpoints=batch.checkpoints,
                                                  execution_id="execution", producer_generation=1)
            async with TaskEngine.open(persistence=state, context=CONTEXT) as engine:
                capture = await engine.inputs.create_agent_input((TextContent("note", metadata={
                    "$dataclass": "archive_root", "fields": {}}),), principal=principal, idempotency_key="note")
            resources = replace(state.snapshot_resources(), read_only=True,
                                durable_domains=frozenset({PersistenceDomain.TRACE, PersistenceDomain.TASK}))
            snapshot = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
                object_store=snapshots, limits=limits)
        finally:
            await native.close()
    before = json.loads(await read_object(snapshots, snapshot.key, expected_digest=snapshot.digest, expected_size=snapshot.size))
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=snapshots, root=target, limits=limits)
    async with Persistence.open("archive-boundaries", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as state:
        assert [item async for item in state.run_store.read_store(PersistenceDomain.TRACE).iter_messages(agent_run_id="run")] == [message]
        async with TaskEngine.open(persistence=state, context=CONTEXT) as engine:
            assert (await engine.inputs.read_agent(capture, principal=principal)).prompt[0].metadata == {
                "$dataclass": "archive_root", "fields": {}}
    async with Persistence.open("archive-boundaries", tenant_id="tenant", storage=PersistenceStorage.from_root(target), read_only=True) as state:
        resources = replace(state.snapshot_resources(), durable_domains=frozenset({PersistenceDomain.TRACE, PersistenceDomain.TASK}))
        snapshot = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(object_store=snapshots, limits=limits)
    after = json.loads(await read_object(snapshots, snapshot.key, expected_digest=snapshot.digest, expected_size=snapshot.size))
    assert before["native_messages"][0]["group_id"] != after["native_messages"][0]["group_id"]
    assert snapshot_manifest_identity(before) == snapshot_manifest_identity(after)


@pytest.mark.asyncio
async def test_archive_attachment_metadata_does_not_own_native_roots(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.model.messages import encode_model_messages
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._archive import PreparedArchive

    storage = PersistenceStorage.sqlite(tmp_path / "archive.sqlite")
    async with Persistence.open("archive-opaque", storage=storage) as state:
        archive = await state.run_store.read_store(PersistenceDomain.TRACE).transcript_repository.content_archive(create=True)
        items = [TextContent("opaque note", metadata={
                     "native": {"$dataclass": "transcript_chunk", "fields": {}},
                     "status": {"$enum": "business", "value": "unknown"},
                     "capture": {"$dataclass": "agent_input_capture_ref", "fields": {}}}),
                 BinaryContent(b"file", media_type="application/octet-stream")]
        raw = encode_model_messages([ModelRequest(parts=[UserPromptPart(items)])])
        prepared = PreparedArchive()
        references = await archive.prepare_content((("model-message", raw),), prepared)
        await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
        assert await archive.read_contents(references, codec="model-message") == (raw,)
        from .test_evaluation_retention import Offline
        await state.purge_unreferenced_objects((), exclusive=Offline())
