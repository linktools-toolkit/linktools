#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native occurrence roots survive retirement of their original snapshot owner."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice, SqlStepStore
from linktools.ai.persistence._archive import ContentArchive, ImmutableSequenceNode
from linktools.ai.persistence._archive_closure import (
    NativeMessageClosure, iter_archive_roots, iter_native_message_roots,
    rekey_archive_roots, rekey_native_message_roots, require_known_retention,
    validate_native_message_content,
)
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._object_cleanup import purge_unreferenced_objects
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import _encode_persisted_domain, encode_domain
from linktools.ai.persistence.contracts import ArchivedContextBlock, RuntimePayloadRef
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_step_content_cleanup import _Offline, _snapshot

@pytest.fixture(autouse=True)
def _exercise_explicit_offline_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _native_content_index

    async def defer(_context, **kwargs) -> None:
        pass

    # These tests deliberately accumulate orphans for the explicit offline
    # collector. Automatic collection has its own owner/concurrency coverage.
    monkeypatch.setattr(_native_content_index, "maintain_native_content", defer)



def _row(node: ImmutableSequenceNode, scope: str = "a" * 64) -> dict[str, object]:
    return dict(store_digest=scope, sha256=node.identity, message_sha256=node.content_id,
                tree_count=node.tree_count, left_sha256=node.left, right_sha256=node.right,
                next_sha256=node.next_forest)


def test_native_closure_retains_only_visible_bodies_and_not_historical_tails() -> None:
    vanished_tail = "f" * 64
    left = ImmutableSequenceNode("1" * 64, 1, next_forest=vanished_tail)
    right = ImmutableSequenceNode("2" * 64, 1)
    head = ImmutableSequenceNode("3" * 64, 3, left.identity, right.identity)
    closure = NativeMessageClosure([_row(node) for node in (left, right, head)])
    closure.mark("a" * 64, (NativeMessageSlice(head.identity, 3, 1, 2),))
    assert closure.retained_messages == {("a" * 64, left.content_id)}
    assert closure.retained_sequences == {("a" * 64, node.identity) for node in (left, right, head)}
    with pytest.raises(AIError):
        closure.mark("b" * 64, (NativeMessageSlice(head.identity, 3, 0, 3),))
    with pytest.raises(AIError):
        closure.mark("a" * 64, (NativeMessageSlice(left.identity, 1, 0, 1),))


@pytest.mark.parametrize("persisted", (False, True))
def test_native_scanner_and_rekey_preserve_declared_context_slots(persisted: bool) -> None:
    root = NativeMessageRoot("group", "a" * 64, (NativeMessageSlice("b" * 64, 1, 0, 1),))
    encode = _encode_persisted_domain if persisted else encode_domain
    context = encode(ArchivedContextBlock(root, "c" * 64, 2))
    business = encode({"root": root.to_json(), "fake": encode(root)})
    assert tuple(iter_native_message_roots(context)) == (root,)
    assert not tuple(iter_native_message_roots(business))
    assert rekey_native_message_roots(business, {}) == business
    rewritten = rekey_native_message_roots(context, {"group": "target"}, {("group", "a" * 64): "d" * 64})
    assert tuple(iter_native_message_roots(rewritten)) == (replace(root, group_id="target", store_digest="d" * 64),)
    manifest = dict(kind="execution-context", version=1, history=root.to_json(),
                    semantic_digest="digest", replace_history_system_prompt=False, unavailable_reason=None,
                    repository_instructions=None, session_metadata=None, memory=None, preload_skills=[])
    wire = {"$dataclass": "execution_record", "fields": {"input_context": encode(
        RuntimePayloadRef(StoredPayload.inline_json(manifest), PersistenceDomain.EXECUTION))}}
    assert tuple(iter_native_message_roots(wire)) == (root,)
    assert not tuple(iter_archive_roots(wire))
    rewritten = rekey_native_message_roots(rekey_archive_roots(wire, {}), {"group": "target"})
    assert tuple(iter_native_message_roots(rewritten)) == (replace(root, group_id="target"),)
    malformed = {**context, "fields": {**context["fields"], "root": {"$dataclass": "unknown_root"}}}
    with pytest.raises(AIError):
        tuple(iter_native_message_roots(malformed))


@pytest.mark.parametrize("value,state", [({}, "UNKNOWN"),
    ({"$dataclass": "execution_record", "fields": {"dependency_hold_ids": {"$tuple": ["hold"]}}}, None)])
def test_unresolved_retention_never_authorizes_collection(value: object, state: str | None) -> None:
    with pytest.raises(AIError):
        require_known_retention(value, state=state)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ("context", "execution-context"))
@pytest.mark.parametrize("object_backed", (False, True))
async def test_cross_owner_message_roots_outlive_native_snapshot_pruning(tmp_path: Path, owner: str, object_backed: bool) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retention.db'}")
    await provision_database(engine)
    state = SqlStateStore(engine)
    await state.initialize()
    group_id, archive_store = await state.storage_group.open_archive(create=True)
    objects = InMemoryObjectStore()
    archive = ContentArchive(archive_store, group_id, namespace="steps", tenant_id="one",
                             object_store=objects if object_backed else None)
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    native.bind_message_archive(archive)
    table = build_step_sql_metadata().tables["ai_step_snapshots"]
    try:
        old = _snapshot("run", 1, "retained elsewhere" * 10_000)
        if object_backed:
            old = replace(old, messages=[ModelRequest(parts=[UserPromptPart([
                "retained elsewhere", BinaryContent(b"retained file", media_type="image/png"),
            ])])])
        await native.save_snapshot(old)
        async with engine.connect() as connection:
            first = await connection.scalar(select(table.c.sequence_root).where(table.c.store_digest == native.store_digest))
            messages = build_step_sql_metadata().tables["ai_step_messages"]
            body = (await connection.execute(select(messages))).mappings().one()
        content = validate_native_message_content(body, group_id=group_id)
        reference = next(iter(content.payload.object_refs()), None)
        assert (reference is not None) is object_backed
        root = NativeMessageRoot(group_id, native.store_digest, (NativeMessageSlice(first, 1, 0, 1),))
        value = encode_domain(ArchivedContextBlock(root, "c" * 64, 2))
        if owner == "execution-context":
            manifest = dict(kind="execution-context", version=1, history=root.to_json(),
                semantic_digest="digest", replace_history_system_prompt=False, unavailable_reason=None,
                repository_instructions=None, session_metadata=None, memory=None, preload_skills=[])
            value = {"$dataclass": "execution_record", "fields": {"input_context": encode_domain(
                RuntimePayloadRef(StoredPayload.inline_json(manifest), PersistenceDomain.EXECUTION))}}
        key = hashlib.sha256(b"retaining-owner").digest()
        record = StoredRecord(key, None, None, "test_owner", "owner", None, 1, None, 0, None, value)
        await state.mutate(lambda tx: tx.insert_record(record))
        await native.save_snapshot(_snapshot("run", 2, "current"))
        result = await native.purge_unreferenced_content(exclusive=_Offline())
        assert result.retained_messages == 2
        assert result.deleted_messages == 0
        if reference is not None:
            protected = await purge_unreferenced_objects(((PersistenceDomain.TRACE, reference),),
                namespace="steps", tenant_id="one", stores={PersistenceDomain.TRACE: state},
                object_stores={PersistenceDomain.TRACE: objects}, exclusive=_Offline())
            assert protected.retained == ((PersistenceDomain.TRACE, reference),)
            assert await objects.stat(reference.key) is not None
        await state.mutate(lambda tx: tx.delete_record(key))
        result = await native.purge_unreferenced_content(exclusive=_Offline())
        assert result.deleted_messages == 1
        assert result.retained_messages == 1
        if reference is not None:
            collected = await purge_unreferenced_objects(((PersistenceDomain.TRACE, reference),),
                namespace="steps", tenant_id="one", stores={PersistenceDomain.TRACE: state},
                object_stores={PersistenceDomain.TRACE: objects}, exclusive=_Offline())
            assert collected.deleted == ((PersistenceDomain.TRACE, reference),)
    finally:
        await native.close()
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_unbound_cleanup_cannot_skip_cross_owner_state(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    try:
        await native.save_snapshot(_snapshot("run", 1, "retired"))
        await native.save_snapshot(_snapshot("run", 2, "current"))
        with pytest.raises(AIError) as failure:
            await native.purge_unreferenced_content(exclusive=_Offline())
        assert failure.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_native_database_integrity_and_attachment_group_boundaries_fail_closed() -> None:
    from linktools.ai.model.attachments import store_attachment_payload
    from linktools.ai.model.messages import encode_model_messages
    from linktools.ai.persistence._archive import ContentRecord
    from linktools.ai.storage import ObjectRef

    raw = b'[{"kind":"request","parts":[]}]'
    digest = hashlib.sha256(raw).hexdigest()
    content = ContentRecord("model-message", digest, len(raw), "raw", body=raw)
    row = dict(store_digest="a" * 64, sha256=digest, **content.to_native_values())
    assert validate_native_message_content(row) == content
    with pytest.raises(AIError):
        validate_native_message_content({**row, "sha256": "f" * 64})
    with pytest.raises(AIError):
        validate_native_message_content({**row, "message": b"corrupt"})

    raw = encode_model_messages([ModelRequest(parts=[UserPromptPart([
        BinaryContent(b"file", media_type="image/png"),
    ])])])
    async def write(data):
        return ObjectRef("runtime", "v1/archive/other/scope/body", hashlib.sha256(data).hexdigest(), len(data))
    payload = await store_attachment_payload(raw, format="model-messages", write_object=write)
    content = ContentRecord("model-message", payload.digest, payload.size, "raw", payload)
    with pytest.raises(AIError):
        validate_native_message_content(dict(store_digest="a" * 64, sha256=payload.digest,
                                            **content.to_native_values()), group_id="group")


@pytest.mark.parametrize("checkpoint", [False, {}, {"output_retries_used": True, "message_sha256": "a" * 64},
    {"output_retries_used": 0, "message_sha256": "unknown"}])
def test_native_snapshot_retention_rejects_unknown_checkpoint_shape(checkpoint: object) -> None:
    import json

    closure = NativeMessageClosure([])
    with pytest.raises(AIError):
        closure.mark_snapshot(dict(store_digest="a" * 64, sequence_root=None, pruned=False,
            messages=json.dumps(dict(stored_messages=0, checkpoint=checkpoint))))


@pytest.mark.parametrize("damage", ({"tree_count": "one"}, {"left_sha256": "b" * 64}, {"message_sha256": None}))
def test_native_closure_rejects_malformed_known_node_fields(damage: dict[str, object]) -> None:
    node = ImmutableSequenceNode("1" * 64)
    closure = NativeMessageClosure([{**_row(node), **damage}])
    with pytest.raises(AIError) as failure:
        closure.mark("a" * 64, (NativeMessageSlice(node.identity, 1, 0, 1),))
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", (PersistenceDomain.TRACE, PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION))
@pytest.mark.parametrize("native_first", (False, True))
async def test_native_only_domain_gc_retains_cross_owner_roots_without_snapshot_table(
    tmp_path: Path, domain: PersistenceDomain, native_first: bool,
) -> None:
    import json

    from linktools.ai.migrate import provision_runtime_database
    from linktools.ai.persistence._archive import ContentRecord
    from linktools.ai.persistence._schema import build_runtime_sql_metadata
    from linktools.ai.persistence._step_schema import build_step_domain_sql_metadata
    from linktools.ai.persistence._history import encode_model_messages

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native-only.db'}")
    objects = InMemoryObjectStore()
    await provision_runtime_database(engine, domains=(domain,), object_store=objects)
    metadata = build_runtime_sql_metadata(frozenset({domain}))
    build_step_domain_sql_metadata((domain,), metadata=metadata)
    assert "ai_step_snapshots" not in metadata.tables
    state = SqlStateStore(engine, metadata=metadata, runtime_domain=domain)
    await state.initialize()
    group_id, archive_store = await state.storage_group.open_archive(create=True)
    archive = ContentArchive(archive_store, group_id, namespace="steps", tenant_id="one", object_store=objects)
    native = SqlStepStore(engine, namespace="steps", tenant_id="one")
    native.bind_message_archive(archive)
    try:
        raw = encode_model_messages([ModelRequest(parts=[UserPromptPart([
            "retained body", BinaryContent(b"retained file", media_type="image/png"),
        ])])])
        compression, payload, encoded = await archive.prepare_payload(raw)
        reference, = payload.object_refs()
        content = ContentRecord("model-message", hashlib.sha256(raw).hexdigest(), len(raw), compression, payload)
        orphan_raw = b'[{"kind":"response","parts":[]}]'
        orphan = ContentRecord("model-message", hashlib.sha256(orphan_raw).hexdigest(), len(orphan_raw),
                               "raw", body=orphan_raw)
        nodes = [ImmutableSequenceNode(value.raw_digest) for value in (content, orphan)]
        async with engine.begin() as connection:
            for value, node in zip((content, orphan), nodes):
                await connection.execute(metadata.tables["ai_step_messages"].insert().values(
                    store_digest=native.store_digest, sha256=value.raw_digest, **value.to_native_values()))
                await connection.execute(metadata.tables["ai_step_sequences"].insert().values(_row(node, native.store_digest)))
        root = NativeMessageRoot(group_id, native.store_digest, (NativeMessageSlice(nodes[0].identity, 1, 0, 1),))
        key = hashlib.sha256(b"native-only-owner").digest()
        record = StoredRecord(key, None, None, "test_owner", "owner", None, 1, None, 0, None,
                              encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        await state.mutate(lambda tx: tx.insert_record(record))
        candidates = ((domain, reference),)
        if native_first:
            retained = await native.purge_unreferenced_content(exclusive=_Offline())
            assert retained.retained_messages == retained.retained_sequences == 1
            assert retained.deleted_messages == retained.deleted_sequences == 1
        protected = await purge_unreferenced_objects(candidates, namespace="steps", tenant_id="one",
            stores={domain: state}, object_stores={domain: objects}, exclusive=_Offline())
        assert protected.retained == candidates
        async with engine.connect() as connection:
            for name, identity in (("ai_step_messages", content.raw_digest),
                                   ("ai_step_sequences", nodes[0].identity)):
                table = metadata.tables[name]
                assert (await connection.execute(select(table.c.sha256))).scalars().all() == [identity]
        assert await archive.read_payload_object(reference) == b"retained file"
        retained = await native.purge_unreferenced_content(exclusive=_Offline())
        assert retained.retained_messages == retained.retained_sequences == 1
        assert retained.deleted_messages == retained.deleted_sequences == 0
        await state.mutate(lambda tx: tx.delete_record(key))
        removed = await native.purge_unreferenced_content(exclusive=_Offline())
        assert removed.deleted_messages == removed.deleted_sequences == 1
        collected = await purge_unreferenced_objects(candidates, namespace="steps", tenant_id="one",
            stores={domain: state}, object_stores={domain: objects}, exclusive=_Offline())
        assert collected.deleted == candidates
    finally:
        await native.close()
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ("ai_step_messages", "ai_step_sequences", "ai_step_snapshots"))
async def test_native_gc_rejects_missing_required_physical_tables(tmp_path: Path, missing: str) -> None:
    from linktools.ai.persistence._step_maintenance import inspect_native_message_tables
    from linktools.ai.storage import create_sql_storage_context

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'missing-schema.db'}")
    await provision_database(engine)
    metadata = build_step_sql_metadata()
    context = create_sql_storage_context(engine)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(lambda connection: metadata.tables[missing].drop(connection))
        with pytest.raises(AIError) as failure:
            await inspect_native_message_tables(context, metadata)
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled", "unknown"))
async def test_object_gc_invalidates_evidence_before_offline_guard_releases(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    import asyncio
    from contextlib import asynccontextmanager

    from linktools.ai.persistence import _object_cleanup
    from linktools.ai.storage import ObjectRef

    engine = create_async_engine("sqlite+aiosqlite://")
    state = SqlStateStore(engine)
    events = []

    class Guard:
        @asynccontextmanager
        async def offline_exclusivity(self):
            events.append("entered")
            try:
                yield
            finally:
                assert events[-1] == "invalidated"
                events.append("released")
                await asyncio.sleep(0)

    async def collect(*args, **kwargs):
        events.append("collected")
        if outcome == "failure":
            raise RuntimeError("injected failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        if outcome == "unknown":
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return _object_cleanup.ObjectCleanupResult()

    def invalidate(cls, context):
        assert context.engine is engine
        events.append("invalidated")

    monkeypatch.setattr(_object_cleanup, "_purge_unreferenced_objects", collect)
    monkeypatch.setattr(SqlStepStore, "invalidate_context", classmethod(invalidate))
    call = purge_unreferenced_objects(((PersistenceDomain.TRACE, ObjectRef("runtime", "candidate", "a" * 64, 0)),),
        namespace="steps", tenant_id="one", stores={PersistenceDomain.TRACE: state}, object_stores={}, exclusive=Guard())
    try:
        if outcome == "success":
            await call
        else:
            error = {"failure": RuntimeError, "cancelled": asyncio.CancelledError, "unknown": AIError}[outcome]
            with pytest.raises(error):
                await call
        assert events == ["entered", "collected", "invalidated", "released"]
    finally:
        await state.close()
        await engine.dispose()
