#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captures keep metadata in state and only typed binary bodies in objects."""

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, TextContent, UserPromptPart

from linktools.ai.core import canonical_json_bytes, canonical_sha256, service_principal
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.inputs import AgentTaskInput, TaskGraphBindingCaptureStore, task_prompt_draft
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.captures import (
    capture_record, capture_record_value, capture_retention_value, capture_state_closure, input_capture_key, input_capture_state_dependency,
    prepare_capture_value, read_capture_record, restore_capture_value,
)
from linktools.ai.persistence.codec import encode_domain, iter_runtime_object_refs
from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import Task, TaskGraph, TaskGraphAdmission, TaskGraphLimits, TaskGraphRequest, TaskInvocationInputContract, TaskNode, TaskRef
from linktools.ai.task.engine import TaskEngine
from .test_evaluation_consumers import CONTEXT, PRINCIPAL
from .test_evaluation_retention import Offline


@pytest.mark.asyncio
async def test_agent_capture_stores_only_binary_files_in_objects_and_roundtrips_snapshot(tmp_path: Path) -> None:
    namespace = "capture/命名:空间"
    principal = service_principal("tenant/租户:2", "capture-owner")
    body = b"\x00\xffcapture-file" * 500
    prompt = ("Inspect this file", BinaryContent(body, media_type="application/octet-stream"))
    objects = InMemoryObjectStore()
    plan = PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True).plan
    source = PersistenceStorage.from_plan(plan, object_store=objects)
    async with Persistence.open(namespace, storage=source, tenant_id=principal.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        text = await engine.inputs.create_agent_input("plain text" * 10000, principal=principal, idempotency_key="text")
        assert [item async for item in objects.list_objects()] == []
        captured = await engine.inputs.create_agent_input(prompt, principal=principal, idempotency_key="binary")
        assert await engine.inputs.create_agent_input(prompt, principal=principal, idempotency_key="binary") == captured
        key, _ = input_capture_state_dependency(captured)
        stored = await read_capture_record(state.task.tasks.state_store, namespace=namespace, tenant_id=principal.tenant_id, key=key)
        assert base64.b64encode(body) not in canonical_json_bytes(dict(stored.data))
        refs = tuple(iter_runtime_object_refs(capture_retention_value(stored), default_domain=PersistenceDomain.TASK))
        assert refs
        assert await state.snapshot_operations().capture_objects((key,)) == tuple(dict.fromkeys(refs))
        await state.snapshot_operations().purge_unreferenced_objects(refs, exclusive=Offline())
        for _domain, ref in refs:
            assert await read_runtime_object(objects, ref) == body
        assert len([item async for item in objects.list_objects()]) == 1
        restored = await engine.inputs.read_agent(captured, principal=principal)
        assert [item.data for item in restored.prompt if isinstance(item, BinaryContent)] == [body]
        assert restored.prompt[0] == "Inspect this file" or (
            isinstance(restored.prompt[0], TextContent) and restored.prompt[0].text == "Inspect this file")
        assert (await engine.inputs.read_agent(text, principal=principal)).prompt == "plain text" * 10000

    snapshots = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    async with Persistence.open(namespace, storage=source, tenant_id=principal.tenant_id, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=snapshots, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=snapshots, root=target, limits=limits)
    async with Persistence.open(namespace, storage=PersistenceStorage.from_root(target), tenant_id=principal.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        assert [item.data for item in (await engine.inputs.read_agent(captured, principal=principal)).prompt
                if isinstance(item, BinaryContent)] == [body]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("agent", "task"))
async def test_capture_attachment_codec_preserves_context_original_input_and_source_identity(kind: str) -> None:
    namespace, tenant = "capture-slots", PRINCIPAL.tenant_id
    body = b"\x01\xfeexact bytes" * 200
    binary = BinaryContent(body, media_type="image/png")
    prompt = task_prompt_draft((binary,))
    context = ExecutionInputContext.from_messages((ModelRequest(parts=[UserPromptPart((binary,))]),))
    original = dict(AgentTaskInput((binary,), input_context=context))
    contract = TaskInvocationInputContract("execution", TaskRef("capture.agent", 1), original, original, {"type": "agent"})
    payload = ({"prompt": prompt, "original_input": original, "input_view": {"prompt": prompt},
                "input_context": context.to_payload(), "task_input": encode_domain(contract)}
               if kind == "agent" else {"contract": encode_domain(contract)})
    value = {"kind": kind, "version": 1, "namespace": namespace, "tenant_id": tenant, "payload": payload}
    key = input_capture_key(namespace, tenant, kind, "capture")
    objects = InMemoryObjectStore()
    stored = await prepare_capture_value(key, value, write_object=lambda raw: put_runtime_object(
        objects, ObjectKeyFactory(namespace), PersistenceDomain.TASK, tenant, raw))
    assert base64.b64encode(body) not in canonical_json_bytes(stored)
    record = capture_record(namespace, tenant, key, stored, source_digest=canonical_sha256(value))
    assert tuple(iter_runtime_object_refs(capture_retention_value(record), default_domain=PersistenceDomain.TASK))
    assert await restore_capture_value(record, read_object=lambda ref: read_runtime_object(objects, ref)) == value
    opaque_key = input_capture_key(namespace, tenant, "result", "result")
    opaque = {"kind": "binary", "data": base64.b64encode(body).decode(), "prompt": prompt}
    assert await prepare_capture_value(opaque_key, opaque, write_object=lambda raw: put_runtime_object(
        objects, ObjectKeyFactory(namespace), PersistenceDomain.TASK, tenant, raw)) == opaque


@pytest.mark.asyncio
@pytest.mark.parametrize("repetitions", (1, 20000))
async def test_task_declarations_share_database_content_and_collect_without_object_candidates(tmp_path: Path, repetitions: int) -> None:
    namespace = "capture-bindings"
    objects = InMemoryObjectStore()
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite", object_store=objects)
    class Runner:
        pass
    task = Task.from_runner("capture.custom", Runner(), contract={
        "version": 1, "type": "custom", "effect_policy": "none", "output_contract": {"kind": "json"},
        "reconcile": False, "config": {"instructions": "large declaration " * repetitions},
    })
    async with Persistence.open(namespace, storage=storage, tenant_id=PRINCIPAL.tenant_id) as state:
        store = state.task.tasks.state_store
        captures = TaskGraphBindingCaptureStore(namespace, store)
        for name in ("first", "second"):
            graph = TaskGraph(name, (TaskNode("task", task=task),))
            admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, name, TaskGraphLimits()))
            await captures.capture(admission, graph, tasks=(task,))
            assert (await captures.load(admission)).tasks[(task.id, task.revision)]["config"]["instructions"].startswith("large declaration")
        _scope, archive = await store.storage_group.open_archive()
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        assert len(rows) == 1
        assert rows[0].payload_bytes is not None
        assert rows[0].data["payload"] is None
        assert [item async for item in objects.list_objects()] == []
        await state.snapshot_operations().purge_unreferenced_objects((), exclusive=Offline())
        assert await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content"))) == ()
        assert await store.read(lambda tx: tx.list_records(RecordQuery(kind="task_graph_binding_capture"))) == ()


@pytest.mark.asyncio
async def test_empty_task_declarations_do_not_create_archive_content() -> None:
    from linktools.ai.persistence._memory import InMemoryStateStore
    from linktools.ai.persistence.captures import prepare_task_binding_value

    store = InMemoryStateStore()
    await store.initialize()
    try:
        manifest = {"tasks": [], "expanders": []}
        assert await prepare_task_binding_value(store, manifest,
            namespace="empty", tenant_id="tenant") == manifest
        assert await store.storage_group.open_archive() is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_capture_expiry_keeps_attachment_cleanup_references(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    objects = InMemoryObjectStore()
    storage = PersistenceStorage.sqlite(tmp_path / "expiry.sqlite", object_store=objects)
    async with Persistence.open("capture-expiry", storage=storage, tenant_id=PRINCIPAL.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        captured = await engine.inputs.create_agent_input(
            (BinaryContent(b"expired binary", media_type="application/octet-stream"),),
            principal=PRINCIPAL, idempotency_key="binary", expires_at=now + timedelta(hours=1))
        key, _ = input_capture_state_dependency(captured)
        maintenance = state.snapshot_operations()
        references = await maintenance.capture_objects((key,))
        assert references
        assert await engine.inputs.expire_inputs(principal=PRINCIPAL, now=now + timedelta(hours=2)) == (key,)
        await maintenance.purge_unreferenced_captures((key,), exclusive=Offline())
        assert await maintenance.capture_objects((key,)) == references
        await maintenance.purge_unreferenced_objects(references, exclusive=Offline())
        assert [item async for item in objects.list_objects()] == []


@pytest.mark.asyncio
async def test_capture_opaque_metadata_does_not_create_retention_dependencies() -> None:
    namespace, tenant = "opaque-capture", PRINCIPAL.tenant_id
    fake = {"kind": "binary", "digest": "a" * 64, "size": 3,
            "$dataclass": "agent_input_capture_ref", "fields": {}}
    contract = TaskInvocationInputContract("execution", TaskRef("custom", 1), fake, fake,
                                           {"type": "custom", "config": fake})
    key = input_capture_key(namespace, tenant, "task", "custom")
    record = capture_record(namespace, tenant, key, {
        "kind": "task", "version": 1, "namespace": namespace, "tenant_id": tenant,
        "payload": {"contract": encode_domain(contract)}})
    assert tuple(iter_runtime_object_refs(capture_retention_value(record), default_domain=PersistenceDomain.TASK)) == ()
    assert capture_state_closure((record,)) == frozenset({key})
    agent_key = input_capture_key(namespace, tenant, "agent", "opaque")
    agent_record = capture_record(namespace, tenant, agent_key, {
        "kind": "agent", "version": 1, "namespace": namespace, "tenant_id": tenant,
        "payload": {"prompt": "ordinary text", "original_input": {"opaque": fake},
                    "input_view": {"prompt": fake}, "task_input": None, "input_context": None}})
    assert tuple(iter_runtime_object_refs(capture_retention_value(agent_record), default_domain=PersistenceDomain.TASK)) == ()
    assert capture_state_closure((agent_record,)) == frozenset({agent_key})


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_task_binding_capture_publishes_content_and_owner_atomically(tmp_path: Path, monkeypatch, backend: str) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    class Runner:
        pass

    task = Task.from_runner("capture.atomic", Runner(), contract={
        "version": 1, "type": "custom", "effect_policy": "none",
        "output_contract": {"kind": "json"}, "reconcile": False,
        "config": {"instructions": "same immutable declaration"},
    })
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "atomic.sqlite")
    async with Persistence.open("capture-atomic", storage=storage, tenant_id=PRINCIPAL.tenant_id) as state:
        store = state.task.tasks.state_store
        captures = TaskGraphBindingCaptureStore("capture-atomic", store)
        graph = TaskGraph("atomic", (TaskNode("task", task=task),))
        admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, "atomic", TaskGraphLimits()))

        async def transaction_type(transaction):
            return type(transaction)

        transaction_class = await store.mutate(transaction_type)
        insert = transaction_class.insert_record

        async def fail_owner(transaction, record):
            if record.kind == "task_graph_binding_capture":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await insert(transaction, record)

        with monkeypatch.context() as patch:
            patch.setattr(transaction_class, "insert_record", fail_owner)
            with pytest.raises(AIError) as error:
                await captures.capture(admission, graph, tasks=(task,))
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        _, archive = store.storage_group.opened_archive
        assert await store.read(lambda tx: tx.list_records(RecordQuery(kind="task_graph_binding_capture"))) == ()
        assert await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content"))) == ()
        expected = await captures.capture(admission, graph, tasks=(task,))
        assert await captures.load(admission) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_task_binding_capture_reads_owner_and_contract_in_one_scope(tmp_path: Path, monkeypatch, backend: str) -> None:
    import linktools.ai.inputs._task_graph_binding_capture as capture_module
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence.state import active_state_scope

    class Runner:
        pass

    task = Task.from_runner("capture.snapshot", Runner(), contract={
        "version": 1, "type": "custom", "effect_policy": "none",
        "output_contract": {"kind": "json"}, "reconcile": False,
    })
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "snapshot.sqlite")
    async with Persistence.open("capture-snapshot", storage=storage, tenant_id=PRINCIPAL.tenant_id) as state:
        store = state.task.tasks.state_store
        captures = TaskGraphBindingCaptureStore("capture-snapshot", store)
        graph = TaskGraph("snapshot", (TaskNode("task", task=task),))
        admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, "snapshot", TaskGraphLimits()))
        owner_scopes, body_scopes = [], []
        read_owner, read_body = capture_module.read_capture_record, ContentArchive.read_contents

        async def owner(*args, **kwargs):
            owner_scopes.append(active_state_scope())
            return await read_owner(*args, **kwargs)

        async def body(*args, **kwargs):
            body_scopes.append(active_state_scope())
            return await read_body(*args, **kwargs)

        monkeypatch.setattr(capture_module, "read_capture_record", owner)
        monkeypatch.setattr(ContentArchive, "read_contents", body)
        expected = await captures.capture(admission, graph, tasks=(task,))
        assert await captures.capture(admission, graph, tasks=(task,)) == expected
        assert owner_scopes[-1] is body_scopes[-1] and body_scopes[-1] is not None
        assert not body_scopes[-1].writable
        assert await captures.load(admission) == expected
        assert owner_scopes[-1] is body_scopes[-1] and body_scopes[-1] is not None
        assert not body_scopes[-1].writable


@pytest.mark.parametrize("inline_slot", ("tasks", "expanders"))
def test_retained_task_declarations_cannot_overwrite_inline_declarations(inline_slot: str) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.captures import _expand_task_binding_value

    raw = canonical_json_bytes({"tasks": [], "expanders": []})
    # This is the shared expansion boundary used by live reads and snapshots.
    with pytest.raises(AIError) as error:
        _expand_task_binding_value({"declarations": {}, inline_slot: []}, raw)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    # Keep unrelated metadata and optional-document omission unchanged.
    assert _expand_task_binding_value({"declarations": {}, "format_version": 1}, raw) == {
        "format_version": 1, "tasks": [], "expanders": [],
    }
