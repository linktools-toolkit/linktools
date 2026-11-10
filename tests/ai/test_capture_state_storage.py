#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captures keep metadata in state and only typed binary bodies in objects."""

import base64
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.inputs import AgentTaskInput, TaskGraphBindingCaptureStore, task_prompt_draft
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.captures import (
    capture_record, capture_record_value, input_capture_key, input_capture_state_dependency,
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
    namespace = "capture-state"
    body = b"\x00\xffcapture-file" * 500
    prompt = ("Inspect this file", BinaryContent(body, media_type="application/octet-stream"))
    objects = InMemoryObjectStore()
    plan = PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True).plan
    source = PersistenceStorage.from_plan(plan, object_store=objects)
    async with Persistence.open(namespace, storage=source, tenant_id=PRINCIPAL.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        text = await engine.inputs.create_agent_input("plain text" * 10000, principal=PRINCIPAL, idempotency_key="text")
        assert [item async for item in objects.list_objects()] == []
        captured = await engine.inputs.create_agent_input(prompt, principal=PRINCIPAL, idempotency_key="binary")
        assert await engine.inputs.create_agent_input(prompt, principal=PRINCIPAL, idempotency_key="binary") == captured
        key, _ = input_capture_state_dependency(captured)
        stored = await read_capture_record(state.task.tasks.state_store, namespace=namespace, tenant_id=PRINCIPAL.tenant_id, key=key)
        assert base64.b64encode(body) not in canonical_json_bytes(dict(stored.data))
        refs = tuple(iter_runtime_object_refs(capture_record_value(stored), default_domain=PersistenceDomain.TASK))
        assert refs
        for _domain, ref in refs:
            assert await read_runtime_object(objects, ref) == body
        assert len([item async for item in objects.list_objects()]) == 1
        restored = await engine.inputs.read_agent(captured, principal=PRINCIPAL)
        assert restored.prompt[1].data == body
        assert (await engine.inputs.read_agent(text, principal=PRINCIPAL)).prompt == "plain text" * 10000

    snapshots = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    async with Persistence.open(namespace, storage=source, tenant_id=PRINCIPAL.tenant_id, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=snapshots, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=snapshots, root=target, limits=limits)
    async with Persistence.open(namespace, storage=PersistenceStorage.from_root(target), tenant_id=PRINCIPAL.tenant_id) as state, TaskEngine.open(persistence=state, context=CONTEXT) as engine:
        assert (await engine.inputs.read_agent(captured, principal=PRINCIPAL)).prompt[1].data == body


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
    assert await restore_capture_value(record, read_object=lambda ref: read_runtime_object(objects, ref)) == value
    opaque_key = input_capture_key(namespace, tenant, "result", "result")
    opaque = {"kind": "binary", "data": base64.b64encode(body).decode(), "prompt": prompt}
    assert await prepare_capture_value(opaque_key, opaque, write_object=lambda raw: put_runtime_object(
        objects, ObjectKeyFactory(namespace), PersistenceDomain.TASK, tenant, raw)) == opaque


@pytest.mark.asyncio
async def test_large_task_declarations_share_database_content_and_collect_without_object_candidates(tmp_path: Path) -> None:
    namespace = "capture-bindings"
    objects = InMemoryObjectStore()
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite", object_store=objects)
    class Runner:
        pass
    task = Task.from_runner("capture.custom", Runner(), contract={
        "version": 1, "type": "custom", "effect_policy": "none", "output_contract": {"kind": "json"},
        "reconcile": False, "config": {"instructions": "large declaration " * 20000},
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
