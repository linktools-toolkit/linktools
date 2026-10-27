#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted contexts reuse verified roots without re-reading prior bodies."""

import hashlib

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.persistence import PersistenceDomain, Persistence
from linktools.ai.persistence._archive import ArchiveRoot
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def test_accepted_context_reuses_verified_same_group_history_without_prefix_io(tmp_path, monkeypatch):
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="context-root", tenant_id="tenant")
    try:
        service = ExecutionContextArchive(
            state.execution.executions.state_store,
            state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="context-root", tenant_id="tenant",
        )
        context = ExecutionInputContext.from_messages(tuple(
            ModelRequest(parts=[UserPromptPart(f"message-{index}")]) for index in range(20)
        ))
        first = await service.prepare(context)
        await first.archive.store.mutate(lambda tx: first.archive.publish(tx, first.records))
        root = ArchiveRoot.from_json(first.reference.payload.decode()["history"])
        reads = []
        original = first.archive.store.read_records

        async def capture(keys):
            reads.append(tuple(keys))
            return await original(keys)

        monkeypatch.setattr(first.archive.store, "read_records", capture)
        reused = await service.prepare(context, history_root=root,
                                       history_digest=hashlib.sha256(context.history).hexdigest())
        assert not reads
        assert not reused.records.records
        assert reused.reference == first.reference
        assert await service.load(reused.reference) == context
        changed_messages = list(context.model_messages())
        changed_messages[-1].parts[0].content = "different"
        changed = ExecutionInputContext.from_messages(changed_messages)
        retained = await service.prepare(changed, history_root=root,
                                         history_digest=hashlib.sha256(context.history).hexdigest())
        assert retained.records.records
        assert retained.reference != first.reference
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_context_archive_handle_is_bound_and_failed_open_is_retryable(tmp_path, monkeypatch, backend):
    from linktools.ai.errors import AIError

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="context-root", tenant_id="tenant")
    store = state.execution.executions.state_store
    group = store.storage_group
    objects = state.archive_object_store(PersistenceDomain.EXECUTION)
    service = ExecutionContextArchive(store, objects, namespace="context-root", tenant_id="tenant")
    calls = []
    original = group.open_archive

    async def observed_open(*, create=False):
        calls.append(create)
        if len(calls) == 1:
            return None
        return await original(create=create)

    monkeypatch.setattr(group, "open_archive", observed_open)
    try:
        with pytest.raises(AIError):
            await service._open()
        assert calls == [False]
        assert service._archive is None
        context = ExecutionInputContext.from_messages((ModelRequest(parts=[UserPromptPart("message")]),))
        prepared = await service.prepare(context)
        await prepared.archive.store.mutate(lambda tx: prepared.archive.publish(tx, prepared.records))
        assert await service._open() is prepared.archive
        assert await service.load(prepared.reference) == context
        assert calls == [False, True]
        other = ExecutionContextArchive(store, objects, namespace="other", tenant_id="tenant")
        with pytest.raises(AIError):
            await other.load(prepared.reference)
    finally:
        await state.close()
    with pytest.raises(AIError):
        await service.load(prepared.reference)
