#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted contexts reuse verified roots without re-reading prior bodies."""

import hashlib
from dataclasses import replace

import pytest
from sqlalchemy import event
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import NativeMessageRoot, PersistenceDomain, Persistence
from linktools.ai.persistence._archive_closure import iter_archive_content_refs, rekey_archive_roots
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.storage import StoredPayload
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def _publish_context(state, prepared, *, execution_id="accepted-context"):
    from .test_runtime_persistence_evidence import _execution

    repository = state.execution.executions
    execution = replace(_execution(), execution_id=execution_id, root_execution_id=execution_id,
                        input_context=prepared.reference)

    async def commit(transaction):
        await prepared.publish()
        await repository.create_execution_in_transaction(transaction, execution)

    await repository.state_store.mutate(commit)


@pytest.mark.parametrize("invalid_skills", ({}, "", [None]))
@pytest.mark.parametrize("reader", ("load", "closure", "rekey"))
async def test_context_archive_rejects_malformed_preload_skills(tmp_path, invalid_skills, reader):
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="context-root", tenant_id="tenant")
    try:
        service = ExecutionContextArchive(
            state.execution.executions.state_store,
            state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="context-root", tenant_id="tenant",
        )
        context = ExecutionInputContext.from_messages(())
        prepared = await service.prepare(context)
        await _publish_context(state, prepared)
        manifest = prepared.reference.payload.decode()
        assert manifest["preload_skills"] == []
        assert await service.load(prepared.reference) == context
        malformed = replace(prepared.reference, payload=StoredPayload.inline_json({
            **manifest, "preload_skills": invalid_skills,
        }))
        wire = {"$dataclass": "execution_record", "fields": {"input_context": encode_domain(malformed)}}
        with pytest.raises(AIError) as failure:
            if reader == "load":
                await service.load(malformed)
            elif reader == "closure":
                tuple(iter_archive_content_refs(wire))
            else:
                rekey_archive_roots(wire, {prepared.archive.scope_id: "restored"})
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


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
            ModelRequest(parts=[UserPromptPart(f"message-{index}")]) for index in range(3)
        ))
        first = await service.prepare(context)
        await _publish_context(state, first)
        root = NativeMessageRoot.from_json(first.reference.payload.decode()["history"])
        reads = []
        original = first.archive.store.read_records

        async def capture(keys):
            reads.append(tuple(keys))
            return await original(keys)

        monkeypatch.setattr(first.archive.store, "read_records", capture)
        sql_reads = []
        engine = state.execution.executions.state_store.context.engine.sync_engine

        def capture_sql(connection, cursor, statement, parameters, execution_context, many):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                sql_reads.append(statement)

        event.listen(engine, "before_cursor_execute", capture_sql)
        try:
            reused = await service.prepare(context, history_root=root,
                                           history_digest=hashlib.sha256(context.history).hexdigest())
        finally:
            event.remove(engine, "before_cursor_execute", capture_sql)
        assert not sql_reads, "verified native prefix reuse must not issue SQL reads"
        assert not reads
        assert not reused.records.records and not reused.records.native_messages
        assert reused.reference == first.reference
        assert await service.load(reused.reference) == context
        changed_messages = list(context.model_messages())
        changed_messages[-1].parts[0].content = "different"
        changed = ExecutionInputContext.from_messages(changed_messages)
        retained = await service.prepare(changed, history_root=root,
                                         history_digest=hashlib.sha256(context.history).hexdigest())
        assert retained.records.native_messages
        assert retained.reference != first.reference
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "sqlite",))
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
        await _publish_context(state, prepared)
        assert await service._open() is prepared.archive
        assert await service.load(prepared.reference) == context
        assert calls == [False, True]
        other = ExecutionContextArchive(store, objects, namespace="other", tenant_id="tenant")
        with pytest.raises(AIError):
            await other.load(prepared.reference)
    finally:
        await state.close()
    with pytest.raises(AIError) as closed:
        await service.load(prepared.reference)
    assert closed.value.code is ErrorCode.STORAGE_CLOSED
    with pytest.raises(AIError) as closed:
        await service.prepare(context)
    assert closed.value.code is ErrorCode.STORAGE_CLOSED


@pytest.mark.parametrize("has_instructions", (False, True))
async def test_instruction_read_does_not_materialize_history_or_other_components(tmp_path, monkeypatch, has_instructions):
    from linktools.ai.persistence import SqlStepStore
    from linktools.ai.persistence.runs import ContentArchive
    from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions

    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="context-instructions", tenant_id="tenant")
    try:
        service = ExecutionContextArchive(
            state.execution.executions.state_store,
            state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="context-instructions", tenant_id="tenant",
        )
        instructions = (RepositoryInstructions((RepositoryInstructionDocument("agents:AGENTS.md", ".", "accepted rule"),)).to_payload()
                        if has_instructions else None)
        context = ExecutionInputContext.from_messages(
            tuple(ModelRequest(parts=[UserPromptPart(f"history-{index}")]) for index in range(3)),
            repository_instructions=instructions, session_metadata={"large": "metadata"},
            memory={"large": {"content": "memory"}}, preload_skills=({"large": "skill"},),
        )
        prepared = await service.prepare(context)
        await _publish_context(state, prepared)
        cold = ExecutionContextArchive(
            state.execution.executions.state_store,
            state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="context-instructions", tenant_id="tenant",
        )
        reads = []
        original = ContentArchive.read_contents

        async def read_contents(self, identities, *, codec, reserve=None):
            reads.extend(identities)
            return await original(self, identities, codec=codec, reserve=reserve)

        async def reject_history(*args, **kwargs):
            raise AssertionError("instruction-only reads must not read history")

        monkeypatch.setattr(ContentArchive, "read_contents", read_contents)
        monkeypatch.setattr(ContentArchive, "read_ranges", reject_history)
        monkeypatch.setattr(SqlStepStore, "read_message_roots", reject_history)
        assert await cold.load_repository_instructions(prepared.reference) == instructions
        component = prepared.reference.payload.decode()["repository_instructions"]
        assert reads == ([] if component is None else [component["content_id"]])
        if component is not None:
            for name, invalid in (("archive_scope", "other"), ("content_scope", "other"), ("content_id", "0" * 64)):
                manifest = prepared.reference.payload.decode()
                manifest["repository_instructions"][name] = invalid
                malformed = replace(prepared.reference, payload=StoredPayload.inline_json(manifest))
                with pytest.raises(AIError) as failure:
                    await cold.load_repository_instructions(malformed)
                assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("has_instructions", (False, True))
async def test_pending_inline_context_instructions_need_no_archive(has_instructions):
    from linktools.ai.execution import RuntimePayloadRef
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions

    instructions = (RepositoryInstructions((RepositoryInstructionDocument("agents:AGENTS.md", ".", "pending rule"),))
                    if has_instructions else None)
    context = ExecutionInputContext.from_messages((), repository_instructions=(
        None if instructions is None else instructions.to_payload()))
    reference = RuntimePayloadRef(StoredPayload.inline_json(context.to_payload()), PersistenceDomain.EXECUTION)
    assert await LocalExecutionBackend.load_input_repository_instructions(object(), reference) == instructions
