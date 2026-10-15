#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation appends fence once when pending head metadata is unchanged."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("transition", ("complete", "clear_pending", "same_shell"))
async def test_observation_append_preserves_pending_facts_and_stale_fence(
    tmp_path: Path, backend: str, transition: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="observation-write-io", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        store = archive.state_store
        repository = archive.transcript_repository
        await repository.create_head("run")
        first = ModelResponse(parts=[TextPart("first")])
        second = replace(first, parts=[TextPart("second")])
        if transition != "complete":
            initial = await repository.prepare_observation(
                "run", (), first_message_index=0, pending=first, pending_keys=("part:0",),
            )
            await store.mutate(lambda tx: repository.commit_observation(tx, initial))
        before = await repository.get_head("run")
        prepared = await repository.prepare_observation(
            "run", (first,), first_message_index=0,
            pending=second if transition == "same_shell" else None,
            pending_keys=("part:0",) if transition == "same_shell" else (),
        )
        statements = []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        engine = store.storage_group.context.engine.sync_engine if backend == "sqlite" else None
        if engine is not None:
            event.listen(engine, "before_cursor_execute", capture)
        try:
            await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        if engine is not None:
            updates = [sql for sql in statements if sql.startswith("UPDATE ai_state_records")]
            assert len(updates) == (2 if transition == "clear_pending" else 1)
        assert await repository.load_messages("run") == (first,)
        head = await repository.get_head("run")
        assert head.message_count == 1
        if transition == "same_shell":
            assert head.pending == before.pending
            assert head.pending_part_count == before.pending_part_count == 1
            parts, keys = await store.read(lambda tx: repository.capture_pending_in_transaction(tx, head))
            restored = await repository.read_pending_message(head.pending, parts)
            assert restored == second and keys == ("part:0",)
            assert prepared.new_pending_parts
        else:
            assert head.pending is None and head.pending_part_count == 0
        assert await repository.verify_observation(prepared)
        with pytest.raises(AIError) as stale:
            await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("failure", ("cas", "body"))
async def test_observation_append_rollback_keeps_original_head_and_retry(
    tmp_path: Path, backend: str, failure: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="observation-rollback", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        store, repository = archive.state_store, archive.transcript_repository
        await repository.create_head("run")
        before = await repository.get_head("run")
        message = ModelResponse(parts=[TextPart("complete")])
        prepared = await repository.prepare_observation(
            "run", (message,), first_message_index=0, pending=None, pending_keys=(),
        )
        async def abort(tx):
            with pytest.MonkeyPatch.context() as patch:
                if failure == "cas":
                    async def reject(*args, **kwargs):
                        return False
                    patch.setattr(tx, "replace_record", reject)
                await repository.commit_observation(tx, prepared)
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "abort")
        with pytest.raises(AIError) as aborted:
            await store.mutate(abort)
        assert aborted.value.code is (ErrorCode.STORAGE_CONFLICT if failure == "cas" else ErrorCode.STORAGE_INTEGRITY_ERROR)
        assert await repository.get_head("run") == before
        assert await repository.load_messages("run") == ()
        assert not await repository.verify_observation(prepared)
        await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        assert await repository.load_messages("run") == (message,)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_empty_unchanged_observation_still_requires_head_guard(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="observation-guard", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        store, repository = archive.state_store, archive.transcript_repository
        await repository.create_head("run")
        prepared = await repository.prepare_observation(
            "run", (), first_message_index=0, pending=None, pending_keys=(),
        )
        async def commit(tx):
            async def fail_guard(*args, **kwargs):
                return None
            monkeypatch.setattr(tx, "guard_record", fail_guard)
            await repository.commit_observation(tx, prepared)
        with pytest.raises(AIError) as conflict:
            await store.mutate(commit)
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()
