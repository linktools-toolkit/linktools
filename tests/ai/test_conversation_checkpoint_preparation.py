#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation preparation reuses only one pre-body metadata capture."""

from linktools.ai.runtime.state import RuntimeState
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._store import active_state_scope

from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("existing", ("absent", "replay", "head_without_run"))
async def test_conversation_preparation_reuses_new_run_metadata(
    tmp_path: Path, existing: str,
) -> None:
    state = _storage("sqlite", tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="conversation-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
    run = AgentRunRecord("run")
    engine = archive.state_store.storage_group.context.engine.sync_engine
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.split()[0].upper())

    try:
        if existing != "absent":
            await archive.materialize_checkpoint(run, _checkpoint(2))
            if existing == "head_without_run":
                await archive.state_store.mutate(lambda tx: tx.delete_record(archive._agent_run_key("run")))
        batches, counts = [], []
        for composed in (False, True):
            statements.clear()
            event.listen(engine, "before_cursor_execute", capture)
            try:
                if composed:
                    relocated, batch = await archive.prepare_conversation_checkpoint(run, _checkpoint(2))
                else:
                    relocated = await archive.relocate_conversation_checkpoint(run, _checkpoint(2))
                    batch = await archive.prepare_checkpoints(run, (relocated,))
                batches.append((relocated, batch))
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            counts.append(statements.count("SELECT"))
        assert batches[0] == batches[1]
        assert counts[1] == counts[0] - (existing != "replay")
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("override", ("relocate", "prepare"))
async def test_conversation_preparation_preserves_custom_owner_operations(
    tmp_path: Path, override: str,
) -> None:
    state = _storage("memory", tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="conversation-preparation", tenant_id="tenant")
    original = state.run_store.read_store(RuntimeDomain.CONVERSATION)
    run = AgentRunRecord("run")

    class RelocatingArchive(ModelInteractionStateStepArchive):
        async def relocate_conversation_checkpoint(self, run, checkpoint):
            relocated = await super().relocate_conversation_checkpoint(run, checkpoint)
            return replace(relocated, step_index=19)

    class PreparingArchive(ModelInteractionStateStepArchive):
        async def prepare_checkpoints(self, run, checkpoints, **kwargs):
            batch = await super().prepare_checkpoints(run, checkpoints, **kwargs)
            return replace(batch, target_transcript_message_count=19)

    custom = (RelocatingArchive if override == "relocate" else PreparingArchive)(
        original.state_store, object_store=None, namespace="conversation-preparation",
        tenant_id="tenant", runtime_domain=RuntimeDomain.CONVERSATION,
    )
    try:
        checkpoint = _checkpoint(1)
        expected_checkpoint = await custom.relocate_conversation_checkpoint(run, checkpoint)
        expected = await custom.prepare_checkpoints(run, (expected_checkpoint,))
        relocated, actual = await custom.prepare_conversation_checkpoint(run, checkpoint)
        assert relocated == expected_checkpoint and actual == expected
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_conversation_preparation_keeps_fresh_commit_guard(
    tmp_path: Path, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="conversation-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
    run = AgentRunRecord("run")
    try:
        _, batch = await archive.prepare_conversation_checkpoint(run, _checkpoint(2))
        await archive.materialize_checkpoint(run, _checkpoint(1))
        with pytest.raises(AIError) as raised:
            await archive.state_store.mutate(
                lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, batch.checkpoints[0])
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert active_state_scope() is None
    finally:
        await state.close()
