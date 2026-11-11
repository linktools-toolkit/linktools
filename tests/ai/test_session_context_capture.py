#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session continuation shares one source metadata capture across its readers."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._conversation_repositories import ConversationHistoryRepositoryImpl
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.contracts import ContextProjection

from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage
from .test_session_history_projection import _session

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("shape", ("empty", "projected", "cleared"))
async def test_session_context_reuses_one_metadata_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="context-capture", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        history_id = session.history_id
        assert history_id is not None
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        repository = archive.transcript_repository
        summary = ModelRequest(parts=[UserPromptPart("retained summary")])
        checkpoint = replace(_checkpoint(2), context_messages=[summary])
        if shape != "empty":
            await archive.materialize_checkpoint(
                AgentRunRecord("run", metadata={"history_id": history_id}), checkpoint,
            )
        if shape == "cleared":
            await archive.state_store.mutate(lambda tx: repository.store_projection(
                tx, history_id, ContextProjection(()),
            ))
        keys = {
            *state.conversation.histories.history_record_keys(history_id),
            repository.projection_key(history_id),
        }
        metadata_queries = []
        statements = []

        def capture(_connection, _cursor, statement, parameters, _context, _many):
            statements.append(statement.split()[0].upper())
            if statement.startswith("SELECT") and keys.intersection(parameters):
                metadata_queries.append(set(parameters).intersection(keys))

        async def uncaptured(_history_id, *, tenant_id):
            return None

        # Compare query batching with the same opened physical archive owner.
        # Cold scope discovery is covered by test_context_metadata_read_io.
        assert await archive.state_store.storage_group.open_archive() is not None
        results, metadata_counts, select_counts = [], [], []
        engine = archive.state_store.storage_group.context.engine.sync_engine
        for use_capture in (False, True):
            statements.clear()
            metadata_queries.clear()
            with monkeypatch.context() as patch:
                if not use_capture:
                    patch.setattr(repository, "capture_session_context_metadata", uncaptured)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    result = await archive.load_loaded_model_context(owner_id=history_id)
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            results.append(result)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            metadata_counts.append(len(metadata_queries))
            select_counts.append(statements.count("SELECT"))
            if use_capture:
                assert metadata_queries == [keys]
        assert results[0] == results[1]
        expected = (summary,) if shape == "projected" else ()
        assert results[1].model_messages() == expected
        assert metadata_counts == ([2, 1] if shape == "empty" else [1, 1])
        assert select_counts[0] - select_counts[1] == (1 if shape == "empty" else 0)

        if shape == "projected":
            await archive.state_store.mutate(lambda tx: repository.store_projection(
                tx, history_id, ContextProjection(()),
            ))
            changed = await archive.load_loaded_model_context(owner_id=history_id)
            assert changed.model_messages() == ()
    finally:
        await state.close()


async def test_session_context_keeps_custom_history_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="context-capture", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        history_id = session.history_id
        assert history_id is not None
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        repository = archive.transcript_repository
        reads = []

        class History(ConversationHistoryRepositoryImpl):
            async def get(self, history_id, *, tenant_id):
                reads.append(history_id)
                return await super().get(history_id, tenant_id=tenant_id)

        custom = History(archive.state_store, namespace="context-capture", tenant_id="tenant")
        monkeypatch.setattr(repository, "_history_repository", custom)
        assert await repository.capture_session_context_metadata(history_id, tenant_id="tenant") is None
        assert not (await archive.load_loaded_model_context(owner_id=history_id)).messages
        assert reads == [history_id]
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("missing_history", "missing_head", "missing_checkpoint", "interrupted", "past_head"))
async def test_session_context_capture_preserves_source_checks(tmp_path: Path, failure: str) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="context-capture", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        history_id = session.history_id
        assert history_id is not None
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        if failure != "missing_checkpoint":
            await archive.materialize_checkpoint(
                AgentRunRecord("run", metadata={"history_id": history_id}),
                replace(_checkpoint(2), state="interrupted" if failure == "interrupted" else "complete"),
            )
        digest = (await archive.checkpoint_at_sequence(agent_run_id="run", checkpoint_sequence=1)).projection_digest if failure != "missing_checkpoint" else "missing"
        history_key, head_key = state.conversation.histories.history_record_keys(history_id)
        if failure in {"missing_history", "missing_head"}:
            key = history_key if failure == "missing_history" else head_key
            await archive.state_store.mutate(lambda tx: tx.delete_record(key))
        with pytest.raises(AIError) as raised:
            if failure in {"missing_history", "missing_head"}:
                await archive.transcript_repository.load_session_raw_model_context(
                    history_id, tenant_id="tenant", message_count=2,
                )
            elif failure == "past_head":
                await archive.transcript_repository.load_session_raw_model_context(
                    history_id, tenant_id="tenant", message_count=3,
                )
            else:
                await archive.load_committed_session_model_context(
                    agent_run_id="run", checkpoint_sequence=1,
                    projection_digest=digest, tenant_id="tenant",
                )
        assert raised.value.code == (ErrorCode.STORAGE_INTEGRITY_ERROR if failure in {"missing_head", "past_head"}
                                    else ErrorCode.SESSION_HISTORY_UNAVAILABLE)
    finally:
        await state.close()
