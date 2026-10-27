#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session pages bind captured history metadata without caching transcript facts."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import PersistenceDomain, Persistence
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence.contracts import ConversationCursor, ConversationHistoryRecord
from linktools.ai.persistence._conversation_repositories import ConversationHistoryRepositoryImpl
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence.state import active_state_scope

from .test_session_refresh import _PRINCIPAL, _admit, _service
from .test_session_timeline import _commit_fields, _materialize_conversation, _session

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def committed_state(tmp_path: Path) -> AsyncIterator[tuple[Persistence, str]]:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "capture.db"))
    await state.initialize(namespace="session-timeline", tenant_id="tenant")
    try:
        created = await state.conversation.sessions.create(_session())
        await _admit(state, "success", ExecutionStatus.SUCCEEDED)
        run_id, count = await _materialize_conversation(state, created.history_id)
        sessions = state.conversation.sessions
        execution = await state.execution.executions.get("success", tenant_id="tenant")

        async def commit(transaction):
            await sessions.commit_timeline_turn_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="success",
                start_message_index=0, end_message_index=count, **_commit_fields(execution),
            )
            await sessions.advance_continuation_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="success", expected=None,
                next_cursor=ConversationCursor(run_id, history_id=created.history_id, message_count=count),
                release_execution=True, history_quality="complete",
            )

        await sessions.state_store.mutate(commit)
        yield state, created.history_id
    finally:
        await state.close()


async def test_session_refresh_keeps_cold_committed_page_within_seven_selects(committed_state) -> None:
    state, _history_id = committed_state
    engine = state.conversation.sessions.state_store.storage_group.context.engine
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        detail = await _service(state).refresh("session", principal=_PRINCIPAL)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert [turn.execution_id for turn in detail.timeline.items] == ["success"]
    assert sum(value.lstrip().upper().startswith("SELECT") for value in statements) <= 7


@pytest.mark.parametrize("mutation", ("append", "delete_facts"))
async def test_session_page_keeps_captured_prefix_but_reads_current_facts(
    committed_state, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    state, history_id = committed_state
    persistence = state.run_store.conversation_persistence
    transcript = persistence.archive.transcript_repository
    read = persistence.iter_conversation_message_range

    async def change_after_capture(**kwargs):
        assert active_state_scope() is None
        if mutation == "append":
            prepared = PreparedArchive()
            chunks = await transcript.prepare_chunks(
                history_id, (ModelResponse(parts=[TextPart("new suffix")]),),
                first_message_index=2, prepared=prepared,
            )
            await transcript.publish_archive(prepared)
            await persistence.archive.state_store.mutate(
                lambda transaction: transcript.append_chunks(transaction, history_id, chunks)
            )
        else:
            await persistence.archive.state_store.mutate(
                lambda transaction: transaction.delete_fact_streams(transcript.head_key(history_id))
            )
        async for message in read(**kwargs):
            assert active_state_scope() is None
            yield message

    monkeypatch.setattr(persistence, "iter_conversation_message_range", change_after_capture)
    if mutation == "delete_facts":
        with pytest.raises(AIError) as error:
            await _service(state).refresh("session", principal=_PRINCIPAL)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    else:
        detail = await _service(state).refresh("session", principal=_PRINCIPAL)
        contents = [item.content for turn in detail.timeline.items for item in turn.items]
        assert "visible answer" in contents
        assert "new suffix" not in contents


@pytest.mark.parametrize("mismatch", ("namespace", "tenant", "owner"))
async def test_history_capture_rejects_same_identity_from_another_scope(mismatch: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="capture-scope", tenant_id="tenant")
    try:
        histories = state.conversation.histories
        await histories.create(ConversationHistoryRecord("history", "session", None, None, 0))
        captured = await histories.get_with_head("history", tenant_id="tenant")
        source = histories
        if mismatch == "owner":
            source = ConversationHistoryRepositoryImpl(
                histories.state_store, namespace="capture-scope", tenant_id="tenant",
            )
        transcript = TranscriptRepository(
            histories.state_store, object_store=None,
            namespace="other" if mismatch == "namespace" else "capture-scope",
            tenant_id="other" if mismatch == "tenant" else "tenant",
            runtime_domain=PersistenceDomain.CONVERSATION, history_repository=histories,
        )
        with pytest.raises(AIError) as error:
            _ = [message async for message in transcript.iter_session_message_range(
                "history", tenant_id="other" if mismatch == "tenant" else "tenant",
                start=0, end=0, history_capture=(source, *captured),
            )]
        assert error.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    finally:
        await state.close()


async def test_session_capture_preserves_custom_transcript_history_owner(
    committed_state, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, history_id = committed_state
    persistence = state.run_store.conversation_persistence
    calls = []

    class CustomHistory(ConversationHistoryRepositoryImpl):
        async def get_in_transaction(self, transaction, history_id, *, tenant_id):
            calls.append(history_id)
            return await super().get_in_transaction(transaction, history_id, tenant_id=tenant_id)

        async def get_with_head_in_transaction(self, transaction, history_id, *, tenant_id):
            pytest.fail("the custom history owner's scalar contract must be retained")

    histories = CustomHistory(
        persistence.archive.state_store, namespace="session-timeline", tenant_id="tenant",
    )
    monkeypatch.setattr(persistence.archive.transcript_repository, "_history_repository", histories)
    detail = await _service(state).refresh("session", principal=_PRINCIPAL)
    assert [turn.execution_id for turn in detail.timeline.items] == ["success"]
    assert calls == [history_id]
