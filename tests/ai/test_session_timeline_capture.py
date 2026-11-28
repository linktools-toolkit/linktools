#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session pages bind captured history metadata without caching transcript facts."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._step_schema import native_message_sql_tables
from linktools.ai.persistence.contracts import ConversationCursor, ConversationHistoryRecord
from linktools.ai.persistence._conversation_repositories import ConversationHistoryRepositoryImpl
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.steps import native_store_key

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

from .test_session_refresh import _PRINCIPAL, _admit, _service
from .test_session_timeline import _materialize_conversation, _session

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def committed_memory_state() -> AsyncIterator[tuple[Persistence, str]]:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-timeline", tenant_id="tenant")
    try:
        created = await state.conversation.sessions.create(_session())
        await _admit(state, "success", ExecutionStatus.SUCCEEDED)
        run_id, count = await _materialize_conversation(state, created.history_id)
        sessions = state.conversation.sessions

        async def commit(transaction):
            await sessions.commit_timeline_turn_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="success",
                start_message_index=0, end_message_index=count,
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


class _HostAuthorization(TenantAuthorizationPolicy):
    """The SQL host authorizes access to these native sessions."""


async def _start_native(runtime, key):
    turn = await runtime.sessions.start_turn(
        "session", key, agent="default", principal=_PRINCIPAL, idempotency_key=key,
    )
    assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    return turn.key.turn_id


@pytest_asyncio.fixture
async def committed_state(tmp_path: Path):
    namespace = "session-timeline"
    storage = PersistenceStorage.sqlite(tmp_path / "capture.db")
    features = RuntimeFeatures(namespace)
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def model(messages, _info):
        prompt = messages[-1].parts[-1].content
        return ModelResponse(parts=[TextPart("visible answer" if prompt == "first" else "new suffix")])

    model_features(_Models(_UsageFunctionModel(model)), features)
    policy = _HostAuthorization()
    async with Runtime.open(
        namespace, storage=storage, tenant_id="tenant", features=features, authorization=policy,
    ) as writer:
        await writer.sessions.create_turn_session("session", principal=_PRINCIPAL)
        first_id = await _start_native(writer, "first")
    # Real cold reader: do not reuse the original runtime or its in-memory roots.
    async with (
        Persistence.open(namespace, storage=storage, tenant_id="tenant") as state,
        Runtime.open(persistence=state, features=features, authorization=policy) as runtime,
    ):
        yield state, runtime, first_id


async def test_session_refresh_keeps_cold_committed_page_within_seven_selects(committed_state) -> None:
    state, runtime, first_id = committed_state
    engine = state.session_turn_reader._context.engine
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        detail = await runtime.history.turn_session("session", principal=_PRINCIPAL)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    items = detail["timeline"]["items"]
    assert [turn["turn_id"] for turn in items] == [first_id]
    assert items[0]["output"] == {"text": "visible answer"}
    assert 0 < sum(value.lstrip().upper().startswith("SELECT") for value in statements) <= 7


@pytest.mark.parametrize("mutation", ("append", "delete_facts"))
async def test_session_page_keeps_captured_prefix_but_reads_current_facts(
    committed_state, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    state, runtime, first_id = committed_state
    reader = state.session_turn_reader
    page = reader.turn_page
    later = None

    async def change_after_capture(session_id, **kwargs):
        nonlocal later
        assert active_state_scope() is None and kwargs["through"] == 1
        if mutation == "append":
            later = await _start_native(runtime, "second")
        else:
            # Simulate corruption after the head capture and before the exact
            # transcript read. A native page must not hide absent facts.
            sequences = native_message_sql_tables()["ai_message_sequences"]
            async with reader._context.engine.begin() as connection:
                result = await connection.execute(sequences.delete().where(
                    sequences.c.store_key == native_store_key("session-timeline", "tenant"),
                ))
                assert result.rowcount > 0
        return await page(session_id, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(reader, "turn_page", change_after_capture)
        if mutation == "delete_facts":
            with pytest.raises(AIError) as error:
                await runtime.history.turn_session("session", principal=_PRINCIPAL)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            return
        detail = await runtime.history.turn_session("session", principal=_PRINCIPAL)
        items = detail["timeline"]["items"]
        assert [turn["turn_id"] for turn in items] == [first_id]
        assert items[0]["output"] == {"text": "visible answer"}
    current = (await runtime.history.turn_session("session", principal=_PRINCIPAL))["timeline"]["items"]
    assert [item["turn_id"] for item in current] == [first_id, later]
    assert current[-1]["output"] == {"text": "new suffix"}


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
    committed_memory_state, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, history_id = committed_memory_state
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
