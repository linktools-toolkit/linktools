#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session admission ownership and status transition conformance."""

from linktools.ai.persistence import Persistence
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.contracts import ConversationCursor, SessionRecord
from linktools.ai.persistence.turns import SessionTurnKey

from .test_session_turn_sql import _TRANSCRIPT, accept, store


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )


async def _assert_admission_contract(state: PersistenceStorage) -> None:
    state = Persistence(state)
    await state.initialize(namespace="admission", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        original = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert original is not None

        admitted = await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution-1",
            expected=None,
        )
        assert admitted.active_execution_id == "execution-1"
        assert admitted.revision == original.revision
        assert admitted.updated_at == original.updated_at

        with pytest.raises(AIError) as busy:
            await state.conversation.sessions.admit_execution(
                "session",
                tenant_id="tenant",
                execution_id="execution-2",
                expected=None,
            )
        assert busy.value.code is ErrorCode.SESSION_BUSY

        metadata_update = replace(
            admitted,
            metadata={"key": "value"},
            revision=admitted.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        for identity_change in (
            {"session_id": "different-session"},
            {"owner_principal_id": "different-owner"},
            {"agent_id": "different-agent"},
            {"history_id": "different-history"},
            {
                "created_at": datetime.fromtimestamp(
                    admitted.created_at.timestamp() + 1,
                    tz=timezone.utc,
                )
            },
        ):
            with pytest.raises(AIError) as identity_error:
                await state.conversation.sessions.compare_and_swap(
                    admitted.session_id,
                    tenant_id="tenant",
                    expected_revision=admitted.revision,
                    next_record=replace(
                        admitted,
                        revision=admitted.revision + 1,
                        updated_at=datetime.now(timezone.utc),
                        **identity_change,
                    ),
                )
            assert identity_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        updated = await state.conversation.sessions.compare_and_swap(
            "session",
            tenant_id="tenant",
            expected_revision=admitted.revision,
            next_record=metadata_update,
        )
        assert updated.active_execution_id == "execution-1"

        closing = await state.conversation.sessions.transition_status(
            "session",
            tenant_id="tenant",
            expected=frozenset({SessionStatus.OPEN}),
            next_status=SessionStatus.CLOSING,
        )
        assert closing.active_execution_id == "execution-1"
        advanced = await state.conversation.sessions.advance_continuation(
            "session",
            tenant_id="tenant",
            execution_id="execution-1",
            expected=None,
            next_cursor=ConversationCursor("run-1"),
        )
        assert advanced.status is SessionStatus.CLOSING
        assert advanced.revision == closing.revision + 1
        assert advanced.active_execution_id == "execution-1"

        released = await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution-1",
        )
        assert released.active_execution_id is None
        assert released.revision == advanced.revision
        assert released.updated_at == advanced.updated_at

        stale_release = await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution-2",
        )
        assert stale_release.active_execution_id is None

        closed = await state.conversation.sessions.transition_status(
            "session",
            tenant_id="tenant",
            expected=frozenset({SessionStatus.CLOSING}),
            next_status=SessionStatus.CLOSED,
            closed_at=datetime.now(timezone.utc),
            require_no_active=True,
        )
        assert closed.active_execution_id is None
        assert closed.continuation == ConversationCursor("run-1")
        with pytest.raises(AIError) as closed_error:
            await state.conversation.sessions.admit_execution(
                "session",
                tenant_id="tenant",
                execution_id="execution-3",
                expected=ConversationCursor("run-1"),
            )
        assert closed_error.value.code is ErrorCode.SESSION_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_session_admission_contract() -> None:
    await _assert_admission_contract(PersistenceStorage.in_memory())


@pytest.mark.asyncio
async def test_sql_session_admission_contract(store) -> None:
    _engine, _metadata, turns = store
    assert (await turns.get_session("chat"))["revision"] == 0
    first = SessionTurnKey("chat", "first")
    assert await accept(turns, first.turn_id, request="first-request")
    acquired = await turns.get_session("chat")
    assert acquired["active_turn_id"] == b"first" and acquired["revision"] == 1
    assert not await accept(turns, "competing", revision=1, request="competing-request")
    assert not await turns.close_session("chat", expected_revision=1)
    assert not await turns.update_session("chat", expected_revision=1, config={"metadata": {"replacement": True}})
    assert await turns.get_session("chat") == acquired

    assert await turns.finish(first, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
    released = await turns.get_session("chat")
    assert released["active_turn_id"] is None and released["last_turn_no"] == 1
    assert released["revision"] == 2
    assert not await turns.finish(first, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
    assert await turns.update_session("chat", expected_revision=2, config={"metadata": {"updated": True}})
    assert not await turns.close_session("chat", expected_revision=2)
    assert await turns.close_session("chat", expected_revision=3)
    closed = await turns.get_session("chat")
    assert closed["active_turn_id"] is None and closed["status"] == "closed"
    assert not await accept(turns, "after-close", revision=closed["revision"], through=1, request="later")
    assert await turns.get_session("chat") == closed
    committed, = await turns.turn_page("chat", through=1, limit=10)
    assert committed["turn_id"] == b"first" and committed["messages"][0].text == "two"
