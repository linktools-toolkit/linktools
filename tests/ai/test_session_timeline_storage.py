#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline storage invariants."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._commands import _timeline_turn_message_range
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ConversationHistoryRecord,
    SessionRecord,
)
from linktools.ai.runtime.state._store import StoredFact


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        agent_id="agent",
    )


@pytest.mark.asyncio
async def test_committed_turn_range_may_skip_uncommitted_turns() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-timeline-storage", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="failed",
            expected=None,
        )
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="failed",
        )
        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="success",
            expected=None,
        )
        assert await state.conversation.sessions.timeline_head(
            "session", tenant_id="tenant"
        ) == 2

        async def commit(transaction) -> None:
            await state.conversation.sessions.commit_timeline_turn_in_transaction(
                transaction,
                "session",
                tenant_id="tenant",
                execution_id="success",
                start_message_index=0,
                end_message_index=2,
            )

        await state.conversation.sessions.state_store.mutate(commit)
        committed = await state.conversation.sessions.list_timeline_commits(
            "session",
            tenant_id="tenant",
            start_sequence=1,
            end_sequence=3,
        )
        assert [(item.sequence, item.execution_id) for item in committed] == [
            (2, "success")
        ]
    finally:
        await state.close()


def test_timeline_range_uses_committed_cursor_when_snapshot_is_already_materialized() -> None:
    history = ConversationHistoryRecord(
        history_id="history",
        session_id="session",
        tenant_id="tenant",
        parent_history_id="parent",
        prefix_index_head_id="node",
        inherited_message_count=3,
    )
    prepared = SimpleNamespace(
        snapshots=(SimpleNamespace(chunks=()),),
        target_transcript_message_count=5,
    )
    expected = ConversationCursor(
        "previous",
        history_id="history",
        message_count=4,
    )

    assert _timeline_turn_message_range(history, prepared, expected) == (4, 8)


def test_timeline_range_allows_root_recovery_after_transcript_materialization() -> None:
    history = ConversationHistoryRecord(
        history_id="history",
        session_id="session",
        tenant_id="tenant",
        parent_history_id=None,
        prefix_index_head_id=None,
        inherited_message_count=0,
    )
    prepared = SimpleNamespace(
        snapshots=(SimpleNamespace(chunks=()),),
        target_transcript_message_count=2,
    )

    assert _timeline_turn_message_range(history, prepared, None) == (0, 2)

@pytest.mark.asyncio
async def test_timeline_commit_rejects_duplicate_admission_fact() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-timeline-duplicate", tenant_id="tenant")
    try:
        repository = state.conversation.sessions
        await repository.create(_session())
        await repository.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="duplicate",
            expected=None,
        )

        async def duplicate(transaction) -> None:
            sequence = await transaction.next_sequence(
                repository._timeline_sequence_key("session")
            )
            await transaction.insert_fact(
                StoredFact(
                    repository._timeline_stream("session"),
                    sequence,
                    repository._key("session", "session"),
                    "session_turn",
                    repository._timeline_subject("duplicate"),
                    None,
                    {"version": 1, "execution_id": "duplicate"},
                )
            )

        await repository.state_store.mutate(duplicate)

        async def commit(transaction) -> None:
            await repository.commit_timeline_turn_in_transaction(
                transaction,
                "session",
                tenant_id="tenant",
                execution_id="duplicate",
                start_message_index=0,
                end_message_index=2,
            )

        with pytest.raises(AIError) as captured:
            await repository.state_store.mutate(commit)
        assert captured.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
