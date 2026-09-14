#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline storage invariants."""

from datetime import datetime, timezone

import pytest

from linktools.ai.core import SessionStatus
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._contracts import SessionRecord


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
