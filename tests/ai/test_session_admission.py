#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session admission ownership and status transition conformance."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_digest="a" * 64,
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )


async def _assert_admission_contract(state: RuntimeState) -> None:
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
        assert admitted.resource_generation == original.resource_generation
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
            resource_generation=admitted.resource_generation + 1,
            updated_at=datetime.now(timezone.utc),
        )
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
    await _assert_admission_contract(RuntimeState.in_memory())


@pytest.mark.asyncio
async def test_sql_session_admission_contract(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    try:
        await _assert_admission_contract(RuntimeState.sql(engine))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_memory_admission_is_single_owner() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="admission-race", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())

        async def attempt(execution_id: str) -> ErrorCode | None:
            try:
                await state.conversation.sessions.admit_execution(
                    "session",
                    tenant_id="tenant",
                    execution_id=execution_id,
                    expected=None,
                )
            except AIError as error:
                return error.code
            return None

        outcomes = await asyncio.gather(attempt("execution-1"), attempt("execution-2"))
        assert sorted(outcomes, key=lambda item: item is not None) == [None, ErrorCode.SESSION_BUSY]
    finally:
        await state.close()
