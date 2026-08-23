#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Committed Session history projection and continuation behavior."""

from datetime import datetime, timezone

import pytest
from linktools.ai.adapter import StepSessionHistoryReader
from linktools.ai.core import (
    HmacCursorSigner,
    Principal,
    SessionStatus,
    TenantAuthorizationPolicy,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import (
    DefaultSessionService,
    ForkSessionRequest,
    RuntimeState,
)
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy.ext.asyncio import create_async_engine


def _session(session_id: str = "session") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id=session_id,
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


def _reader(state: RuntimeState) -> StepSessionHistoryReader:
    return StepSessionHistoryReader(
        store=state.steps.read_store(RuntimeDomain.CONVERSATION),
        cursor_signer=HmacCursorSigner("session-history", b"session-history-key"),
    )


async def _advance(
    state: RuntimeState,
    expected: ConversationCursor | None,
    next_cursor: ConversationCursor,
) -> None:
    execution_id = "history-execution"
    await state.conversation.sessions.admit_execution(
        "session",
        tenant_id="tenant",
        execution_id=execution_id,
        expected=expected,
    )
    try:
        await state.conversation.sessions.advance_continuation(
            "session",
            tenant_id="tenant",
            execution_id=execution_id,
            expected=expected,
            next_cursor=next_cursor,
        )
    finally:
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id=execution_id,
        )


def _service(state: RuntimeState) -> DefaultSessionService:
    return DefaultSessionService(
        state.conversation,
        state.execution.executions,
        TenantAuthorizationPolicy(),
        object(),
        HmacCursorSigner("session", b"session-key"),
        history_reader=_reader(state),
    )


async def _materialize(
    state: RuntimeState,
    run_id: str,
    prompts: tuple[str, ...],
    messages: list[object] | None = None,
) -> None:
    conversation_id = step_conversation_id(
        namespace="session-history",
        tenant_id="tenant",
        execution_id=run_id,
    )
    now = datetime.now(timezone.utc)
    await state.steps.register_run(
        RunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            metadata={"agent_name": "default"},
            started_at=now,
        )
    )
    snapshot_messages = messages or []
    if not snapshot_messages:
        for prompt in prompts:
            snapshot_messages.extend(
                (
                    ModelRequest(
                        parts=[UserPromptPart(content=prompt)],
                        conversation_id=conversation_id,
                    ),
                    ModelResponse(
                        parts=[TextPart(content=f"answer:{prompt}")],
                        conversation_id=conversation_id,
                    ),
                )
            )
    await state.steps.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=len(snapshot_messages),
            messages=snapshot_messages,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            timestamp=now,
            state="complete",
        )
    )
    await state.steps.materialize_conversation(step_run_id=run_id)


@pytest.mark.asyncio
async def test_empty_and_committed_session_history_use_continuation_only() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-history", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        service = _service(state)
        principal = Principal("owner", "tenant")

        empty = await service.history("session", principal=principal)
        assert empty.items == ()
        assert empty.next_cursor is None
        with pytest.raises(AIError) as error:
            await service.history(
                "session",
                principal=principal,
                cursor="stale",
            )
        assert error.value.code is ErrorCode.CURSOR_INVALID

        run_id = step_run_id(
            namespace="session-history",
            tenant_id="tenant",
            execution_id="execution",
            segment_sequence=1,
        )
        await _materialize(state, run_id, ('  {"question":"你好\\nworld"}  ',))
        await _advance(state, None, ConversationCursor(run_id))

        page = await service.history("session", principal=principal)
        assert [(item.item_kind, item.content) for item in page.items] == [
            ("user", '  {"question":"你好\\nworld"}  '),
            ("assistant", 'answer:  {"question":"你好\\nworld"}  '),
        ]
        assert [item.sequence for item in page.items] == [1, 2]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_cursor_binds_to_current_continuation() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-history-cursor", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        first_run = "session-history-first"
        second_run = "session-history-second"
        await _materialize(state, first_run, ("A", "B"))
        await _advance(state, None, ConversationCursor(first_run))
        service = _service(state)
        first_page = await service.history("session", principal=Principal("owner", "tenant"), limit=2)
        assert first_page.next_cursor is not None

        await _materialize(state, second_run, ("A", "B", "C"))
        await _advance(
            state,
            ConversationCursor(first_run),
            ConversationCursor(second_run),
        )
        with pytest.raises(AIError) as error:
            await service.history(
                "session",
                principal=Principal("owner", "tenant"),
                cursor=first_page.next_cursor,
                limit=2,
            )
        assert error.value.code is ErrorCode.CURSOR_INVALID
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_uses_projection_v1_mapping_and_empty_strings() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-history-projection", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        run_id = "session-history-projection-run"
        conversation_id = step_conversation_id(
            namespace="session-history-projection",
            tenant_id="tenant",
            execution_id=run_id,
        )
        await _materialize(
            state,
            run_id,
            (),
            messages=[
                ModelRequest(
                    parts=[
                        SystemPromptPart(content="system"),
                        UserPromptPart(content=""),
                        UserPromptPart(
                            content=["", TextContent(content="inline")],
                        ),
                        ToolReturnPart(
                            tool_name="lookup",
                            tool_call_id="return-1",
                            content={"ok": True},
                        ),
                        RetryPromptPart(content="retry"),
                    ],
                    conversation_id=conversation_id,
                ),
                ModelResponse(
                    parts=[
                        TextPart(content=""),
                        TextPart(content="answer"),
                        ToolCallPart(
                            tool_name="lookup",
                            tool_call_id="call-1",
                            args={"query": "value"},
                        ),
                    ],
                    conversation_id=conversation_id,
                ),
            ],
        )
        await _advance(state, None, ConversationCursor(run_id))

        page = await _service(state).history(
            "session",
            principal=Principal("owner", "tenant"),
        )

        assert [item.item_kind for item in page.items] == [
            "system",
            "user",
            "user",
            "tool_result",
            "retry",
            "assistant",
            "assistant",
            "tool_call",
        ]
        assert [item.content for item in page.items] == [
            "system",
            "",
            ["", "inline"],
            {"ok": True},
            "retry",
            "",
            "answer",
            {"query": "value"},
        ]
        assert page.items[3].tool_name == "lookup"
        assert page.items[3].tool_call_id == "return-1"
        assert page.items[7].tool_name == "lookup"
        assert page.items[7].tool_call_id == "call-1"
        assert [item.sequence for item in page.items] == list(range(1, 9))
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_fork_copies_continuation_without_execution_lookup() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-history-fork", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        run_id = "session-history-fork-run"
        await _materialize(state, run_id, ("A",))
        await _advance(state, None, ConversationCursor(run_id))
        service = _service(state)
        principal = Principal("owner", "tenant")
        await service.fork(
            "a" * 64,
            "session",
            ForkSessionRequest(principal, "fork", "fork-operation"),
        )

        source = await service.history("session", principal=principal)
        fork = await service.history("fork", principal=principal)
        assert fork.items == source.items
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_reports_missing_committed_snapshot() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-history-missing", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _advance(state, None, ConversationCursor("missing-run"))
        with pytest.raises(AIError) as error:
            await _service(state).history(
                "session",
                principal=Principal("owner", "tenant"),
            )
        assert error.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_durable_session_history_survives_runtime_state_reopen(tmp_path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="session-history-durable", tenant_id="tenant")
    run_id = "session-history-durable-run"
    await state.conversation.sessions.create(_session())
    await _materialize(state, run_id, ("A", "B"))
    await _advance(state, None, ConversationCursor(run_id))
    before_close = await _service(state).history(
        "session",
        principal=Principal("owner", "tenant"),
    )
    await state.close()

    reopened = RuntimeState.filesystem(tmp_path / "runtime")
    await reopened.initialize(namespace="session-history-durable", tenant_id="tenant")
    try:
        after_reopen = await _service(reopened).history(
            "session",
            principal=Principal("owner", "tenant"),
        )
        assert after_reopen == before_close
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_sql_session_history_survives_runtime_state_reopen(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="session-history-sql", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        run_id = "session-history-sql-run"
        await _materialize(state, run_id, ("SQL",))
        await _advance(state, None, ConversationCursor(run_id))
        before_close = await _service(state).history(
            "session",
            principal=Principal("owner", "tenant"),
        )
        await state.close()

        reopened = RuntimeState.sql(engine)
        await reopened.initialize(namespace="session-history-sql", tenant_id="tenant")
        try:
            after_reopen = await _service(reopened).history(
                "session",
                principal=Principal("owner", "tenant"),
            )
            assert after_reopen == before_close
        finally:
            await reopened.close()
    finally:
        if state.ready:
            await state.close()
        await engine.dispose()
