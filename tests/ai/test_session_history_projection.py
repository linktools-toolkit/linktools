#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Committed Session history projection and continuation behavior."""

import asyncio

from linktools.ai.persistence import Persistence
from datetime import datetime, timezone

import pytest
from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    canonical_sha256,
    Principal,
    SessionStatus,
    TenantAuthorizationPolicy,
    agent_conversation_id as make_agent_conversation_id,
    agent_run_id as make_agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.session import CreateSessionRequest, ForkSessionRequest
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.history._projection import StepSessionHistoryReader
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.persistence.runs import SESSION_HISTORY_VIEW_V1
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.contracts import ConversationCursor, SessionRecord
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from linktools.ai.persistence._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from sqlalchemy.ext.asyncio import create_async_engine

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


def _session(session_id: str = "session") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id=session_id,
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


def _reader(state: Persistence) -> StepSessionHistoryReader:
    return StepSessionHistoryReader(
        store=state.run_store.read_store(PersistenceDomain.CONVERSATION),
        cursor_signer=HmacCursorSigner("session-history", b"session-history-key"),
        sessions=state.conversation.sessions,
    )


async def _advance(
    state: Persistence,
    expected: ConversationCursor | None,
    next_cursor: ConversationCursor,
) -> None:
    execution_id = "history-execution"
    session = await state.conversation.sessions.get("session", tenant_id="tenant")
    assert session is not None and session.history_id is not None
    effective_expected = expected
    if expected is not None and session.continuation is not None:
        assert session.continuation.agent_run_id == expected.agent_run_id
        effective_expected = session.continuation
    message_count = await state.run_store.conversation_persistence.conversation_message_count(
        history_id=session.history_id,
        agent_run_id=next_cursor.agent_run_id,
        tenant_id="tenant",
    )
    effective_next = ConversationCursor(
        next_cursor.agent_run_id,
        history_id=session.history_id,
        message_count=message_count,
    )
    await state.conversation.sessions.admit_execution(
        "session",
        tenant_id="tenant",
        execution_id=execution_id,
        expected=effective_expected,
    )
    try:
        await state.conversation.sessions.advance_continuation(
            "session",
            tenant_id="tenant",
            execution_id=execution_id,
            expected=effective_expected,
            next_cursor=effective_next,
        )
    finally:
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id=execution_id,
        )


def _service(state: Persistence) -> RuntimeSessionService:
    return RuntimeSessionService(
        state.conversation,
        state.execution.executions,
        TenantAuthorizationPolicy(),
        object(),
        HmacCursorSigner("session", b"session-key"),
        history_reader=_reader(state),
    )


async def _materialize(
    state: Persistence,
    agent_run_id: str,
    prompts: tuple[str, ...],
    messages: list[object] | None = None,
) -> None:
    agent_conversation_id = make_agent_conversation_id(
        namespace="session-history",
        tenant_id="tenant",
        execution_id=agent_run_id,
    )
    now = datetime.now(timezone.utc)
    session = await state.conversation.sessions.get("session", tenant_id="tenant")
    assert session is not None and session.history_id is not None
    await state.run_store.register_agent_run(
        AgentRunRecord(
            agent_run_id=agent_run_id,
            agent_conversation_id=agent_conversation_id,
            parent_agent_run_id=None,
            agent_id="default",
            metadata={
                "agent_id": "default",
                "history_id": session.history_id,
            },
            started_at=now,
        )
    )
    checkpoint_messages = messages or []
    if not checkpoint_messages:
        for prompt in prompts:
            checkpoint_messages.extend(
                (
                    ModelRequest(
                        parts=[UserPromptPart(content=prompt)],
                        conversation_id=agent_conversation_id,
                    ),
                    ModelResponse(
                        parts=[TextPart(content=f"answer:{prompt}")],
                        conversation_id=agent_conversation_id,
                    ),
                )
            )
    await state.run_store.save_checkpoint(
        AgentRunCheckpoint(
            agent_run_id=agent_run_id,
            step_index=len(checkpoint_messages),
            messages=checkpoint_messages,
            agent_conversation_id=agent_conversation_id,
            parent_agent_run_id=None,
            agent_id="default",
            timestamp=now,
            state="complete",
            transcript_message_count_before=0,
        )
    )
    await state.run_store.materialize_conversation(agent_run_id=agent_run_id)


@pytest.mark.asyncio
async def test_empty_and_committed_session_history_use_continuation_only() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
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

        agent_run_id = make_agent_run_id(
            namespace="session-history",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_seq=1,
        )
        await _materialize(state, agent_run_id, ('  {"question":"你好\\nworld"}  ',))
        await _advance(state, None, ConversationCursor(agent_run_id))

        page = await service.history("session", principal=principal)
        assert [(item.item_kind, item.content) for item in page.items] == [
            ("user", '  {"question":"你好\\nworld"}  '),
            ("assistant", 'answer:  {"question":"你好\\nworld"}  '),
        ]
        assert [item.message_seq for item in page.items] == [1, 2]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_cursor_binds_to_current_continuation() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-history-cursor", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        first_run = "session-history-first"
        second_run = "session-history-second"
        await _materialize(state, first_run, ("A", "B"))
        await _advance(state, None, ConversationCursor(first_run))
        service = _service(state)
        first_page = await service.history(
            "session", principal=Principal("owner", "tenant"), limit=2
        )
        assert first_page.next_cursor is not None
        assert SESSION_HISTORY_VIEW_V1 == 1
        payload = HmacCursorSigner("session-history", b"session-history-key").decode(first_page.next_cursor)
        assert payload.filter_digest == canonical_sha256({
            "session_id": "session", "projection_version": 1,
        })
        continuation = await service.history(
            "session", principal=Principal("owner", "tenant"),
            cursor=first_page.next_cursor, limit=2,
        )
        assert [item.content for item in continuation.items] == ["B", "answer:B"]

        await _materialize(state, second_run, ("C",))
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
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-history-projection", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        agent_run_id = "session-history-projection-run"
        agent_conversation_id = make_agent_conversation_id(
            namespace="session-history-projection",
            tenant_id="tenant",
            execution_id=agent_run_id,
        )
        retry = RetryFeedbackPart(content="retry", retry=RetryInfo(cause="model_retry"))
        await _materialize(
            state,
            agent_run_id,
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
                        retry,
                    ],
                    conversation_id=agent_conversation_id,
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
                    conversation_id=agent_conversation_id,
                ),
            ],
        )
        await _advance(state, None, ConversationCursor(agent_run_id))

        page = await _service(state).history(
            "session",
            principal=Principal("owner", "tenant"),
        )

        assert [item.item_kind for item in page.items] == [
            "system",
            "user",
            "user",
            "tool_result",
            "retry-feedback",
            "assistant",
            "assistant",
            "tool_call",
        ]
        assert [item.content for item in page.items] == [
            "system",
            "",
            ["", "inline"],
            {"ok": True},
            page.items[4].content,
            "",
            "answer",
            {"query": "value"},
        ]
        assert page.items[4].content["content"] == "retry"
        assert page.items[4].content["retry"]["cause"] == "model_retry"
        assert page.items[4].tool_call_id is None
        assert page.items[3].tool_name == "lookup"
        assert page.items[3].tool_call_id == "return-1"
        assert page.items[7].tool_name == "lookup"
        assert page.items[7].tool_call_id == "call-1"
        assert [item.message_seq for item in page.items] == [1, 1, 1, 1, 1, 2, 2, 2]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_fork_excludes_uncommitted_physical_tail() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-history-uncommitted-fork", tenant_id="tenant")
    try:
        source = await state.conversation.sessions.create(_session())
        assert source.history_id is not None
        await _materialize(
            state,
            "session-history-uncommitted-run",
            ("stale",),
        )
        service = _service(state)
        principal = Principal("owner", "tenant")

        await service.fork(
            "agent",
            "session",
            ForkSessionRequest(
                principal,
                "fork",
                "fork-uncommitted-operation",
            ),
        )

        child = await state.conversation.sessions.get("fork", tenant_id="tenant")
        assert child is not None and child.history_id is not None
        child_history = await state.conversation.histories.get(
            child.history_id,
            tenant_id="tenant",
        )
        assert child_history is not None
        assert child_history.inherited_message_count == 0
        page = await service.history("fork", principal=principal)
        assert page.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_fork_copies_continuation_without_execution_lookup() -> (
    None
):
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-history-fork", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        agent_run_id = "session-history-fork-run"
        await _materialize(state, agent_run_id, ("A",))
        await _advance(state, None, ConversationCursor(agent_run_id))
        service = _service(state)
        principal = Principal("owner", "tenant")
        await service.fork(
            "agent",
            "session",
            ForkSessionRequest(principal, "fork", "fork-operation"),
        )

        source = await service.history("session", principal=principal)
        fork = await service.history("fork", principal=principal)
        assert fork.items == source.items
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_session_history_reports_missing_committed_checkpoint() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
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
@pytest.mark.parametrize("local", (True, False), ids=("local", "engine"))
async def test_sql_session_history_survives_runtime_storage_reopen(tmp_path, local) -> None:
    class _HostAuthorization(TenantAuthorizationPolicy):
        """Use the same explicit host policy for production and reopened reads."""

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("SQL")])

    namespace = "session-history-sql"
    principal = Principal("owner", "tenant")
    features = RuntimeFeatures(namespace, defaults={"execution": {"retain_history": True}})
    features.agent("default", allow_tools=(), allow_capabilities=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    try:
        if not local:
            await provision_database(engine)

        def open_storage():
            return (PersistenceStorage.from_root(tmp_path / "local", exclusive_writer=True)
                    if local else PersistenceStorage.sql(engine))

        storage = open_storage()
        async with (
            Persistence.open(namespace, storage=storage, tenant_id=principal.tenant_id) as state,
            Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
        ):
            created = await runtime.sessions.create(CreateSessionRequest(principal, "session", "create"))
            session = await runtime.sessions.get(created.session_id, principal=principal)
            async with asyncio.timeout(5), session.start("question", agent=runtime.agents.get(), idempotency_key="turn") as (turn, frames):
                terminal = [frame async for frame in frames][-1]
                assert terminal["type"] == "terminal" and terminal["state"] == "succeeded"
                result = await turn.result()
            assert result.key == turn.key and result.status is ExecutionStatus.SUCCEEDED
            assert result.view["output"] == {"text": "SQL"}
            before_close = await session.timeline()
            assert len(before_close["items"]) == 1
            assert before_close["items"][0]["turn_id"] == result.key.turn_id

        reopened = open_storage()
        async with (
            Persistence.open(namespace, storage=reopened, tenant_id=principal.tenant_id) as state,
            Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
        ):
            session = await runtime.sessions.get("session", principal=principal)
            assert await session.timeline() == before_close
            reopened_result = await runtime.history.turn_result("session", result.key.turn_id, principal=principal)
            assert reopened_result["conversation_committed"] and reopened_result["output"] == {"text": "SQL"}
    finally:
        await engine.dispose()
