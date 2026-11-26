#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only session paging keeps bounded raw ranges and authorization scopes."""

from dataclasses import replace
from types import SimpleNamespace

from linktools.ai.persistence import Persistence

import pytest
from pydantic_ai.messages import (
    BinaryContent, ModelRequest, ModelResponse, SystemPromptPart, TextPart,
    ToolCallPart, ToolReturnPart, UserPromptPart,
)

from linktools.ai.core import HmacCursorSigner, Page, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.session import ConversationCursor

from .test_session_timeline import _session


class _Authorization:
    async def authorize(self, principal, action, resource):
        if principal.tenant_id != "tenant":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _Sessions:
    def __init__(self, count=4):
        self.record = replace(_session(), history_id="raw", continuation=ConversationCursor(
            "committed-run", history_id="raw", message_count=count,
        ))
        self.pages = []

    async def get_header(self, session_id, *, tenant_id):
        return ResourceRef(ResourceKind.SESSION, session_id, tenant_id, "owner")

    async def get(self, session_id, *, tenant_id):
        return replace(self.record, session_id=session_id)

    async def list_page(self, **kwargs):
        self.pages.append(kwargs)
        offset = int(kwargs["cursor"] or 0)
        values = tuple(replace(self.record, session_id=f"session-{n}")
                       for n in range(offset, min(offset + kwargs["limit"], 2000)))
        next_offset = offset + len(values)
        return 17, Page(values, str(next_offset) if next_offset < 2000 else None)


class _Transcript:
    def __init__(self):
        self.messages = [
            ModelRequest(parts=[SystemPromptPart("system"), UserPromptPart([
                "question", BinaryContent(data=b"image", media_type="image/png"),
            ])]),
            ModelResponse(parts=[ToolCallPart("lookup", {"key": "value"}, "call")]),
            ModelRequest(parts=[ToolReturnPart("lookup", {"found": True}, "call")]),
            ModelResponse(parts=[TextPart("answer")]),
        ]
        self.ranges = []
        self.count_reads = 0

    async def committed_conversation_message_count(self, **kwargs):
        self.count_reads += 1
        assert kwargs["agent_run_id"] == "committed-run"
        return 4

    async def iter_conversation_message_range(self, **kwargs):
        self.ranges.append(kwargs)
        for value in self.messages[kwargs["start"]:kwargs["end"]]:
            yield value


def _reader(count=4):
    sessions, transcript = _Sessions(count), _Transcript()
    reader = History(
        SimpleNamespace(), tenant_id="tenant", sessions=sessions,
        authorization=_Authorization(), cursor_signer=HmacCursorSigner("history", b"test"),
        session_transcript_store=transcript,
    )
    return reader, sessions, transcript


_OWNER = Principal("owner", "tenant")


@pytest.mark.asyncio
async def test_runtime_history_session_reader_raw_pages_preserve_all_parts_and_fixed_prefix():
    reader, sessions, transcript = _reader()
    first = await reader.messages("session", principal=_OWNER, limit=2)
    assert first.items == tuple(transcript.messages[:2])
    assert isinstance(first.items[0].parts[1].content[1], BinaryContent)
    assert first.next_cursor is not None
    transcript.messages += [ModelResponse(parts=[TextPart("future")])]
    sessions.record = replace(sessions.record, continuation=ConversationCursor(
        "later-run", history_id="raw", message_count=5,
    ))
    second = await reader.messages("session", principal=_OWNER, cursor=first.next_cursor, limit=2)
    assert second.items == tuple(transcript.messages[2:4])
    assert second.next_cursor is None
    assert [(value["start"], value["end"]) for value in transcript.ranges] == [(0, 2), (2, 4)]
    assert transcript.count_reads == 0


@pytest.mark.asyncio
async def test_runtime_history_session_reader_cursors_bind_principal_session_and_kind():
    reader, _, transcript = _reader()
    first = await reader.messages("session", principal=_OWNER, limit=1)
    for session_id, principal in (("other", _OWNER), ("session", Principal("other", "tenant"))):
        with pytest.raises(AIError) as error:
            await reader.messages(session_id, principal=principal, cursor=first.next_cursor)
        assert error.value.code is ErrorCode.CURSOR_INVALID
    with pytest.raises(AIError) as error:
        await reader.list_sessions(principal=_OWNER, cursor=first.next_cursor)
    assert error.value.code is ErrorCode.CURSOR_INVALID
    assert len(transcript.ranges) == 1


@pytest.mark.asyncio
async def test_runtime_history_session_reader_only_counts_committed_boundary_once():
    reader, _, transcript = _reader(None)
    transcript.messages.append(ModelResponse(parts=[TextPart("uncommitted")]))
    first = await reader.messages("session", principal=_OWNER, limit=3)
    second = await reader.messages("session", principal=_OWNER, cursor=first.next_cursor)
    assert len(first.items) + len(second.items) == 4
    assert transcript.count_reads == 1
    assert transcript.ranges[-1]["end"] == 4


@pytest.mark.asyncio
async def test_runtime_history_session_reader_list_is_bounded_metadata_page():
    reader, sessions, transcript = _reader()
    first = await reader.list_sessions(principal=_OWNER, limit=2)
    second = await reader.list_sessions(principal=_OWNER, cursor=first.next_cursor, limit=2)
    assert [item.session_id for item in first.items + second.items] == [f"session-{i}" for i in range(4)]
    assert len(sessions.pages) == 2
    assert sessions.pages[1]["snapshot"] == 17
    assert all(value["limit"] == 2 for value in sessions.pages)
    assert not transcript.ranges and transcript.count_reads == 0
    with pytest.raises(AIError) as error:
        await reader.list_sessions(principal=Principal("other", "tenant"), cursor=first.next_cursor)
    assert error.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
async def test_runtime_history_session_reader_empty_unavailable_and_invalid_limits():
    reader, sessions, transcript = _reader(0)
    assert (await reader.messages("session", principal=_OWNER)).items == ()
    sessions.record = replace(sessions.record, continuation=None)
    assert (await reader.messages("session", principal=_OWNER)).items == ()
    sessions.record = replace(sessions.record, history_quality="unavailable")
    with pytest.raises(AIError) as error:
        await reader.messages("session", principal=_OWNER)
    assert error.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
    for limit in (0, -1, True, 1001):
        with pytest.raises((AIError, ValueError)):
            await reader.list_sessions(principal=_OWNER, limit=limit)
        with pytest.raises((AIError, ValueError)):
            await reader.messages("session", principal=_OWNER, limit=limit)
    assert not transcript.ranges


@pytest.mark.asyncio
async def test_runtime_history_session_reader_live_and_readonly_share_bounded_sql_page(tmp_path):
    from sqlalchemy import event

    from linktools.ai.persistence import PersistenceStorage
    from .test_pending_session_projection import _runtimes, _SessionAuthorization

    async with _runtimes(tmp_path, cache=None, shared=False, native=True) as (runtimes, *_):
        runtime, _ = runtimes[0]
        principal = runtime.default_principal
        await runtime.agents.get().create_session("session", principal=principal)
        turn = await runtime.sessions.start(
            "session", "question", agent="default", principal=principal, idempotency_key="request",
        )
        result = await turn.wait_final(timeout_seconds=5)
        expected = await runtime.sessions.refresh("session", principal=principal, limit=1)
        assert expected["timeline"]["items"] == [result.view]
        async with (
            Persistence.open(
                "pending-session-display", read_only=True, storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
            ) as state,
            History.open(persistence=state, authorization=_SessionAuthorization()) as reader,
        ):
            engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
            statements = []

            def observe(_connection, _cursor, statement, *_args):
                statements.append(statement.lstrip().split()[0].upper())

            event.listen(engine, "before_cursor_execute", observe)
            try:
                for _ in range(2):
                    statements.clear()
                    assert await reader.turn_session("session", principal=principal, limit=1) == expected
                    assert 0 < statements.count("SELECT") + statements.count("WITH") <= 5
                    assert set(statements) <= {"SELECT", "WITH", "PRAGMA", "BEGIN"}
            finally:
                event.remove(engine, "before_cursor_execute", observe)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("complete", "cancel"))
@pytest.mark.parametrize("borrowed", [False, True])
async def test_runtime_history_session_reader_observes_native_occupancy_and_terminal(
    tmp_path, monkeypatch, transition, borrowed,
):
    import asyncio
    from contextlib import AsyncExitStack, nullcontext
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.storage._database import SqlStorageContext
    from .test_pending_session_projection import _runtimes, _SessionAuthorization

    async with (
        _runtimes(tmp_path, cache=None, shared=borrowed, native=True) as (runtimes, entered, finish, block),
        AsyncExitStack() as readers,
    ):
        owner, _ = runtimes[0]
        principal = owner.default_principal
        await owner.agents.get().create_session("session", principal=principal)

        def no_runtime(*args, **kwargs):
            pytest.fail("read-only history must not build a Runtime")

        monkeypatch.setattr(Runtime, "open", no_runtime)
        reader_context = nullcontext(runtimes[1][0].history) if borrowed else History.open(
            persistence=await readers.enter_async_context(Persistence.open(
                "pending-session-display", read_only=True, storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
            )), authorization=_SessionAuthorization(),
        )
        async with reader_context as reader:
            assert (await reader.turn_session("session", principal=principal))["timeline"]["items"] == []
            block()
            turn = await owner.sessions.start(
                "session", "pending question", agent="default", principal=principal, idempotency_key="pending",
            )
            await asyncio.wait_for(entered.wait(), 5)

            async def no_mutation(*args, **kwargs):
                pytest.fail("administrative reads must not enter the write boundary")

            async def capture():
                with monkeypatch.context() as patch:
                    patch.setattr(SqlStorageContext, "run_mutation", no_mutation)
                    listed = await reader.turn_sessions(principal=principal, session_ids=("session",))
                    state = await reader.turn_session_state("session", principal=principal)
                    page = await reader.turn_session("session", principal=principal)
                    exact = await reader.turn_result("session", turn.key.turn_id, principal=principal)
                    assert listed["items"] == [state] and page["session"] == state
                    return page, exact

            try:
                page, exact = await capture()
                assert page["session"]["active_turn_id"] == turn.key.turn_id
                assert page["timeline"]["items"] == []
                assert not exact["conversation_committed"] and exact["output"] is None
                if transition == "cancel":
                    assert await turn.cancel()
                else:
                    finish.set()
                terminal = await turn.wait_final(timeout_seconds=5)
                page, exact = await capture()
                assert page["session"]["active_turn_id"] is None
                assert page["timeline"]["items"] == [terminal.view] and exact == terminal.view
                assert exact["conversation_committed"]
                if transition == "cancel":
                    assert terminal.status is ExecutionStatus.CANCELLED
                    assert exact["items"] == [] and exact["output"] is None
                else:
                    assert terminal.status is ExecutionStatus.SUCCEEDED
                    assert exact["output"] == {"text": "answer"}
            finally:
                finish.set()


@pytest.mark.asyncio
async def test_native_history_reader_fork_prefix_and_cold_readonly_open(tmp_path, monkeypatch):
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage._database import SqlStorageContext
    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    class HostAuthorization(TenantAuthorizationPolicy):
        """Host session access for the live runtime and the offline reader."""

    async def model(messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    storage = PersistenceStorage.from_root(tmp_path / "fork-history")
    authorization = HostAuthorization()
    async with Runtime.open(
        "fork-history", storage=storage, authorization=authorization,
        features=model_features(_Models(_UsageFunctionModel(model))),
    ) as runtime:
        agent = runtime.agents.get()
        principal = runtime.default_principal
        parent = await agent.create_session("parent")
        first = await agent.start("prefix question", session_id="parent", idempotency_key="first")
        await first.wait_final(timeout_seconds=5)
        parent_state = await parent.inspect()
        child = await parent.fork("child", expected_revision=parent_state["revision"], idempotency_key="fork")
        assert child is not None
        future = await agent.start("parent suffix", session_id="parent", idempotency_key="future")
        await future.wait_final(timeout_seconds=5)
        child_turn = await agent.start("child suffix", session_id="child", idempotency_key="child")
        await child_turn.wait_final(timeout_seconds=5)

    async def no_mutation(*args, **kwargs):
        raise AssertionError("opening, paging, and closing read-only history must not mutate storage")

    monkeypatch.setattr(SqlStorageContext, "run_mutation", no_mutation)
    async with (
        Persistence.open("fork-history", storage=storage, tenant_id=principal.tenant_id, read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        recent = (await history.turn_session("child", principal=principal, limit=1))["timeline"]
        assert [(turn["turn_id"], turn["user_input"]["prompt"]) for turn in recent["items"]] == [
            (child_turn.key.turn_id, "child suffix"),
        ]
        assert recent["next_cursor"] is not None
        older = (await history.turn_session("child", principal=principal,
                                            cursor=recent["next_cursor"], limit=1))["timeline"]
        assert [(turn["turn_id"], turn["user_input"]["prompt"]) for turn in older["items"]] == [
            (first.key.turn_id, "prefix question"),
        ]
        assert older["next_cursor"] is None
        listed = await history.turn_sessions(principal=principal, session_ids=("parent", "child"))
        assert {session["session_id"] for session in listed["items"]} == {"parent", "child"}


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_pending", [True, False])
async def test_runtime_history_session_reader_independent_terminal_commit_is_authoritative(
    tmp_path, keep_pending,
):
    import asyncio
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.persistence.coordination import PendingTurnCoordination
    from linktools.ai.storage import InMemoryCacheBackend
    from .test_pending_coordination import _claim
    from .test_pending_session_projection import _runtimes, _SessionAuthorization
    from .test_session_timeline import _execution

    # This writer does not use the reader's cache; leave stale pending data there
    # after a native cancellation, as if a separate publisher lost its acknowledgement.
    coordination = InMemoryCacheBackend(max_bytes=2**20)
    observer = PendingTurnCoordination.select(
        coordination, namespace="pending-session-display", tenant_id="default",
    )
    async with _runtimes(tmp_path, cache=None, shared=False, native=True) as (runtimes, entered, finish, block):
        runtime, _ = runtimes[0]
        principal = runtime.default_principal
        await runtime.agents.get().create_session("session", principal=principal)
        block()
        turn = await runtime.sessions.start(
            "session", "question", agent="default", principal=principal, idempotency_key="request",
        )
        await asyncio.wait_for(entered.wait(), 5)
        stale = _execution("execution", status=ExecutionStatus.STARTED, prompt="question")
        lease = await _claim(observer)
        try:
            await observer.publish(stale, lease)
            async with (
                Persistence.open(
                    "pending-session-display", read_only=True,
                    storage=PersistenceStorage.sqlite(tmp_path / "state.db", coordination=coordination),
                ) as state,
                History.open(persistence=state, authorization=_SessionAuthorization()) as reader,
            ):
                before = await reader.turn_session("session", principal=principal)
                assert before["session"]["active_turn_id"] == turn.key.turn_id
                assert before["timeline"]["items"] == []
                assert await turn.cancel()
                terminal = await turn.wait_final(timeout_seconds=5)
                assert terminal.status is ExecutionStatus.CANCELLED
                if not keep_pending:
                    assert await coordination.release_turn(lease.lease)
                assert (await observer.read_session_execution("session", tenant_id="default") is not None) is keep_pending
                listed = await reader.turn_sessions(principal=principal, session_ids=("session",))
                after = await reader.turn_session("session", principal=principal)
                assert listed["items"] == [after["session"]]
                assert after["session"]["active_turn_id"] is None
                assert after["timeline"]["items"] == [terminal.view]
                assert terminal.view["conversation_committed"]
                assert await reader.turn_result("session", turn.key.turn_id, principal=principal) == terminal.view
        finally:
            await coordination.release_turn(lease.lease)
            finish.set()
