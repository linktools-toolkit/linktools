#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only session paging keeps bounded raw ranges and authorization scopes."""

from dataclasses import replace
from types import SimpleNamespace

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
async def test_runtime_history_session_reader_shared_display_cache_preserves_sql_budget(tmp_path):
    from sqlalchemy import event
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.persistence import Persistence
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.storage import InMemoryCacheBackend
    from .test_session_display_cache import _prepare, _service

    cache = InMemoryCacheBackend(max_bytes=2**20)
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "history.db", cache=cache))
    await _prepare(state, body=True)
    statements = []
    engine = state.conversation.sessions.state_store.storage_group.context.engine.sync_engine

    def observe(_connection, _cursor, statement, *_args):
        statements.append(statement.lstrip().split()[0].upper())

    event.listen(engine, "before_cursor_execute", observe)
    try:
        signer = HmacCursorSigner("display", b"display-test-key")
        reader = History(
            SimpleNamespace(), tenant_id="tenant", sessions=state.conversation.sessions,
            executions=state.execution.executions, conversation=state.conversation,
            authorization=TenantAuthorizationPolicy(), cursor_signer=signer,
            session_transcript_store=state.run_store.conversation_persistence, cache=cache,
        )
        live = _service(state, cache)
        expected = await live.refresh("session", principal=_OWNER, max_age=3)
        statements.clear()
        actual = await reader.session_detail("session", principal=_OWNER, max_age=3)
        assert actual == expected
        assert actual.timeline.items[0].output == {"text": "ok"}
        assert not any(value in {"INSERT", "UPDATE", "DELETE", "CREATE"} for value in statements)
        statements.clear()
        assert await reader.session_detail("session", principal=_OWNER, max_age=3) == expected
        assert statements == []
    finally:
        event.remove(engine, "before_cursor_execute", observe)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("complete", "cancel", "promote"))
@pytest.mark.parametrize("borrowed", [False, True])
async def test_runtime_history_session_reader_pending_overlay_and_canonical_boundary(
    tmp_path, monkeypatch, transition, borrowed,
):
    import asyncio
    from contextlib import AsyncExitStack, nullcontext
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import CommitPolicy, ExecutionStatus
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.execution import CancelExecutionRequest
    from linktools.ai.storage import InMemoryCacheBackend
    from linktools.ai.storage._database import SqlStorageContext
    from .test_pending_session_projection import _runtimes

    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, shared=borrowed) as (runtimes, entered, finish, block), AsyncExitStack() as readers:
        owner, _ = runtimes[0]
        session = await owner.agents.get().create_session("session")
        features = RuntimeFeatures("read-history")

        def no_runtime(*args, **kwargs):
            raise AssertionError("read-only history must not build a Runtime")

        monkeypatch.setattr(Runtime, "open", no_runtime)
        reader_context = (nullcontext(runtimes[1][0].history) if borrowed else History.open(
            persistence=await readers.enter_async_context(Persistence.open(
                "pending-session-display", read_only=True,
                storage=PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache),
            )), authorization=features.authorization,
        ))
        async with reader_context as reader:
            principal = owner.default_principal
            await reader.session_detail("session", principal=principal, max_age=3)
            block()
            execution = await session.start("pending question", idempotency_key="pending")
            await asyncio.wait_for(entered.wait(), 5)
            mutations = []
            original = SqlStorageContext.run_mutation

            async def record_mutation(context, callback, **kwargs):
                mutations.append(kwargs.get("domain"))
                return await original(context, callback, **kwargs)

            monkeypatch.setattr(SqlStorageContext, "run_mutation", record_mutation)
            info = await reader.inspect_execution(execution.execution_id, principal=principal, include_usage=False)
            assert info.status is ExecutionStatus.STARTED and info.usage is None
            assert (await reader.list_sessions(principal=principal)).items[0].active_execution_id == execution.execution_id
            detail = await reader.session_detail("session", principal=principal, max_age=3)
            assert detail.session.active_execution_id == execution.execution_id
            assert [turn.execution_id for turn in detail.timeline.items] == [execution.execution_id]
            assert not detail.timeline.items[0].conversation_committed
            assert (await reader.messages("session", principal=principal)).items == ()
            assert mutations == []
            observe = reader._pending_session_execution
            crossed = False

            async def cross_boundary(session_id, *, tenant_id):
                nonlocal crossed
                pending = await observe(session_id, tenant_id=tenant_id)
                if pending is not None and not crossed:
                    crossed = True
                    if transition == "complete":
                        finish.set()
                        await execution.wait(timeout_seconds=5)
                    elif transition == "cancel":
                        await owner.executions.cancel(execution.execution_id, CancelExecutionRequest(principal, "cancel"))
                    else:
                        await owner.executions._service.acquire_dependency_hold(
                            execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold",
                        )
                return pending

            reader._pending_session_execution = cross_boundary
            listed = await reader.list_sessions(principal=principal)
            assert listed.items[0].active_execution_id == (execution.execution_id if transition == "promote" else None)
            mutations.clear()
            after = await reader.session_detail("session", principal=principal, max_age=3)
            final_info = await reader.inspect_execution(execution.execution_id, principal=principal, include_usage=False)
            assert final_info.status is after.timeline.items[0].status
            assert mutations == []
            assert crossed
            assert [turn.execution_id for turn in after.timeline.items] == [execution.execution_id]
            assert after.session.active_execution_id == (execution.execution_id if transition == "promote" else None)
            assert after.timeline.items[0].conversation_committed is (transition != "promote")
            if transition == "cancel":
                assert after.timeline.items[0].status is ExecutionStatus.CANCELLED
                assert after.timeline.items[0].items == () and after.timeline.items[0].output is None
            if transition == "promote":
                await owner.executions._service.release_dependency_hold(
                    execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold",
                )
            finish.set()
            await execution.wait(timeout_seconds=5)


@pytest.mark.asyncio
async def test_runtime_history_session_reader_fork_raw_prefix_and_cold_readonly_open(tmp_path, monkeypatch):
    from pydantic_ai.messages import UserPromptPart
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.storage import InMemoryCacheBackend
    from linktools.ai.storage._database import SqlStorageContext
    from .test_pending_session_projection import _runtimes

    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, shared=False) as (runtimes, _entered, _finish, _block):
        owner, _ = runtimes[0]
        principal = owner.default_principal
        parent = await owner.agents.get().create_session("parent")
        first = await parent.start("prefix question")
        await first.wait()
        child = await parent.fork("child")
        future = await parent.start("parent suffix")
        await future.wait()
        child_turn = await child.start("child suffix")
        await child_turn.wait()

    async def no_mutation(*args, **kwargs):
        raise AssertionError("read-only open/list/messages/detail/close must not mutate storage")

    monkeypatch.setattr(SqlStorageContext, "run_mutation", no_mutation)
    async with Persistence.open('pending-session-display', storage=PersistenceStorage.sqlite(tmp_path / 'state.db'), tenant_id="default", read_only=True) as history_persistence_327, History.open(persistence=history_persistence_327) as history:
        page = await history.messages("child", principal=principal, limit=1)
        messages = list(page.items)
        while page.next_cursor is not None:
            page = await history.messages("child", principal=principal, cursor=page.next_cursor, limit=1)
            messages.extend(page.items)
        prompts = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart)]
        assert prompts == ["prefix question", "child suffix"]
        detail = await history.session_detail("child", principal=principal, limit=1)
        assert [turn.execution_id for turn in detail.timeline.items] == [child_turn.execution_id]
        older = await history.session_detail("child", principal=principal, cursor=detail.timeline.next_cursor, limit=1)
        assert [turn.execution_id for turn in older.timeline.items] == [first.execution_id]
        assert older.timeline.next_cursor is None
        assert len((await history.list_sessions(principal=principal)).items) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_pending", [True, False])
async def test_runtime_history_session_reader_independent_terminal_commit_is_authoritative(tmp_path, keep_pending):
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import CommitPolicy, ExecutionStatus
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.persistence import Persistence
    from linktools.ai.persistence.coordination import PendingTurnCoordination
    from linktools.ai.storage import InMemoryCacheBackend
    from .test_pending_coordination import _claim
    from .test_session_timeline import _execution, _commit_empty_turn

    path = tmp_path / "independent.db"
    # The writer does not participate in this reader's disposable read cache.
    writer = Persistence(PersistenceStorage.sqlite(path))
    await writer.initialize(namespace="independent", tenant_id="tenant")
    await writer.conversation.sessions.create(_session())
    coordination = InMemoryCacheBackend(max_bytes=2**20)
    observer = PendingTurnCoordination.select(coordination, namespace="independent", tenant_id="tenant")
    pending = _execution("execution", status=ExecutionStatus.STARTED, prompt="question")
    lease = await _claim(observer)
    await observer.publish(pending, lease)
    features = RuntimeFeatures("independent-history")
    try:
        async with Persistence.open('independent', storage=PersistenceStorage.sqlite(path, cache=InMemoryCacheBackend(max_bytes=2 ** 20), coordination=coordination), tenant_id='tenant', read_only=True) as history_persistence_372, History.open(persistence=history_persistence_372, authorization=features.authorization) as reader:
            assert (await reader.inspect_execution(
                pending.execution_id, principal=_OWNER, include_usage=False,
            )).status is ExecutionStatus.STARTED
            for _ in range(3):
                current = await reader.session_detail("session", principal=_OWNER, max_age=30)
                assert current.session.active_execution_id == pending.execution_id
            cancelled = _execution("execution", status=ExecutionStatus.CANCELLED,
                                   prompt="question", error_code="EXECUTION_CANCELLED")
            await writer.execution.executions.create(cancelled)
            await writer.conversation.sessions.admit_execution(
                "session", tenant_id="tenant", execution_id="execution", expected=None,
            )
            await _commit_empty_turn(writer, cancelled)
            await writer.conversation.sessions.release_execution(
                "session", tenant_id="tenant", execution_id="execution",
            )
            if not keep_pending:
                await observer.release(lease.lease)
                assert await observer.read_execution("execution", tenant_id="tenant") is None
                assert (await reader.inspect_execution(
                    "execution", principal=_OWNER, include_usage=False,
                )).status is ExecutionStatus.CANCELLED
            # Simulate a lost acknowledgement: the old live lease/publication
            # remains, even though canonical cancellation has committed.
            assert (await observer.read_session_execution("session", tenant_id="tenant") is not None) is keep_pending
            listed = await reader.list_sessions(principal=_OWNER)
            assert listed.items[0].active_execution_id is None
            final = await reader.session_detail("session", principal=_OWNER, max_age=30)
            assert final.session.active_execution_id is None
            assert len(final.timeline.items) == 1
            assert final.timeline.items[0].status is ExecutionStatus.CANCELLED
            assert final.timeline.items[0].conversation_committed
            await observer.release(lease.lease)
            assert await observer.read_execution("execution", tenant_id="tenant") is None
            assert (await reader.inspect_execution(
                "execution", principal=_OWNER, include_usage=False,
            )).status is ExecutionStatus.CANCELLED
    finally:
        await writer.close()
