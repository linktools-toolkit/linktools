#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live cached Conversation turns share Session display paging without durable writes."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, RuntimeFeatures
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ConversationCapability
from linktools.ai.runtime import Runtime, RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.service_api import CancelExecutionRequest, ListSessionRequest
from linktools.ai.runtime.state._store import active_state_scope, record_key_digest
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from linktools.ai.storage._database import SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_session_refresh import _service
from .test_session_timeline import _commit_empty_turn, _execution, _session

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _runtimes(tmp_path, *, cache, shared=True):
    entered, finish = asyncio.Event(), asyncio.Event()
    pause = False

    async def model(messages, info):
        if pause:
            entered.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    def group():
        value = RuntimeFeatures("pending-session-display")
        value.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=cache), scope="runtime")
        value.agent("default", model="default", allow_tools=())
        value.capability(ConversationCapability(), scope="runtime")
        return value

    def block_next():
        nonlocal pause
        pause = True

    models = _Models(_UsageFunctionModel(model))
    async with AsyncExitStack() as stack:
        values = []
        for _ in range(2 if shared and cache is not None else 1):
            storage = RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache)
            runtime = await stack.enter_async_context(Runtime.open(
                "pending-session-display", storage=storage, models=models, features=group(),
            ))
            values.append((runtime, runtime._state))
        try:
            yield values, entered, finish, block_next
        finally:
            finish.set()


@pytest.mark.parametrize("history_count,max_age,shared", (
    (0, 0, False), (0, 60, True), (11, 0, True), (11, 60, True),
))
async def test_pending_session_projection_paging_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    history_count: int, max_age: float, shared: bool,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, shared=shared) as (runtimes, entered, finish, block):
        owner, storage = runtimes[0]
        session = await owner.agents.get().create_session("session")
        completed = []
        for index in range(history_count):
            execution = await session.start(f"question {index}", idempotency_key=str(index))
            assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
            completed.append(execution.execution_id)
        # A warm idle page must not hide a newly admitted, source-absent turn.
        for reader, _ in runtimes:
            await reader.sessions.refresh("session", principal=reader.default_principal, limit=10, max_age=max_age)
        cursors = []
        mutations = []
        original_mutation = SqlStorageContext.run_mutation

        async def count_mutation(context, callback, **kwargs):
            result = await original_mutation(context, callback, **kwargs)
            mutations.append(kwargs.get("domain"))
            return result

        monkeypatch.setattr(SqlStorageContext, "run_mutation", count_mutation)
        block()
        execution = await session.start("pending question", idempotency_key="pending")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert await storage.execution.executions.get(execution.execution_id, tenant_id=owner.tenant_id) is None
            for reader, _ in runtimes:
                principal = reader.default_principal
                service = reader.sessions._service
                assert (await service.get("session", principal=principal)).active_execution_id == execution.execution_id
                assert (await reader.sessions.list(ListSessionRequest(principal))).items[0].active_execution_id == execution.execution_id
                detail = await reader.sessions.refresh("session", principal=principal, limit=10, max_age=max_age)
                assert detail.session.active_execution_id == execution.execution_id
                expected = completed[-9:] + [execution.execution_id]
                assert [turn.execution_id for turn in detail.timeline.items] == expected
                pending = detail.timeline.items[-1]
                assert pending.status is ExecutionStatus.STARTED and not pending.conversation_committed
                assert pending.items == () and pending.user_input["prompt"]["text"] == "pending question"
                page = await reader.sessions.timeline("session", principal=principal, limit=10)
                assert page.items == detail.timeline.items
                assert bool(page.next_cursor) == bool(detail.timeline.next_cursor)
                if history_count:
                    assert page.next_cursor is not None
                    cursors.append((reader, page.next_cursor))
                    older = await reader.sessions.refresh("session", principal=principal, cursor=page.next_cursor, limit=10, max_age=max_age)
                    assert [turn.execution_id for turn in older.timeline.items] == completed[:2]
                    assert older.timeline.next_cursor is None
                    assert older.session.active_execution_id == execution.execution_id
                else:
                    assert page.next_cursor is None
                latest = await reader.sessions.timeline("session", principal=principal, limit=1)
                assert [turn.execution_id for turn in latest.items] == [execution.execution_id]
                seen = []
                cursor = latest.next_cursor
                while cursor is not None:
                    older = await reader.sessions.timeline("session", principal=principal, cursor=cursor, limit=1)
                    seen.extend(turn.execution_id for turn in older.items)
                    cursor = older.next_cursor
                assert seen == list(reversed(completed))
                refreshed = await reader.sessions.refresh("session", principal=principal, limit=10, max_age=max_age)
                assert refreshed.timeline.items == detail.timeline.items
            assert mutations == []
        finally:
            finish.set()
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert len(mutations) == 1
        for reader, _ in runtimes:
            detail = await reader.sessions.refresh("session", principal=reader.default_principal, limit=10, max_age=max_age)
            assert detail.session.active_execution_id is None
            assert [turn.execution_id for turn in detail.timeline.items] == (completed + [execution.execution_id])[-10:]
            assert detail.timeline.items[-1].conversation_committed
            assert detail.timeline.items[-1].status is ExecutionStatus.SUCCEEDED
            assert any(item.content == "answer" for item in detail.timeline.items[-1].items)
        for reader, cursor in cursors:
            older = await reader.sessions.timeline("session", principal=reader.default_principal, cursor=cursor, limit=10)
            assert [turn.execution_id for turn in older.items] == completed[:2]
            assert older.next_cursor is None
        assert len(mutations) == 1


@pytest.mark.parametrize("transition", ("complete", "cancel", "promote"))
@pytest.mark.parametrize("remote", (False, True))
async def test_pending_session_projection_source_wins_observation_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str, remote: bool,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, reader_storage = runtimes[int(remote)]
        session = await owner.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        service = reader.sessions._service
        observe = service._pending_session_execution
        crossed = False

        async def cross_boundary(session_id, *, tenant_id):
            nonlocal crossed
            assert active_state_scope() is None
            pending = await observe(session_id, tenant_id=tenant_id)
            if pending is not None and not crossed:
                crossed = True
                if transition == "complete":
                    finish.set()
                    await execution.wait(timeout_seconds=5)
                elif transition == "cancel":
                    await owner.executions.cancel(execution.execution_id, CancelExecutionRequest(owner.default_principal, "cancel"))
                else:
                    assert await owner.executions._service.acquire_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold")
            return pending

        monkeypatch.setattr(service, "_pending_session_execution", cross_boundary)
        detail = await reader.sessions.refresh("session", principal=reader.default_principal, limit=1, max_age=60)
        assert crossed
        assert [turn.execution_id for turn in detail.timeline.items] == [execution.execution_id]
        assert detail.timeline.next_cursor is None
        canonical = await reader_storage.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id)
        assert canonical is not None and detail.timeline.items[0].status is canonical.status
        assert detail.session.active_execution_id == (execution.execution_id if transition == "promote" else None)
        assert detail.timeline.items[0].conversation_committed is (transition in {"complete", "cancel"})
        if transition == "cancel":
            assert detail.timeline.items[0].status is ExecutionStatus.CANCELLED
            assert detail.timeline.items[0].output is None and detail.timeline.items[0].items == ()
        if transition == "promote":
            await owner.executions._service.release_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold")
        finish.set()
        await execution.wait(timeout_seconds=5)


@pytest.mark.parametrize("principal", (Principal("other", "tenant"), Principal("owner", "other")))
@pytest.mark.parametrize("method", ("get", "refresh", "timeline"))
async def test_pending_session_projection_denial_precedes_observation(principal: Principal, method: str) -> None:
    storage = RuntimeStorage.in_memory()
    storage = RuntimeState(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await storage.conversation.sessions.create(_session())
        service = _service(storage)

        async def forbidden(*args, **kwargs):
            raise AssertionError("unauthorized caller reached a live observation")

        service._pending_session_execution = forbidden
        with pytest.raises(AIError) as denied:
            await getattr(service, method)("session", principal=principal)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await storage.close()


@pytest.mark.parametrize("fault", ("owner", "session", "child", "terminal", "older_session"))
async def test_pending_session_projection_rejects_unbound_observations(fault: str) -> None:
    storage = RuntimeStorage.in_memory()
    storage = RuntimeState(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        record = await storage.conversation.sessions.create(_session())
        pending = _execution("pending", status=ExecutionStatus.STARTED, prompt="question")
        changes = {
            "owner": {"principal_id": "other"}, "session": {"session_id": "other"},
            "child": {"parent_execution_id": "parent", "parent_invocation_id": "invocation",
                      "lineage_kind": ExecutionLineageKind.SUBAGENT},
            "terminal": {"status": ExecutionStatus.CANCELLED},
            "older_session": {"created_at": record.created_at - timedelta(seconds=1)},
        }
        pending = replace(pending, **changes[fault])
        service = _service(storage)

        async def observe(session_id, *, tenant_id):
            assert (session_id, tenant_id) == ("session", "tenant")
            return pending

        service._pending_session_execution = observe
        principal = Principal("owner", "tenant")
        detail = await service.refresh("session", principal=principal)
        assert detail.session.active_execution_id is None and detail.timeline.items == ()
        assert (await service.get("session", principal=principal)).active_execution_id is None
        assert (await service.timeline("session", principal=principal)).items == ()
    finally:
        await storage.close()


async def test_pending_session_projection_remote_requires_live_coordination(tmp_path: Path) -> None:
    class HiddenOwnerCache(InMemoryCacheBackend):
        hidden = False

        async def read_turn(self, scope, execution_id):
            return None if self.hidden else await super().read_turn(scope, execution_id)

    cache = HiddenOwnerCache(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, _ = runtimes[1]
        session = await owner.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        assert (await reader.sessions.refresh("session", principal=reader.default_principal)).session.active_execution_id == execution.execution_id
        cache.hidden = True
        try:
            detail = await reader.sessions.refresh("session", principal=reader.default_principal, max_age=60)
            assert detail.session.active_execution_id is None and detail.timeline.items == ()
        finally:
            cache.hidden = False
            finish.set()
        await execution.wait(timeout_seconds=5)


async def test_pending_session_projection_no_cache_keeps_source_admission(tmp_path: Path) -> None:
    async with _runtimes(tmp_path, cache=None) as (runtimes, entered, finish, block):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        source = await storage.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
        assert source.active_execution_id == execution.execution_id
        detail = await runtime.sessions.refresh("session", principal=runtime.default_principal, max_age=60)
        assert detail.session.active_execution_id == execution.execution_id
        assert [turn.execution_id for turn in detail.timeline.items] == [execution.execution_id]
        finish.set()
        await execution.wait(timeout_seconds=5)


@pytest.mark.parametrize("limit", (1, 10))
async def test_pending_session_projection_cannot_fill_missing_source_execution(limit: int) -> None:
    storage = RuntimeStorage.in_memory()
    storage = RuntimeState(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await storage.conversation.sessions.create(_session())
        pending = _execution("pending", status=ExecutionStatus.STARTED, prompt="question")
        sessions = storage.conversation.sessions
        await sessions.admit_execution("session", tenant_id="tenant", execution_id="pending", expected=None)
        await sessions.release_execution("session", tenant_id="tenant", execution_id="pending")
        service = _service(storage)

        async def observe(session_id, *, tenant_id):
            return pending

        service._pending_session_execution = observe
        with pytest.raises(AIError) as unavailable:
            await service.refresh("session", principal=Principal("owner", "tenant"), limit=limit)
        assert unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.parametrize("method", ("timeline", "refresh"))
async def test_pending_session_projection_older_page_cannot_fill_missing_source_execution(method: str) -> None:
    storage = RuntimeStorage.in_memory()
    storage = RuntimeState(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        sessions = storage.conversation.sessions
        await sessions.create(_session())
        pending = _execution("missing", status=ExecutionStatus.STARTED, prompt="cache observation")
        for execution_id in ("missing", "newer"):
            if execution_id == "newer":
                await storage.execution.executions.create(replace(
                    _execution(execution_id, status=ExecutionStatus.FAILED, prompt="canonical"), result=None,
                ))
            await sessions.admit_execution("session", tenant_id="tenant", execution_id=execution_id, expected=None)
            if execution_id == "newer":
                await _commit_empty_turn(
                    storage, await storage.execution.executions.get(execution_id, tenant_id="tenant"),
                )
            await sessions.release_execution("session", tenant_id="tenant", execution_id=execution_id)
        service = _service(storage)
        principal = Principal("owner", "tenant")
        newest = await service.timeline("session", principal=principal, limit=1)
        assert [turn.execution_id for turn in newest.items] == ["newer"]
        assert newest.next_cursor is not None
        read = getattr(service, method)
        with pytest.raises(AIError) as unavailable:
            await read("session", principal=principal, limit=1, cursor=newest.next_cursor)
        assert unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        async def observe(session_id, *, tenant_id):
            return pending

        service._pending_session_execution = observe
        with pytest.raises(AIError) as still_unavailable:
            await read("session", principal=principal, limit=1, cursor=newest.next_cursor)
        assert still_unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.parametrize("observation", ("absent", "rejected"))
@pytest.mark.parametrize("method,max_age,keep_active", (
    ("timeline", 0, False), ("refresh", 0, False), ("refresh", 60, False), ("refresh", 0, True),
))
async def test_pending_session_projection_rechecks_owner_after_empty_observation(
    method: str, max_age: float, keep_active: bool, observation: str,
) -> None:
    storage = RuntimeStorage.in_memory()
    storage = RuntimeState(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        sessions = storage.conversation.sessions
        record = await sessions.create(_session())
        service = _service(storage)
        service._cache = InMemoryCacheBackend(max_bytes=2**20)
        principal = Principal("owner", "tenant")
        if max_age:
            await service.refresh("session", principal=principal, limit=1, max_age=max_age)

        async def observe(session_id, *, tenant_id):
            kinds = (("session", "session"), ("conversation_history", record.history_id),
                     ("transcript_head", record.history_id))
            keys = tuple(record_key_digest(
                sessions.namespace, sessions.tenant_id, RuntimeDomain.CONVERSATION.value, kind, identity,
            ) for kind, identity in kinds)
            await sessions.state_store.mutate(lambda transaction: transaction.delete_records(keys))
            await sessions.create(replace(_session(), owner_principal_id="other", metadata={"secret": "other owner"}))
            await storage.execution.executions.create(replace(
                _execution("other-execution", status=ExecutionStatus.FAILED, prompt="OTHER OWNER INPUT"),
                principal_id="other", result=None,
            ))
            await sessions.admit_execution("session", tenant_id="tenant", execution_id="other-execution", expected=None)
            if not keep_active:
                await sessions.release_execution("session", tenant_id="tenant", execution_id="other-execution")
            if observation == "absent":
                return None
            return replace(_execution("rejected", status=ExecutionStatus.STARTED, prompt="observation"), principal_id="other")

        service._pending_session_execution = observe
        options = {"max_age": max_age} if method == "refresh" else {}
        with pytest.raises(AIError) as denied:
            await getattr(service, method)("session", principal=principal, limit=1, **options)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        current = await sessions.get("session", tenant_id="tenant")
        assert current.active_execution_id == ("other-execution" if keep_active else None)
    finally:
        await storage.close()
