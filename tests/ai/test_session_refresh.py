#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinated session refresh captures fresh authority, admission, and timeline."""

import asyncio
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import (
    ExecutionLineageKind, ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeSessions, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute, SessionDetail
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.state._contracts import ConversationCursor
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_session_timeline import _execution, _materialize_conversation, _session

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("owner", "tenant")


def _service(state: RuntimeStorage, authorization=None) -> DefaultSessionService:
    return DefaultSessionService(
        state.conversation, state.execution.executions,
        authorization or TenantAuthorizationPolicy(), object(),
        HmacCursorSigner("refresh", b"session-refresh-key"),
        history_reader=object(), transcript_store=state.run_store,
    )


async def _admit(state, execution_id="active", status=ExecutionStatus.FAILED):
    await state.execution.executions.create(replace(
        _execution(execution_id, status=status, prompt=execution_id), result=None,
    ))
    return await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id=execution_id, expected=None,
    )


async def _release(state, execution_id):
    return await state.conversation.sessions.release_execution(
        "session", tenant_id="tenant", execution_id=execution_id,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("status", (ExecutionStatus.PENDING_START, ExecutionStatus.FAILED))
async def test_refresh_reconciles_once_outside_metadata_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, status: ExecutionStatus,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, status=status)
        service = _service(state)
        reconcile = service._reconcile_terminal_admission
        calls = []

        async def checked(record):
            assert active_state_scope() is None
            calls.append(record.active_execution_id)
            return await reconcile(record)

        monkeypatch.setattr(service, "_reconcile_terminal_admission", checked)
        detail = await service.refresh("session", principal=_PRINCIPAL)
        assert isinstance(detail, SessionDetail)
        assert calls == ([] if status is ExecutionStatus.PENDING_START else ["active"])
        expected = "active" if status is ExecutionStatus.PENDING_START else None
        assert detail.session.active_execution_id == expected
        assert [turn.execution_id for turn in detail.timeline.items] == ["active"]
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == expected
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_refresh_uses_header_authority_and_fresh_record_after_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    pending = None
    try:
        created = await state.conversation.sessions.create(_session())
        sessions = state.conversation.sessions
        get_header = sessions.get_header
        entered, proceed = asyncio.Event(), asyncio.Event()
        seen = []

        async def custom_header(*args, **kwargs):
            header = await get_header(*args, **kwargs)
            return replace(header, owner_principal_id="header-owner")

        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                assert resource.owner_principal_id == "header-owner"
                seen.append(resource)
                entered.set()
                await proceed.wait()

        monkeypatch.setattr(sessions, "get_header", custom_header)
        service = _service(state, Authorization())
        pending = asyncio.create_task(service.refresh("session", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await sessions.compare_and_swap(
            "session", tenant_id="tenant", expected_revision=created.revision,
            next_record=replace(created, metadata={"updated": True}, revision=1),
        )
        await _admit(state, "after-authorization", ExecutionStatus.PENDING_START)
        proceed.set()
        detail = await pending
        assert len(seen) == 1
        assert detail.session.metadata == {"updated": True}
        assert detail.session.active_execution_id == "after-authorization"
        assert [turn.execution_id for turn in detail.timeline.items] == ["after-authorization"]
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


@pytest.mark.parametrize("principal", (Principal("other", "tenant"), Principal("owner", "other")))
async def test_refresh_denied_authority_does_not_cleanup_admission(principal: Principal) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state)
        with pytest.raises(AIError) as error:
            await _service(state).refresh("session", principal=principal)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == "active"
    finally:
        await state.close()


@pytest.mark.parametrize("fault,expected", (
    ("limit", ErrorCode.PAGE_LIMIT_INVALID),
    ("cursor", ErrorCode.CURSOR_INVALID),
    ("history", ErrorCode.SESSION_HISTORY_UNAVAILABLE),
    ("dependency", ErrorCode.RUNTIME_DEPENDENCY_NOT_READY),
))
async def test_refresh_cleans_admission_before_page_errors(
    monkeypatch: pytest.MonkeyPatch, fault: str, expected: ErrorCode,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state)
        service = _service(state)
        kwargs = {}
        if fault == "limit":
            kwargs["limit"] = 0
        elif fault == "cursor":
            kwargs["cursor"] = "invalid"
        elif fault == "dependency":
            service._transcript_store = None
        else:
            async def unavailable(*args, **kwargs):
                record = await state.conversation.sessions.get("session", tenant_id="tenant")
                assert record.active_execution_id is None
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            monkeypatch.setattr(state.conversation.sessions, "timeline_head", unavailable)
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, **kwargs)
        assert error.value.code is expected
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id is None
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("status", (ExecutionStatus.PENDING_START, ExecutionStatus.FAILED))
async def test_refresh_captures_concurrent_replacement_without_reconciling_twice(
    monkeypatch: pytest.MonkeyPatch, status: ExecutionStatus,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, "first")
        service = _service(state)
        release = state.conversation.sessions.release_execution
        released = []

        async def replace_during_release(*args, **kwargs):
            assert active_state_scope() is None
            released.append(kwargs["execution_id"])
            await release(*args, **kwargs)
            await _admit(state, "second", status)
            return await release(*args, **kwargs)

        monkeypatch.setattr(state.conversation.sessions, "release_execution", replace_during_release)
        detail = await service.refresh("session", principal=_PRINCIPAL)
        assert released == ["first"]
        assert detail.session.active_execution_id == (
            "second" if status is ExecutionStatus.PENDING_START else None
        )
        assert [turn.execution_id for turn in detail.timeline.items] == ["first", "second"]
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == "second"
    finally:
        await state.close()


@pytest.mark.parametrize("corruption", (None, "other_session", "child", "missing"))
async def test_refresh_batches_and_validates_active_execution_outside_page(
    monkeypatch: pytest.MonkeyPatch, corruption: str | None,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, "older")
        await _release(state, "older")
        await _admit(state, "active", ExecutionStatus.PENDING_START)
        service = _service(state)
        page = await service.timeline("session", principal=_PRINCIPAL, limit=1)
        assert page.next_cursor is not None
        executions = state.execution.executions
        get_many = executions.get_many
        seen = []

        async def checked(ids, **kwargs):
            seen.append(tuple(ids))
            values = dict(await get_many(ids, **kwargs))
            if corruption == "other_session":
                values["active"] = replace(values["active"], session_id="other")
            elif corruption == "child":
                values["active"] = replace(
                    values["active"], parent_execution_id="parent",
                    parent_invocation_id="invocation", lineage_kind=ExecutionLineageKind.SUBAGENT,
                )
            elif corruption == "missing":
                del values["active"]
            return values

        monkeypatch.setattr(executions, "get_many", checked)
        if corruption:
            with pytest.raises(AIError) as error:
                await service.refresh("session", principal=_PRINCIPAL, limit=1, cursor=page.next_cursor)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            detail = await service.refresh("session", principal=_PRINCIPAL, limit=1, cursor=page.next_cursor)
            assert [turn.execution_id for turn in detail.timeline.items] == ["older"]
            assert detail.session.active_execution_id == "active"
        assert seen == [("older", "active")]
    finally:
        await state.close()


async def test_refresh_keeps_split_database_routes_and_domain_facade(tmp_path: Path) -> None:
    state = RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=RuntimeStorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=RuntimeStorageRoute.sqlite(tmp_path / "conversation.db"),
    ))
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, status=ExecutionStatus.PENDING_START)
        checked = []
        facade = RuntimeSessions(_service(state), object(), ensure_open=lambda: checked.append(True))
        detail = await facade.refresh("session", principal=_PRINCIPAL)
        assert detail.session.active_execution_id == "active"
        assert detail.timeline.items[0].execution_id == "active"
        assert checked == [True]
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_refresh_idle_has_one_authority_read_and_one_metadata_snapshot(tmp_path: Path) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "refresh.db")
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state)
        await _release(state, "active")
        engine = state.conversation.sessions.state_store.storage_group.context.engine
        statements, begins = [], []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        def begin(connection):
            begins.append(connection)
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        event.listen(engine.sync_engine, "begin", begin)
        try:
            detail = await _service(state).refresh("session", principal=_PRINCIPAL)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
            event.remove(engine.sync_engine, "begin", begin)
        assert detail.timeline.items[0].execution_id == "active"
        assert len(begins) == 2
        assert sum(statement.lstrip().upper().startswith("SELECT") for statement in statements) == 6
    finally:
        await state.close()


@pytest.mark.parametrize("cancel", (False, True))
async def test_refresh_bodies_leave_metadata_scope_and_hold_consumer(
    monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-timeline", tenant_id="tenant")
    pending = None
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
                transaction, "session", tenant_id="tenant", execution_id="success",
                expected=None, next_cursor=ConversationCursor(run_id, history_id=created.history_id, message_count=count),
                release_execution=True, history_quality="complete",
            )
        await sessions.state_store.mutate(commit)
        service = _service(state)
        read = state.run_store.iter_conversation_message_range
        entered, proceed, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def body(**kwargs):
            assert active_state_scope() is None
            entered.set()
            await proceed.wait()
            async with aclosing(read(**kwargs)) as stream:
                async for message in stream:
                    assert active_state_scope() is None
                    yield message
        async def release(*args, **kwargs):
            assert active_state_scope() is None
            released.set()
        monkeypatch.setattr(state.run_store, "iter_conversation_message_range", body)
        monkeypatch.setattr(service, "_release_terminal", release)
        pending = asyncio.create_task(service.refresh("session", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service._request_session_release("session", "tenant", None)
        assert not released.is_set()
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            proceed.set()
            detail = await pending
            assert detail.timeline.items[0].conversation_committed is True
        assert released.is_set()
        assert active_state_scope() is None
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


@pytest.mark.parametrize("principal,limit,cursor,expected", (
    (Principal("other", "tenant"), 0, None, ErrorCode.PAGE_LIMIT_INVALID),
    (Principal("other", "tenant"), 100, "invalid", ErrorCode.AUTHORIZATION_DENIED),
    (_PRINCIPAL, 100, "invalid", ErrorCode.CURSOR_INVALID),
))
async def test_timeline_retains_validation_authority_order_and_read_only_admission(
    principal: Principal, limit: int, cursor: str | None, expected: ErrorCode,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state)
        with pytest.raises(AIError) as error:
            await _service(state).timeline("session", principal=principal, limit=limit, cursor=cursor)
        assert error.value.code is expected
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == "active"
    finally:
        await state.close()


async def test_runtime_sessions_refresh_delegates_to_the_injected_service() -> None:
    calls = []
    result = object()
    class Service:
        async def refresh(self, session_id, *, principal, cursor, limit):
            calls.append((session_id, principal, cursor, limit))
            return result
    facade = RuntimeSessions(Service(), object(), ensure_open=lambda: calls.append("open"))
    assert await facade.refresh("session", principal=_PRINCIPAL, cursor="cursor", limit=2) is result
    assert calls == ["open", ("session", _PRINCIPAL, "cursor", 2)]


async def test_refresh_cancellation_during_metadata_releases_scope_and_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    pending = None
    try:
        await state.conversation.sessions.create(_session())
        service = _service(state)
        entered, blocked, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def head(*args, **kwargs):
            assert active_state_scope() is not None
            entered.set()
            await blocked.wait()
        async def release(*args, **kwargs):
            assert active_state_scope() is None
            released.set()
        monkeypatch.setattr(state.conversation.sessions, "timeline_head", head)
        monkeypatch.setattr(service, "_release_terminal", release)
        pending = asyncio.create_task(service.refresh("session", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service._request_session_release("session", "tenant", None)
        assert not released.is_set()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert released.is_set()
        assert active_state_scope() is None
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


async def test_refresh_live_admission_reuses_the_metadata_snapshot(tmp_path: Path) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "live-refresh.db")
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, status=ExecutionStatus.PENDING_START)
        engine = state.conversation.sessions.state_store.storage_group.context.engine
        statements, begins = [], []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        def begin(connection):
            begins.append(connection)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        event.listen(engine.sync_engine, "begin", begin)
        try:
            detail = await _service(state).refresh("session", principal=_PRINCIPAL)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
            event.remove(engine.sync_engine, "begin", begin)
        assert detail.session.active_execution_id == "active"
        assert detail.timeline.items[0].execution_id == "active"
        assert len(begins) == 2
        assert sum(sql.lstrip().upper().startswith("SELECT") for sql in statements) == 6
    finally:
        await state.close()


@pytest.mark.parametrize("status", (ExecutionStatus.PENDING_START, ExecutionStatus.FAILED))
@pytest.mark.parametrize("corruption", ("missing", "other_session", "child"))
async def test_refresh_validates_initial_admission_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, status: ExecutionStatus, corruption: str,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await _admit(state, status=status)
        executions = state.execution.executions
        get = executions.get

        async def corrupted(*args, **kwargs):
            value = await get(*args, **kwargs)
            assert value is not None
            if corruption == "missing":
                return None
            if corruption == "other_session":
                return replace(value, session_id="other")
            return replace(
                value, parent_execution_id="parent", parent_invocation_id="invocation",
                lineage_kind=ExecutionLineageKind.SUBAGENT,
            )

        monkeypatch.setattr(executions, "get", corrupted)
        with pytest.raises(AIError) as error:
            await _service(state).refresh("session", principal=_PRINCIPAL)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        session = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert session.active_execution_id == "active"
        assert active_state_scope() is None
    finally:
        await state.close()
