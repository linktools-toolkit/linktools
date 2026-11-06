#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session lifecycle cooperates through typed readers and retained consumers."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import AuthorizationAction, HmacCursorSigner, Principal, ResourceKind, ResourceRef, SessionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import Persistence
from linktools.ai.session import CloseSessionRequest, ForkSessionRequest, ListSessionRequest, SessionManager, SessionView, UpdateSessionRequest

from .test_session_timeline import _session

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("owner", "tenant")


def _manager(state, **kwargs):
    return SessionManager(
        state.conversation, state.execution.executions,
        TenantAuthorizationPolicy(), object(),
        HmacCursorSigner("session-owner", b"session-owner-key"), **kwargs,
    )


async def test_session_manager_injected_read_and_observation_keep_control_separate() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-owner", tenant_id="tenant")
    reads, observations = [], []

    async def read_session(session_id, *, tenant_id):
        reads.append((session_id, tenant_id))
        return await state.conversation.sessions.get(session_id, tenant_id=tenant_id)

    async def observe_view(record, principal, *, active):
        observations.append(record.session_id)
        view = await manager.project_view(record, principal, active=active)
        return replace(view, active_execution_id="observed")

    manager = _manager(state, read_session=read_session, observe_view=observe_view)
    try:
        await state.conversation.sessions.create(_session())
        assert (await manager.get("session", principal=_PRINCIPAL)).active_execution_id == "observed"
        assert (await manager.list(ListSessionRequest(_PRINCIPAL))).items[0].active_execution_id == "observed"
        assert observations == ["session", "session"] and reads == [("session", "tenant")]
        closed = await manager.close("session", CloseSessionRequest(_PRINCIPAL, "close"))
        assert closed.active_execution_id is None
        assert observations == ["session", "session"]
        assert reads == [("session", "tenant")] * 2
        assert await manager.close("session", CloseSessionRequest(_PRINCIPAL, "close")) == closed
        assert reads == [("session", "tenant")] * 4
    finally:
        await state.close()


async def test_session_consumer_retains_handoff_until_cancelled_display_finishes() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-owner", tenant_id="tenant")
    releases = []
    entered = asyncio.Event()

    async def release_terminal(session_id, *, tenant_id, continuation):
        releases.append((session_id, tenant_id, continuation))

    manager = _manager(state, release_terminal=release_terminal)

    async def display():
        async with manager.consume_session("session", "tenant", display_read=True):
            entered.set()
            await asyncio.Future()

    display_task = drain_task = None
    try:
        await state.conversation.sessions.create(_session())
        display_task = asyncio.create_task(display())
        await asyncio.wait_for(entered.wait(), 5)
        await manager.close("session", CloseSessionRequest(_PRINCIPAL, "close"))
        assert releases == []
        drain_task = asyncio.create_task(manager.drain_display_reads())
        await asyncio.sleep(0)
        assert not drain_task.done()
        display_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await display_task
        await asyncio.wait_for(drain_task, 5)
        assert releases == [("session", "tenant", None)]
        with pytest.raises(AIError) as closed:
            async with manager.consume_session("session", "tenant", display_read=True):
                pytest.fail("display entered after drain")
        assert closed.value.code is ErrorCode.STORAGE_CLOSED
    finally:
        for task in (display_task, drain_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await state.close()


async def test_session_projection_checks_captured_authority_before_admission() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-owner", tenant_id="tenant")
    try:
        manager = _manager(state)
        record = replace(_session(), active_execution_id="missing")
        header = ResourceRef(ResourceKind.SESSION, record.session_id, "tenant", "other")
        with pytest.raises(AIError) as denied:
            await manager.project_view(record, _PRINCIPAL, authorized_header=header)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        view = await manager.project_view(_session(), _PRINCIPAL, active=None)
        assert isinstance(view, SessionView) and view.active_execution_id is None
    finally:
        await state.close()


async def test_session_authorization_uses_one_record_and_rechecks_each_operation(monkeypatch) -> None:
    async with Persistence.open("session-owner", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        record = await state.conversation.sessions.create(_session())
        reads = []
        decisions = []
        allowed = True
        original = state.conversation.sessions.get

        async def get(session_id, *, tenant_id):
            reads.append((session_id, tenant_id))
            return await original(session_id, tenant_id=tenant_id)

        async def no_header(*args, **kwargs):
            pytest.fail("the record already contains its authorization owner")

        class Policy(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                decisions.append(resource.owner_principal_id)
                await super().authorize(principal, action, resource)
                if not allowed:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        monkeypatch.setattr(state.conversation.sessions, "get", get)
        monkeypatch.setattr(state.conversation.sessions, "get_header", no_header)
        manager = SessionManager(state.conversation, state.execution.executions, Policy(),
            object(), HmacCursorSigner("session-owner", b"session-owner-key"))
        assert await manager.read_authorized("session", _PRINCIPAL, AuthorizationAction.SESSION_READ) == record
        allowed = False
        with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
            await manager.read_authorized("session", _PRINCIPAL, AuthorizationAction.SESSION_READ)
        assert reads == [("session", "tenant")] * 2
        assert decisions == ["owner"] * 2


@pytest.mark.parametrize("operation", ("close", "force-close", "update", "fork"))
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_session_mutation_guards_owner_changed_during_policy(tmp_path, operation, backend) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("session-owner", storage=storage, tenant_id="tenant") as state:
        original = await state.conversation.sessions.create(_session())
        changed = False

        class Policy(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                nonlocal changed
                await super().authorize(principal, action, resource)
                if changed:
                    return
                changed = True

                async def remove(transaction):
                    records = await transaction.scan_records()
                    await transaction.delete_records(tuple(record.key_digest for record in records
                        if record.kind in {"session", "conversation_history", "transcript_head"}))

                await state.conversation.sessions.state_store.mutate(remove)
                await state.conversation.sessions.create(replace(original, owner_principal_id="other"))

        manager = SessionManager(state.conversation, state.execution.executions, Policy(),
            object(), HmacCursorSigner("session-owner", b"session-owner-key"))
        with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
            if operation in {"close", "force-close"}:
                await manager.close("session", CloseSessionRequest(_PRINCIPAL, "request", force=operation == "force-close"))
            elif operation == "update":
                await manager.update(original.agent_id, "session", UpdateSessionRequest(
                    _PRINCIPAL, original.revision, "request", {"changed": True}))
            else:
                await manager.fork(original.agent_id, "session", ForkSessionRequest(_PRINCIPAL, "fork", "request"))
        current = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert current.owner_principal_id == "other" and current.revision == original.revision
        assert current.status is SessionStatus.OPEN and current.metadata == original.metadata
        assert await state.conversation.sessions.get("fork", tenant_id="tenant") is None


@pytest.mark.parametrize("resource", ("session", "execution"))
async def test_management_authorization_uses_one_sql_select(tmp_path, resource) -> None:
    from sqlalchemy import event
    from linktools.ai.core import PersistenceDomain
    from linktools.ai.execution import ExecutionManager
    from linktools.ai.persistence._execution_management import StoredExecutionManagementAdapter
    from .test_runtime_persistence_evidence import _execution

    async with Persistence.open("single-authorization-read", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        await state.conversation.sessions.create(_session())
        await state.execution.executions.create(replace(_execution(), principal_id="owner"))
        sessions = _manager(state)
        executions = ExecutionManager(StoredExecutionManagementAdapter(
            state.execution, state.object_store(PersistenceDomain.EXECUTION)), TenantAuthorizationPolicy())
        engine = state.conversation.sessions.state_store.storage_group.context.engine
        selects = []

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            if resource == "session":
                record = await sessions.read_authorized("session", _PRINCIPAL, AuthorizationAction.SESSION_READ)
                assert record.owner_principal_id == "owner"
            else:
                record = await executions.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ)
                assert record.principal_id == "owner"
            assert len(selects) == 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
