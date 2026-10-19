#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session lifecycle cooperates through typed readers and retained consumers."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import HmacCursorSigner, Principal, ResourceKind, ResourceRef, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.session import CloseSessionRequest, ListSessionRequest, SessionManager, SessionView

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
    state = RuntimeState(RuntimeStorage.in_memory())
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
        assert observations == ["session", "session"] and reads == []
        closed = await manager.close("session", CloseSessionRequest(_PRINCIPAL, "close"))
        assert closed.active_execution_id is None
        assert observations == ["session", "session"]
        assert reads and set(reads) == {("session", "tenant")}
    finally:
        await state.close()


async def test_session_consumer_retains_handoff_until_cancelled_display_finishes() -> None:
    state = RuntimeState(RuntimeStorage.in_memory())
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
    state = RuntimeState(RuntimeStorage.in_memory())
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
