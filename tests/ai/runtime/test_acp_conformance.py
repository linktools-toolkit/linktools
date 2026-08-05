#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

import pytest

from linktools.ai.agent.mcp.client import (
    McpCloseFailure,
    McpCloseResult,
    McpResourceState,
    McpSessionResources,
)
from linktools.ai.errors import RuntimeInitializationError, SessionClosedError
from linktools.ai.governance.identity import trusted_local_principal
from linktools.ai.execution.live_events import ExecutionEventHub
from linktools.ai.runtime import LocalDirectoryStorage, RuntimeDependencies, build_runtime
from tests.ai.fakes.model import make_router
from linktools.ai.runtime.interaction import InteractiveRunService
from linktools.ai.runtime.session import (
    RuntimeSessionService,
    SessionOperationCoordinator,
    SessionOperationKind,
    SessionSettings,
    SessionWorkspace,
)


@pytest.mark.asyncio
async def test_load_reopens_closed_session_after_runtime_restart(tmp_path) -> None:
    principal = trusted_local_principal(tenant_id="tenant")
    storage = LocalDirectoryStorage(tmp_path)
    service = RuntimeSessionService(storage.execution)
    workspace = SessionWorkspace(cwd=str(tmp_path))
    await service.create(
        session_id="session",
        workspace=workspace,
        settings=SessionSettings(agent_id="default"),
        principal=principal,
    )
    assert (await service.close("session", principal=principal)).closed

    restarted = RuntimeSessionService(storage.execution)
    async with await restarted.prepare_load(
        session_id="session",
        workspace=workspace,
        settings=SessionSettings(agent_id="default"),
        principal=principal,
    ) as transaction:
        record = await transaction.commit()
    assert record.state.value == "open"
    assert (await restarted.get("session", principal=principal)).record.state.value == "open"
    assert (await restarted.close("session", principal=principal)).closed


@pytest.mark.asyncio
async def test_closed_session_rejects_prompt() -> None:
    from linktools.ai.runtime.session import ActiveRuntimeSession
    from linktools.ai.execution.session import SessionRecord, SessionState
    from datetime import datetime, timezone

    record = SessionRecord(
        id="session",
        user_id=None,
        tenant_id=None,
        next_turn_sequence=1,
        latest_completed_run_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        state=SessionState.CLOSED,
    )
    active = ActiveRuntimeSession(record)
    with pytest.raises(SessionClosedError):
        await SessionOperationCoordinator().reserve(active, SessionOperationKind.PROMPT)


@pytest.mark.asyncio
async def test_mcp_close_failure_is_retryable() -> None:
    class Pool:
        def __init__(self):
            self.calls = 0

        async def close(self):
            self.calls += 1
            if self.calls == 1:
                return McpCloseResult(
                    False, (McpCloseFailure("server", "fingerprint", "close_failed"),)
                )
            return McpCloseResult(True)

    pool = Pool()
    resources = McpSessionResources(
        pool=pool,
        state=McpResourceState.OPEN,
        _toolsets=(object(),),
    )
    first = await resources.close("session")
    assert first and resources.state is McpResourceState.OPEN
    second = await resources.close("session")
    assert not second and resources.state is McpResourceState.CLOSED
    assert pool.calls == 2


@pytest.mark.asyncio
async def test_orphan_registry_is_empty_after_done_callback() -> None:
    class Sessions:
        def set_interaction_canceller(self, callback):
            self.cancel = callback

        def set_interaction_owner(self, owner):
            self.owner = owner

    service = InteractiveRunService(object(), object(), Sessions())
    pending = asyncio.Event()
    task = asyncio.create_task(pending.wait())
    service._register_orphan("session", task)
    assert not service.is_empty("session")
    pending.set()
    await task
    await asyncio.sleep(0)
    assert service.is_empty("session")


@pytest.mark.asyncio
async def test_close_and_shutdown_join_one_session_close(tmp_path) -> None:
    principal = trusted_local_principal(tenant_id="tenant")
    storage = LocalDirectoryStorage(tmp_path)
    service = RuntimeSessionService(storage.execution)
    await service.create(
        session_id="session",
        workspace=SessionWorkspace(cwd=str(tmp_path)),
        settings=SessionSettings(agent_id="default"),
        principal=principal,
    )
    closed, shutdown = await asyncio.gather(
        service.close("session", principal=principal), service.shutdown()
    )
    assert closed.closed
    assert shutdown[0].closed
    record = await storage.execution.get_session("session")
    assert record is not None and record.state.value == "closed"


def test_runtime_builder_rejects_two_event_hubs(tmp_path) -> None:
    with pytest.raises(RuntimeInitializationError, match="multiple_execution_event_hubs"):
        build_runtime(
            storage=LocalDirectoryStorage(tmp_path),
            dependencies=RuntimeDependencies(
                model_resolver=make_router(), live_events=ExecutionEventHub()
            ),
            event_hub=ExecutionEventHub(),
        )
