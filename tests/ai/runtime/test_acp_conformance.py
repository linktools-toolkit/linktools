#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

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
from linktools.ai.execution.live_events import (
    AssistantTextDelta,
    ExecutionCompleted,
    ExecutionEvent,
)
from linktools.ai.execution.domain import RunStatus
from linktools.ai.runtime import LocalDirectoryStorage, RuntimeDependencies, build_runtime
from tests.ai.fakes.model import make_router
from linktools.ai.runtime.interaction import InteractiveRunService
from linktools.ai.runtime.session import (
    RuntimeSessionService,
    ActiveRuntimeSession,
    SessionCommitter,
    SessionOperationCoordinator,
    SessionOperationKind,
    SessionSettings,
    SessionState,
    SessionWorkspace,
)
from linktools.ai.execution.session import SessionRecord, UpdateSession
from linktools.ai.execution.live_events import CompositeRunLiveSink
from linktools.ai.acp.client import AcpClient, AcpClientSessionResources


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
    assert first and resources.state is McpResourceState.CLEANUP_REQUIRED
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


@pytest.mark.asyncio
async def test_session_commit_cancellation_waits_for_active_sync() -> None:
    principal_record = SessionRecord(
        id="session",
        user_id=None,
        tenant_id=None,
        next_turn_sequence=1,
        latest_completed_run_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        state=SessionState.OPEN,
    )
    updated = replace(principal_record, revision=principal_record.revision + 1)
    store_updated = asyncio.Event()
    release_store = asyncio.Event()

    class Store:
        async def update_session(self, command):
            store_updated.set()
            await release_store.wait()
            return updated

    active = ActiveRuntimeSession(principal_record)
    lease = await SessionOperationCoordinator().reserve(active, SessionOperationKind.UPDATE)
    committer = SessionCommitter(Store())
    caller = asyncio.create_task(
        committer.commit_update(
            active,
            lease,
            UpdateSession(
                session_id="session",
                expected_revision=principal_record.revision,
            ),
        )
    )
    await store_updated.wait()
    await active.lock.acquire()
    release_store.set()
    caller.cancel()
    active.lock.release()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert active.record == updated
    await lease.release()


@pytest.mark.asyncio
async def test_lease_release_cancellation_clears_operation() -> None:
    record = SessionRecord(
        id="session",
        user_id=None,
        tenant_id=None,
        next_turn_sequence=1,
        latest_completed_run_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        state=SessionState.OPEN,
    )
    active = ActiveRuntimeSession(record)
    lease = await SessionOperationCoordinator().reserve(
        active, SessionOperationKind.PROMPT, execution_id="execution"
    )
    await active.lock.acquire()
    caller = asyncio.create_task(lease.release())
    await asyncio.sleep(0)
    caller.cancel()
    active.lock.release()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert active.operation is None
    assert active.active_execution_id is None
    assert lease.done.is_set()
    assert lease._released


@pytest.mark.asyncio
async def test_extra_sink_isolated_from_canonical_publish() -> None:
    gate = asyncio.Event()

    class HangingSink:
        async def publish(self, event):
            await gate.wait()

    canonical = ExecutionEventHub()
    subscription = await canonical.subscribe("execution")
    composite = CompositeRunLiveSink(canonical, HangingSink())
    await asyncio.wait_for(
        composite.publish_execution(
            "execution", AssistantTextDelta(execution_id="execution", text="hello")
        ),
        timeout=0.1,
    )
    await asyncio.wait_for(
        composite.publish_execution(
            "execution", ExecutionCompleted(execution_id="execution")
        ),
        timeout=0.1,
    )
    assert (await subscription.__anext__()).text == "hello"
    assert isinstance(await subscription.__anext__(), ExecutionCompleted)
    await composite.close_execution("execution")
    gate.set()


@pytest.mark.asyncio
async def test_interaction_drains_fast_execution_before_response() -> None:
    principal = trusted_local_principal(tenant_id="tenant")
    record = SessionRecord(
        id="session",
        user_id=None,
        tenant_id="tenant",
        next_turn_sequence=1,
        latest_completed_run_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        state=SessionState.OPEN,
    )
    active = ActiveRuntimeSession(record)
    coordinator = SessionOperationCoordinator()
    hub = ExecutionEventHub()

    class Sessions:
        def set_interaction_canceller(self, callback):
            self.cancel_callback = callback

        def set_interaction_owner(self, owner):
            self.owner = owner

        async def reserve(self, session_id, kind, *, principal, execution_id):
            return await coordinator.reserve(active, kind, execution_id=execution_id)

        async def toolsets(self, session_id, *, principal, lease):
            assert active.operation is lease
            return ()

    class Execution:
        async def run(self, *args, execution_id, **kwargs):
            await hub.publish(
                execution_id,
                AssistantTextDelta(execution_id=execution_id, text="text-1"),
            )
            await hub.publish(
                execution_id,
                AssistantTextDelta(execution_id=execution_id, text="text-2"),
            )
            await hub.publish(execution_id, ExecutionCompleted(execution_id=execution_id))
            return None

        async def get_execution_record(self, execution_id, *, principal):
            return SimpleNamespace(status=RunStatus.COMPLETED)

    observed: list[ExecutionEvent] = []

    class Observer:
        async def publish(self, event):
            observed.append(event)

        async def request_approval(self, request, cancellation):
            return None

    service = InteractiveRunService(Execution(), hub, Sessions())
    result = await service.execute(
        session_id="session",
        spec=object(),
        prompt="prompt",
        observer=Observer(),
        principal=principal,
    )
    assert result.status is RunStatus.COMPLETED
    assert [event.text for event in observed[:2]] == ["text-1", "text-2"]
    assert isinstance(observed[-1], ExecutionCompleted)
    assert active.operation is None


@pytest.mark.asyncio
async def test_acp_owner_identity_preserves_replacement() -> None:
    client = AcpClient(project_root=".")
    old = client.resource_owner("session")
    replacement_resources = AcpClientSessionResources(client, "session")
    client._resources["session"] = replacement_resources
    replacement = client.resource_owner("session")
    await old.close("session")
    assert client._resources["session"] is replacement._resources


def test_runtime_builder_rejects_two_event_hubs(tmp_path) -> None:
    with pytest.raises(RuntimeInitializationError, match="multiple_execution_event_hubs"):
        build_runtime(
            storage=LocalDirectoryStorage(tmp_path),
            dependencies=RuntimeDependencies(
                model_resolver=make_router(), live_events=ExecutionEventHub()
            ),
            event_hub=ExecutionEventHub(),
        )
