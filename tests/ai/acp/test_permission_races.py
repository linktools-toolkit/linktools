#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import acp.schema as schema
import pytest

from linktools.ai.acp.agent import LinktoolsAcpAgent
from linktools.ai.acp.capabilities import AcpMode
from linktools.ai.acp.client_services import AcpClientServices
from linktools.ai.acp.persistence import AcpSessionRecord, AcpSessionRepository
from linktools.ai.acp.session_models import ActiveAcpSession
from linktools.ai.acp.sessions import AcpSessionService
from linktools.ai.execution.domain import ApprovalDecision, RunStatus
from linktools.ai.execution.live_events import ExecutionCompleted, ExecutionEventHub, ExecutionPaused
from linktools.ai.governance.identity import trusted_local_principal


class _Runtime:
    def __init__(self, hub: ExecutionEventHub) -> None:
        self.execution_event_hub = hub
        self.current = None
        self.cancel_count = 0
        self.decide_count = 0

    async def inspect(self, *, run_id, principal):
        return SimpleNamespace(tool_calls=(SimpleNamespace(arguments={}),))

    async def get_execution_record(self, execution_id, *, principal):
        return self.current

    async def cancel(self, execution_id, *, principal):
        self.cancel_count += 1
        self.current.status = RunStatus.CANCELLED

    async def decide_approval(self, execution_id, *, approval_id, decision, principal):
        self.decide_count += 1
        assert decision in {ApprovalDecision.ALLOW, ApprovalDecision.DENY}


def _active(tmp_path, runtime: _Runtime, session_id: str) -> ActiveAcpSession:
    now = datetime.now(timezone.utc)
    record = AcpSessionRecord(
        schema_version=1,
        session_id=session_id,
        cwd=str(tmp_path),
        additional_directories=(),
        mode_id="default",
        config_values={},
        mcp_server_fingerprints=(),
        title=None,
        closed=False,
        created_at=now,
        updated_at=now,
    )
    runtime.current = SimpleNamespace(
        id="execution-1",
        status=RunStatus.PAUSED,
        approval=SimpleNamespace(
            approval_id="approval-1",
            tool_call_id="tool-1",
            tool_name="tool",
        ),
    )
    return ActiveAcpSession(
        record,
        asyncio.Lock(),
        "execution-1",
        SimpleNamespace(),
        set(),
        set(),
    )


@pytest.mark.asyncio
async def test_cancel_interrupts_pending_permission_and_ignores_late_allow(tmp_path) -> None:
    hub = ExecutionEventHub()
    runtime = _Runtime(hub)
    service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path / "state"),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )

    async def resolve(mode):
        return object()

    agent = LinktoolsAcpAgent(
        runtime=runtime,
        event_hub=hub,
        session_service=service,
        project_root=str(tmp_path),
        spec_resolver=resolve,
        modes=(AcpMode("default", "Default"),),
    )
    agent._initialized = True

    for _ in range(100):
        started = asyncio.Event()
        release = asyncio.Event()

        class Connection:
            async def request_permission(self, session_id, tool_call, options):
                started.set()
                await release.wait()
                return schema.RequestPermissionResponse(
                    outcome=schema.AllowedOutcome(
                        optionId="allow_once",
                        outcome="selected",
                    )
                )

        active = _active(tmp_path, runtime, "session-1")
        service.active_sessions.clear()
        service.active_sessions[active.record.session_id] = active
        agent.on_connect(Connection())
        permission = asyncio.create_task(
            agent.prompt_service.request_permission(active, runtime.current)
        )
        await started.wait()
        await asyncio.wait_for(agent.cancel(active.record.session_id), timeout=0.1)
        release.set()
        assert await permission is None
        assert active.pending_permission is None
        assert runtime.decide_count == 0
        assert active.active_execution_id is None
        assert await asyncio.wait_for(active.lock.acquire(), timeout=0.1)
        active.lock.release()

    assert runtime.cancel_count == 100


@pytest.mark.asyncio
async def test_prompt_handles_two_sequential_permissions(tmp_path) -> None:
    hub = ExecutionEventHub()
    runtime = _Runtime(hub)
    runtime.resume_count = 0

    async def run(spec, prompt, *, principal, session_id, execution_id, extra_toolsets):
        runtime.current = SimpleNamespace(
            id=execution_id,
            status=RunStatus.PAUSED,
            approval=SimpleNamespace(
                approval_id="approval-1",
                tool_call_id="tool-1",
                tool_name="first",
            ),
        )
        await hub.publish(
            execution_id,
            ExecutionPaused(
                execution_id=execution_id,
                approval_id="approval-1",
                tool_call_id="tool-1",
                tool_name="first",
            ),
        )

    async def resume(execution_id, *, principal, extra_toolsets):
        runtime.resume_count += 1
        if runtime.resume_count == 1:
            runtime.current = SimpleNamespace(
                id=execution_id,
                status=RunStatus.PAUSED,
                approval=SimpleNamespace(
                    approval_id="approval-2",
                    tool_call_id="tool-2",
                    tool_name="second",
                ),
            )
            await hub.publish(
                execution_id,
                ExecutionPaused(
                    execution_id=execution_id,
                    approval_id="approval-2",
                    tool_call_id="tool-2",
                    tool_name="second",
                ),
            )
        else:
            runtime.current.status = RunStatus.COMPLETED
            await hub.close(execution_id, ExecutionCompleted(execution_id=execution_id))

    async def decide(execution_id, *, approval_id, decision, principal):
        runtime.decide_count += 1
        assert decision is ApprovalDecision.ALLOW
        runtime.current.status = RunStatus.RUNNING

    async def create_session(session_id, *, principal):
        return None

    runtime.create_session = create_session
    runtime.run = run
    runtime.resume = resume
    runtime.decide_approval = decide
    session_service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path / "state"),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )

    async def resolve(mode):
        return object()

    agent = LinktoolsAcpAgent(
        runtime=runtime,
        event_hub=hub,
        session_service=session_service,
        project_root=str(tmp_path),
        spec_resolver=resolve,
        modes=(AcpMode("default", "Default"),),
    )
    agent._initialized = True
    permission_count = 0

    class Connection:
        async def session_update(self, session_id, update):
            return None

        async def request_permission(self, session_id, tool_call, options):
            nonlocal permission_count
            permission_count += 1
            return schema.RequestPermissionResponse(
                outcome=schema.AllowedOutcome(
                    optionId="allow_once",
                    outcome="selected",
                )
            )

    agent.on_connect(Connection())
    active = await session_service.create(cwd=str(tmp_path))

    response = await agent.prompt(
        active.record.session_id,
        [schema.TextContentBlock(type="text", text="run")],
    )

    assert response.stop_reason == "end_turn"
    assert permission_count == 2
    assert runtime.decide_count == 2
    assert runtime.resume_count == 2


@pytest.mark.asyncio
async def test_cancel_bounds_permission_callback_that_ignores_cancellation(tmp_path) -> None:
    hub = ExecutionEventHub()
    runtime = _Runtime(hub)
    service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path / "state"),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )
    active = _active(tmp_path, runtime, "session-1")
    service.active_sessions[active.record.session_id] = active
    callback_started = asyncio.Event()
    force_finish = asyncio.Event()

    class Connection:
        async def request_permission(self, session_id, tool_call, options):
            callback_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await force_finish.wait()
                raise

    agent = LinktoolsAcpAgent(
        runtime=runtime,
        event_hub=hub,
        session_service=service,
        project_root=str(tmp_path),
        spec_resolver=lambda mode: None,
        modes=(AcpMode("default", "Default"),),
    )
    agent._initialized = True
    agent.on_connect(Connection())
    permission = asyncio.create_task(
        agent.prompt_service.request_permission(active, runtime.current)
    )
    await callback_started.wait()

    await asyncio.wait_for(agent.cancel(active.record.session_id), timeout=2)
    assert runtime.cancel_count == 1
    force_finish.set()
    assert await permission is None
