#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Focused regression coverage for the final ACP lifecycle boundaries."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.acp.client import AcpClient
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.execution.domain import RunStatus
from linktools.ai.execution.live_events import ExecutionEventHub, ExecutionFailed
from linktools.ai.governance.identity import trusted_local_principal
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime
from linktools.ai.runtime.session import (
    RuntimeSessionService,
    SessionOperationKind,
    SessionSettings,
    SessionWorkspace,
)
from linktools.ai.errors import SessionClosedError
from tests.ai.fakes.model import make_router


@pytest.mark.asyncio
async def test_detached_terminal_request_is_compensated_after_late_response() -> None:
    started = asyncio.Event()
    response_ready = asyncio.Event()
    killed: list[str] = []
    released: list[str] = []
    request_cancelled = False

    class Connection:
        async def create_terminal(self, session_id: str, **kwargs: object) -> object:
            nonlocal request_cancelled
            started.set()
            try:
                await response_ready.wait()
            except asyncio.CancelledError:
                request_cancelled = True
                raise
            return SimpleNamespace(terminal_id="late-terminal")

        async def kill_terminal(self, session_id: str, terminal_id: str) -> None:
            killed.append(terminal_id)

        async def release_terminal(self, session_id: str, terminal_id: str) -> None:
            released.append(terminal_id)

    client = AcpClient(project_root=".")
    client.set_connection(Connection(), SimpleNamespace(terminal=True))
    session = SimpleNamespace(
        record=SimpleNamespace(id="session", state=SimpleNamespace(value="open")),
        closing_requested=False,
        active_execution_id="execution",
    )

    caller = asyncio.create_task(client.create_terminal(session))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert client.client_operation_count == 1
    response_ready.set()
    await asyncio.wait_for(_wait_for_no_operations(client), timeout=2.0)
    assert not request_cancelled
    assert killed == ["late-terminal"]
    assert released == ["late-terminal"]
    assert not client.resources("session").terminal_handles


@pytest.mark.asyncio
async def test_close_retains_noncooperative_fs_request_until_late_completion(tmp_path) -> None:
    started = asyncio.Event()
    response_ready = asyncio.Event()
    request_cancelled = False

    class Connection:
        async def read_text_file(self, session_id: str, path: str, **kwargs: object) -> object:
            nonlocal request_cancelled
            started.set()
            try:
                await response_ready.wait()
            except asyncio.CancelledError:
                request_cancelled = True
                raise
            return SimpleNamespace(content="late")

    client = AcpClient(project_root=tmp_path)
    client.set_connection(
        Connection(), SimpleNamespace(fs=SimpleNamespace(read_text_file=True))
    )
    session = SimpleNamespace(
        record=SimpleNamespace(
            id="session",
            workspace=SimpleNamespace(cwd=str(tmp_path), additional_directories=()),
        )
    )
    caller = asyncio.create_task(client.read_text_file(session, "answer.txt"))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    resources = client.resources("session")
    failures = await resources.close("session")
    assert failures and resources.operation_count == 1
    assert not request_cancelled
    response_ready.set()
    await asyncio.wait_for(_wait_for_no_operations(client), timeout=2.0)
    assert resources.is_empty("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ("compile", "usage", "start"))
async def test_execution_preparation_failure_publishes_one_failed_boundary(
    tmp_path, failure_stage: str
) -> None:
    hub = ExecutionEventHub()
    runtime = build_runtime(
        storage=LocalDirectoryStorage(tmp_path),
        event_hub=hub,
        model_resolver=make_router(),
    )
    spec = AgentSpec(
        "agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer")
    )
    principal = trusted_local_principal(tenant_id="tenant")
    subscription = await hub.subscribe("execution")

    async def fail_compile(specification: object) -> object:
        raise RuntimeError("compile failed")

    async def fail_usage(record: object) -> object:
        raise RuntimeError("usage failed")

    async def fail_start(
        execution_id: str, coroutine: object, token: object
    ) -> object:
        close = getattr(coroutine, "close", None)
        if close is not None:
            close()
        raise RuntimeError("start failed")

    if failure_stage == "compile":
        runtime.execution._compiler.compile = fail_compile
    elif failure_stage == "usage":
        runtime.execution._usage_capture = fail_usage
    else:
        runtime.execution._controller.start = fail_start

    with pytest.raises(RuntimeError):
        await runtime.run(
            spec,
            "hello",
            principal=principal,
            session_id="session",
            execution_id="execution",
        )

    event = await asyncio.wait_for(subscription.__anext__(), timeout=1.0)
    record = await runtime.execution._store.get_run("execution")
    assert isinstance(event, ExecutionFailed)
    assert record is not None and record.status is RunStatus.FAILED
    assert hub.active_subscription_count == 0
    await subscription.release()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_closed_session_validation_does_not_grow_active_cache(tmp_path) -> None:
    principal = trusted_local_principal(tenant_id="tenant")
    storage = LocalDirectoryStorage(tmp_path)
    service = RuntimeSessionService(storage.execution)
    workspace = SessionWorkspace(cwd=str(tmp_path))
    for index in range(20):
        session_id = f"closed-{index}"
        await service.create(
            session_id=session_id,
            workspace=workspace,
            settings=SessionSettings(agent_id="default"),
            principal=principal,
        )
        assert (await service.close(session_id, principal=principal)).closed

    assert service.active_session_count == 0
    assert len(await service.list(principal=principal)) == 20
    assert service.active_session_count == 0
    with pytest.raises(SessionClosedError):
        await service.update("closed-0", principal=principal)
    with pytest.raises(SessionClosedError):
        await service.fork(
            "closed-0",
            "forked",
            workspace=workspace,
            settings=SessionSettings(agent_id="default"),
            principal=principal,
        )
    with pytest.raises(SessionClosedError):
        await service.reserve(
            "closed-0", SessionOperationKind.PROMPT, principal=principal
        )
    assert service.active_session_count == 0


async def _wait_for_no_operations(client: AcpClient) -> None:
    while client.client_operation_count:
        await asyncio.sleep(0)
