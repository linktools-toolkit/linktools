#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The public inactivity facade observes a real worker without consuming its result."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.history import History
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.storage import InMemoryCacheBackend

from .test_pending_session_projection import _runtimes


@pytest.mark.asyncio
async def test_runtime_inactivity_waits_for_local_worker_and_preserves_retained_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, shared=False) as (runtimes, entered, finish, block):
        runtime, state = runtimes[0]
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        block()
        execution = await session.start("question", idempotency_key="pending")
        await asyncio.wait_for(entered.wait(), 5)
        # Terminal-only persistence leaves this live admission outside the SQL execution table.
        assert await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
        waiting = asyncio.Event()
        wait_terminal = LocalExecutionBackend.wait_terminal

        async def observed_wait(backend, execution_id, *, tenant_id):
            assert execution_id == execution.execution_id
            assert backend.owns_execution(execution_id, tenant_id=tenant_id)
            waiting.set()
            await wait_terminal(backend, execution_id, tenant_id=tenant_id)

        async def forbidden_consumption(*args, **kwargs):
            pytest.fail("inactivity observation must not consume or materialize the execution result")

        with monkeypatch.context() as patch:
            patch.setattr(LocalExecutionBackend, "wait_terminal", observed_wait)
            patch.setattr(runtime.executions._service, "wait", forbidden_consumption)
            patch.setattr(runtime.executions._service, "result", forbidden_consumption)
            patch.setattr(runtime.executions._service, "consume_result_from_execution", forbidden_consumption)
            patch.setattr(History, "result", forbidden_consumption)
            task = asyncio.create_task(runtime.sessions.wait_inactive(
                "session", principal=principal, timeout_seconds=5,
            ))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                assert not task.done()
                finish.set()
                view = await asyncio.wait_for(task, 5)
                assert view.session_id == "session"
                assert view.active_execution_id is None
            finally:
                finish.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task

        # Read retained output only after the observer has finished and its guards are removed.
        result = await runtime.history.result(execution.execution_id, principal=principal)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "answer"}
