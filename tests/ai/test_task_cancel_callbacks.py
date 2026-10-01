#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declared Task cleanup and external effect certainty remain separate."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_policy", ("none", "non_replay_safe"))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_declared_cancel_callback_runs_without_claiming_external_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effect_policy: str,
    cleanup_fails: bool,
) -> None:
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path / "linktools"))
    started = asyncio.Event()
    stopped = asyncio.Event()
    calls: list[str] = []

    async def execute(context: TaskNodeContext[None]) -> JsonValue:
        del context
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return None

    async def cancel(context: TaskNodeContext[None]) -> None:
        assert stopped.is_set()
        calls.append(context.execution_id)
        if cleanup_fails:
            raise RuntimeError("cleanup outcome is uncertain")

    task = Task("declared-cleanup", execute, effect_policy=effect_policy, cancel=cancel)
    graph = TaskGraph("declared-cleanup", (TaskNode("node", task=task),))
    storage = RuntimeStorage.in_memory()
    try:
        async with Runtime.open(
            "declared-cleanup",
            models=RuntimeUsageModels(),
            storage=storage,
        ) as runtime:
            run = await runtime.tasks.bind(task).start(graph, idempotency_key="start")
            await asyncio.wait_for(started.wait(), timeout=10)
            execution = await run.execution("node")
            result = await asyncio.wait_for(
                run.cancel(idempotency_key="cancel"), timeout=10
            )
            expected = (
                TaskStatus.CANCELLED
                if effect_policy == "none" and not cleanup_fails
                else TaskStatus.RECOVERY_REQUIRED
            )
            assert result.status is expected
            assert calls == [execution.execution_id]
            view = await runtime.executions.inspect(
                execution.execution_id,
                principal=runtime.default_principal,
            )
            if expected is TaskStatus.CANCELLED:
                assert view.status is ExecutionStatus.CANCELLED
            elif not cleanup_fails:
                assert view.status is ExecutionStatus.RECOVERY_REQUIRED
                state = await run.state(include_content=True)
                assert (
                    state.node_states[0].error_code
                    == ErrorCode.TASK_EFFECT_UNKNOWN.value
                )
            else:
                assert view.status not in {
                    ExecutionStatus.CANCELLED,
                    ExecutionStatus.SUCCEEDED,
                }

            repeated = await run.cancel(idempotency_key="cancel")
            assert repeated.status is expected
            if expected is TaskStatus.RECOVERY_REQUIRED:
                recovered = await run.recover(idempotency_key="recover")
                assert recovered.status is TaskStatus.RECOVERY_REQUIRED
            assert calls == [execution.execution_id]
    finally:
        await storage.close()
