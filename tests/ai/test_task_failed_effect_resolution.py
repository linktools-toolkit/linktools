#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit failure acceptance preserves already performed Task effects."""
import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskEffectResolution, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("sql", (False, True))
@pytest.mark.parametrize("lost_stage", ("graph", "execution"))
async def test_failed_effect_resolution_retains_partial_effect_and_reopens(tmp_path: Path, sql: bool, lost_stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    effects: list[str] = []
    principal = Principal("owner", "default")


    async def body(ctx):
        effects.append("applied")
        raise RuntimeError("failed after partial effect")

    async def dependent(ctx):
        effects.append("dependent")
        return "unexpected"

    task = Task("partial", body, effect_policy="non_replay_safe")
    child = Task("dependent", dependent, effect_policy="none")
    storage = PersistenceStorage.sqlite(tmp_path / "task.sqlite") if sql else PersistenceStorage.from_root(tmp_path / "local", exclusive_writer=True)
    for first in (True, False):
        async with Persistence.open("partial", storage=storage) as persistence, TaskEngine.open(task, child, persistence=persistence) as engine:
            if first:
                run = await engine.start(TaskGraph("partial", (TaskNode("n", task=task), TaskNode("dependent", task=child, dependencies=("n",)))), principal=principal, idempotency_key="start")
                assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
                node = next(n for n in (await run.state()).node_states if n.node_id == "n")
                fence = node.fence
                assert node.error_code == ErrorCode.TASK_EFFECT_UNKNOWN.value
                assert effects == ["applied"]
                resolution = TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED, safe_details={"phase": "after_partial_effect"})
                await run.resolve_effect("n", fence, TaskEffectResolution("unknown"), idempotency_key="unknown")
                assert (await run.state()).status is TaskStatus.RECOVERY_REQUIRED
                with pytest.raises(AIError) as stale:
                    await run.resolve_effect("n", fence + 1, resolution, idempotency_key="stale")
                assert stale.value.code in {ErrorCode.TASK_NOT_READY, ErrorCode.TASK_FENCE_STALE}
                with pytest.raises(AIError) as denied:
                    outsider = await engine.get("partial", principal=Principal("outsider", persistence.tenant_id))
                    await outsider.resolve_effect("n", fence, resolution, idempotency_key="other")
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                owner = persistence.task.tasks if lost_stage == "graph" else persistence.execution.executions
                method = "fail" if lost_stage == "graph" else "commit_terminal"
                original_fail = getattr(owner, method)
                async def lost_response(*args, **kwargs):
                    await original_fail(*args, **kwargs)
                    monkeypatch.setattr(owner, method, original_fail)
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                monkeypatch.setattr(owner, method, lost_response)
                with pytest.raises(AIError) as lost:
                    await run.resolve_effect("n", fence, resolution, idempotency_key="failure")
                assert lost.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            else:
                run = await engine.get("partial", principal=principal)
            await run.resolve_effect("n", fence, resolution, idempotency_key="failure")
            state = await run.state()
            assert state.status is TaskStatus.FAILED
            assert next(n for n in state.node_states if n.node_id == "n").status is TaskStatus.FAILED
            assert next(n for n in state.node_states if n.node_id == "n").error_code == ErrorCode.TASK_NODE_FAILED.value
            assert effects == ["applied"]
            with pytest.raises(AIError) as conflict:
                await run.resolve_effect("n", fence, TaskEffectResolution("failed", error_code=ErrorCode.EXECUTION_WAIT_TIMEOUT), idempotency_key="failure")
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_failed_effect_resolution_does_not_enable_automatic_failure_acceptance(tmp_path: Path) -> None:
    calls = 0

    async def body(ctx):
        nonlocal calls
        calls += 1
        raise RuntimeError("partial effect")

    async def reconcile(ctx):
        return TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED)

    task = Task("partial", body, effect_policy="non_replay_safe", reconcile=reconcile)
    async with Persistence.open("partial", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence, TaskEngine.open(task, persistence=persistence) as engine:
        run = await engine.start(TaskGraph("graph", (TaskNode("n", task=task),)), idempotency_key="start")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
        await run.recover(idempotency_key="recover")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
        assert calls == 1


@pytest.mark.asyncio
async def test_failed_effect_resolution_parallel_confirmations(tmp_path: Path) -> None:
    calls = 0

    async def body(ctx):
        nonlocal calls
        calls += 1
        raise RuntimeError("partial effect")

    task = Task("partial", body, effect_policy="non_replay_safe")
    async with Persistence.open("parallel", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence, TaskEngine.open(task, persistence=persistence) as engine:
        run = await engine.start(TaskGraph("parallel", (TaskNode("n", task=task),)), idempotency_key="start")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
        fence = (await run.state()).node_states[0].fence
        resolution = TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED)
        results = await asyncio.gather(*(run.resolve_effect("n", fence, resolution, idempotency_key="accept") for _ in range(2)), return_exceptions=True)
        assert any(not isinstance(result, BaseException) for result in results)
        for result in results:
            if isinstance(result, BaseException):
                assert isinstance(result, AIError)
                assert result.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.TASK_FENCE_STALE, ErrorCode.EXECUTION_RESULT_CONFLICT}
        await run.resolve_effect("n", fence, resolution, idempotency_key="accept")
        assert (await run.state()).status is TaskStatus.FAILED
        assert calls == 1


@pytest.mark.asyncio
async def test_failed_effect_resolution_waits_for_sibling_drain(tmp_path: Path) -> None:
    entered, cleanup, release, drained = (asyncio.Event() for _ in range(4))

    async def partial(ctx):
        await entered.wait()
        raise RuntimeError("partial effect")

    async def sibling(ctx):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()
            drained.set()

    task = Task("partial", partial, effect_policy="non_replay_safe")
    other = Task("sibling", sibling, effect_policy="none")
    async with Persistence.open("drain", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence, TaskEngine.open(task, other, persistence=persistence) as engine:
        run = await engine.start(TaskGraph("graph", (TaskNode("n", task=task), TaskNode("sibling", task=other))), idempotency_key="start")
        await asyncio.wait_for(cleanup.wait(), 5)
        state = await run.state()
        node = next(n for n in state.node_states if n.node_id == "n")
        async with Persistence.open("drain", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as second_state, TaskEngine.open(task, other, persistence=second_state) as second_engine:
            second_run = await second_engine.get("graph")
            with pytest.raises(AIError) as foreign:
                await second_run.resolve_effect("n", node.fence, TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED), idempotency_key="second-owner")
            assert foreign.value.code is ErrorCode.TASK_NOT_READY
            assert not drained.is_set()
            assert (await second_run.state()).status is TaskStatus.RECOVERY_REQUIRED
        acceptance = asyncio.create_task(run.resolve_effect("n", node.fence, TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED), idempotency_key="accept"))
        try:
            await asyncio.sleep(0.02)
            assert not acceptance.done()
        finally:
            release.set()
        with pytest.raises(AIError) as unsettled:
            await asyncio.wait_for(acceptance, 5)
        assert unsettled.value.code is ErrorCode.TASK_NOT_READY
        assert drained.is_set()
        assert (await run.state()).status is TaskStatus.RECOVERY_REQUIRED
        await run.cancel(idempotency_key="cancel-siblings")
        settled = await run.state()
        assert settled.status is TaskStatus.RECOVERY_REQUIRED
        assert next(n for n in settled.node_states if n.node_id == "n").status is TaskStatus.RECOVERY_REQUIRED
        assert next(n for n in settled.node_states if n.node_id == "sibling").status is TaskStatus.CANCELLED
        await run.resolve_effect("n", node.fence, TaskEffectResolution("failed", error_code=ErrorCode.TASK_NODE_FAILED), idempotency_key="accept")
        assert (await run.state()).status is TaskStatus.FAILED
