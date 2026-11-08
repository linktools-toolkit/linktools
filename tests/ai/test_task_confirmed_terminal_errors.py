#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
import pytest
from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

@pytest.mark.asyncio
async def test_artifact_confirmation_failure_settles_task_without_replaying_body(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("probe", storage=PersistenceStorage.in_memory()) as state:
        original = state.artifact.records.put_metadata
        pending = True
        async def uncertain(record):
            nonlocal pending
            result = await original(record)
            if pending:
                pending = False
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        get = state.artifact.records.get_metadata
        async def unavailable_once(*args, **kwargs):
            if not pending:
                monkeypatch.setattr(state.artifact.records, "get_metadata", get)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await get(*args, **kwargs)
        monkeypatch.setattr(state.artifact.records, "put_metadata", uncertain)
        monkeypatch.setattr(state.artifact.records, "get_metadata", unavailable_once)
        calls = 0
        async def body(ctx):
            nonlocal calls
            calls += 1
            async def chunks():
                yield b"data"
            await ctx.artifacts.publish("a", chunks(), mime_type="text/plain", expected_size=4, expected_digest=hashlib.sha256(b"data").hexdigest())
            return "done"
        task = Task("publish", body, effect_policy="none")
        async with TaskEngine.open(task, persistence=state) as engine:
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
            initial = (await run.wait(timeout_seconds=5)).result
            assert initial.status is TaskStatus.FAILED
            assert initial.node_states[0].error_code == ErrorCode.STORAGE_COMMIT_UNKNOWN.value
            assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (
    ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_RECOVERY_REQUIRED,
    ErrorCode.EXECUTION_START_UNKNOWN, ErrorCode.STORAGE_CONFLICT,
    ErrorCode.EXECUTION_RESULT_CONFLICT,
    ErrorCode.TASK_FENCE_STALE,
))
async def test_confirmed_callable_failure_retains_error_without_recovery_loop(code: ErrorCode) -> None:
    calls = 0
    async def body(ctx):
        nonlocal calls
        calls += 1
        raise AIError(code)
    task = Task("failure", body, effect_policy="none")
    async with Persistence.open("terminal-errors", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(task, persistence=state) as engine:
        run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        result = (await run.wait(timeout_seconds=5)).result
        assert result.status is TaskStatus.FAILED
        assert result.node_states[0].error_code == code.value
        assert calls == 1


async def _persistent_terminal_failure(path: Path, stage: str) -> None:
    from linktools.ai.task import TaskNodeRunError
    calls = 0
    async def body(ctx):
        nonlocal calls
        calls += 1
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    task = Task("persistent-failure", body, effect_policy="none")
    async with Persistence.open("persistent-terminal", storage=PersistenceStorage.sqlite(path)) as state, TaskEngine.open(task, persistence=state) as engine:
        original = TaskNodeRunError.__init__
        def legacy_error(self, *args, **kwargs):
            kwargs.pop("terminal_status", None)
            original(self, *args, **kwargs)
        if stage == "seed":
            TaskNodeRunError.__init__ = legacy_error
        try:
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
            if stage == "recover":
                await run.recover(idempotency_key="recover")
            result = (await run.wait(timeout_seconds=5)).result
            expected = TaskStatus.RECOVERY_REQUIRED if stage == "seed" else TaskStatus.FAILED
            assert result.status is expected
            assert result.node_states[0].error_code == ErrorCode.STORAGE_COMMIT_UNKNOWN.value
            assert calls == (1 if stage == "seed" else 0)
        finally:
            TaskNodeRunError.__init__ = original


def test_preexisting_failed_unknown_recovers_in_fresh_process(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve()
    for stage in ("seed", "recover"):
        completed = subprocess.run(
            [sys.executable, str(script), str(tmp_path / "state.sqlite"), stage],
            env=dict(os.environ), capture_output=True, text=True, timeout=30,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr




@pytest.mark.asyncio
async def test_non_replay_safe_unknown_requires_recovery_without_body_replay() -> None:
    calls = 0
    async def body(ctx):
        nonlocal calls
        calls += 1
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    task = Task("external", body, effect_policy="non_replay_safe")
    async with Persistence.open("external-unknown", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(task, persistence=state) as engine:
        run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
        await run.recover(idempotency_key="recover")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.RECOVERY_REQUIRED
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_CONFLICT))
async def test_effect_resolution_uses_confirmed_cancelled_state(code: ErrorCode) -> None:
    from linktools.ai.core import ExecutionStatus, Principal
    from linktools.ai.task import LocalTaskGraphLauncher, TaskGraphLaunch, TaskGraphLimits, TaskEffectResolution, TaskNodeRunError
    from tests.ai._task_test_helpers import admit_graph

    class Runner:
        async def resolve_effect(self, invocation, execution_id, resolution):
            raise TaskNodeRunError(code, execution_id, terminal_status=ExecutionStatus.CANCELLED)

    async with Persistence.open("resolved-cancel", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("node"),)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="owner", lease_seconds=30)
        await repository.mark_recovery_required(
            lease, tenant_id="tenant", execution_id="execution",
            error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value, error_digest="b" * 64,
        )
        launcher = LocalTaskGraphLauncher(repository, Runner(), owner="owner")
        try:
            await launcher.resolve_effect(
                TaskGraphLaunch("graph", Principal("actor", "tenant"), TaskGraphLimits()),
                "node", "execution", lease.fence, TaskEffectResolution("not_applied"),
            )
            node = (await repository.list_nodes("graph", tenant_id="tenant"))[0]
            assert node.status is TaskStatus.CANCELLED
        finally:
            await launcher.shutdown()


def test_terminal_error_rejects_untyped_or_nonterminal_evidence() -> None:
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.task import TaskNodeRunError
    for value in ("CANCELLED", "FAILED", ExecutionStatus.STARTED, True):
        with pytest.raises(ValueError):
            TaskNodeRunError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "execution", terminal_status=value)


if __name__ == "__main__":
    import asyncio
    import sys
    asyncio.run(_persistent_terminal_failure(Path(sys.argv[1]), sys.argv[2]))
