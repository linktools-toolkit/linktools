#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional writer ownership is bounded by active execution producers."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError
from linktools.ai.runtime import Runtime, RuntimeStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_capability_atomic_writes import _opened
pytestmark = pytest.mark.asyncio


async def test_ordinary_runs_do_not_allocate_optional_writer_manager() -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("ordinary-lifecycle")
    features.agent("default", model="default", allow_tools=())
    async with Runtime.open("ordinary-lifecycle", storage=RuntimeStorage.in_memory(), features=features,
                            models=_Models(_UsageFunctionModel(model))) as runtime:
        for _ in range(12):
            execution = await runtime.agents.get().start("question")
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            assert runtime._state.run_store.capability_writes is None
            assert runtime._state.run_store._execution_producers == {}


@pytest.mark.parametrize("terminal", (False, True))
async def test_completed_custom_runs_release_guards_but_reject_late_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool,
) -> None:
    policy = CommitPolicy.terminal() if terminal else CommitPolicy.sync()
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy), terminal=terminal) as opened:
        run_store = opened.runtime._state.run_store
        manager = run_store.capability_writes
        assert manager is not None
        for index in range(12):
            opened.release.clear()
            bound, _ = await opened.start()
            execution_id = opened.executions[-1].execution_id
            run_id = next(scope.run_id for scope in manager._scopes.values() if scope.execution_id == execution_id)
            assert bound.writer is not None
            if not terminal:
                await bound.writer.submit(key=f"write-{index}", payload={"run": execution_id, "value": index})
            await opened.finish()
            assert run_store._execution_producers == {}
            assert manager._sealed == set()
            assert manager._scopes == {}
            with pytest.raises(AIError):
                await bound.writer.submit(key="late", payload={"value": 1})
            with pytest.raises(AIError):
                manager.bind("audit", run_id, execution_id, opened.templates[0])


async def test_failed_terminal_prefix_keeps_ownership_until_confirmed_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CommitPolicy.terminal()
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        bound, _ = await opened.start()
        execution_id = opened.executions[-1].execution_id
        run_store = opened.runtime._state.run_store
        manager = run_store.capability_writes
        assert manager is not None and bound.writer is not None
        scope = next(scope for scope in manager._scopes.values() if scope.execution_id == execution_id)
        await bound.writer.submit(key="retained", payload={"run": execution_id})
        opened.probe.fail.add("audit")
        with pytest.raises(AIError):
            await manager.prepare_terminal(execution_id, None, scope.group)
        assert scope.pending and execution_id in manager._sealed
        assert run_store.has_active_execution_producer(execution_id)
        with pytest.raises(AIError):
            manager.bind("audit", scope.run_id, execution_id, opened.templates[0])
        with pytest.raises(AIError):
            await bound.writer.submit(key="late", payload={"run": execution_id})
        opened.probe.fail.clear()
        await bound.writer.flush()
        assert not scope.pending
        assert execution_id in manager._sealed
        await opened.finish()
        assert manager._scopes == {} and manager._sealed == set()
        assert run_store._execution_producers == {}


@pytest.mark.parametrize("pending", (False, True))
async def test_releasing_one_producer_seals_only_its_own_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending: bool,
) -> None:
    policy = CommitPolicy.terminal()
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        original, _ = await opened.start()
        execution_id = opened.executions[-1].execution_id
        run_store = opened.runtime._state.run_store
        manager = run_store.capability_writes
        assert manager is not None and original.writer is not None
        active_scope = next(iter(manager._scopes.values()))
        released_run = active_scope.run_id + "-superseded"
        run_store.bind_execution_producer(released_run, execution_id=execution_id, producer_generation=0)
        released = manager.bind("audit", released_run, execution_id, opened.templates[0])
        assert released.writer is not None
        if pending:
            await released.writer.submit(key="accepted", payload={"run": released_run})
        await run_store.release_staging_many(candidate_agent_run_ids=(released_run,), execution_id=execution_id)
        assert active_scope in manager._scopes.values() and active_scope.accepting
        assert run_store.has_active_execution_producer(execution_id)
        with pytest.raises(AIError):
            await released.writer.submit(key="late", payload={"run": released_run})
        with pytest.raises(AIError):
            manager.bind("audit", released_run, execution_id, opened.templates[0])
        if pending:
            assert any(scope.run_id == released_run and scope.pending for scope in manager._scopes.values())
            await released.writer.flush()
        await original.writer.submit(key="active", payload={"run": execution_id})
        await opened.finish()
        assert manager._scopes == {} and manager._sealed == set()
