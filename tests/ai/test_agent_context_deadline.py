#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Public admission deadlines retain their durable scope without hot-path reads."""
from ._runtime_test_helpers import model_features

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from linktools.ai.capability import ExecutionCapability, CheckpointCapability

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, Principal, RunBudget, TaskStatus
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.execution import ExecutionRequest
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import RuntimeUsageModels


_DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)


class _Capture(AbstractCapability[AgentContext[None]]):
    id = "host.deadline"

    def __init__(self, contexts: list[AgentContext[None]]) -> None:
        self.contexts = contexts

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        self.contexts.append(ctx.deps)


def _group(contexts: list[AgentContext[None]]) -> RuntimeFeatures[None]:
    group: RuntimeFeatures[None] = RuntimeFeatures("deadline")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    group.capability(_Capture(contexts), scope="runtime")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    return group


def test_deadline_is_optional_readonly_and_timezone_aware() -> None:
    context = AgentContext(None, Principal("user", "tenant"), "deadline", None, "execution", {})
    assert context.deadline_at is None
    with pytest.raises(FrozenInstanceError):
        context.deadline_at = _DEADLINE
    with pytest.raises(ValueError, match="timezone-aware"):
        AgentContext(None, Principal("user", "tenant"), "deadline", None, "execution", {},
                     deadline_at=datetime(2099, 1, 1))


@pytest.mark.asyncio
async def test_deadline_inherits_across_retry_fork_and_graph_without_worker_reads(monkeypatch) -> None:
    contexts: list[AgentContext[None]] = []
    storage = PersistenceStorage.in_memory()
    async with Persistence.open('context-deadline', storage=storage) as persistence_59, Runtime.attach(persistence=persistence_59, features=model_features(RuntimeUsageModels(), _group(contexts))) as runtime, TaskEngine.open(persistence=persistence_59, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        reads: list[str] = []
        read = runtime._state.execution.budgets.read

        async def counted_read(scope_id: str):
            reads.append(scope_id)
            return await read(scope_id)

        monkeypatch.setattr(runtime._state.execution.budgets, "read", counted_read)
        backend = runtime._execution_service.runtime_backend()
        launch = backend.launch
        launch_reads: dict[str, int] = {}

        async def recorded_launch(request, execution, **kwargs):
            launch_reads[execution.execution_id] = len(reads)
            await launch(request, execution, **kwargs)

        monkeypatch.setattr(backend, "launch", recorded_launch)

        async def completed(run, expected):
            assert (await run.wait()).result.status is ExecutionStatus.SUCCEEDED
            assert contexts[-1].execution_id == run.execution_id
            assert contexts[-1].deadline_at == expected
            assert len(reads) == launch_reads[run.execution_id]

        root = await runtime.agents.get().start("root", budget=RunBudget(deadline_at=_DEADLINE))
        await completed(root, _DEADLINE)
        agent = runtime.agents.get()
        compiled = runtime._compiled_agent(agent.id, agent.revision, agent.compiled)
        binding = runtime._compiler.bind_subagent(compiled)
        child = await runtime._execution_service.start_subagent(
            binding.binding_digest,
            ExecutionRequest("child", runtime.default_principal, "child", None, "run", False, False),
            parent_execution_id=root.execution_id, root_execution_id=root.execution_id,
            parent_invocation_id="child", binding_contract=binding.binding_contract,
        )
        await completed(await runtime.executions.get(child.execution_id), _DEADLINE)
        await completed(await root.retry("retry"), _DEADLINE)
        await completed(await root.fork("fork"), _DEADLINE)
        await completed(await root.fork("fresh", budget=RunBudget(model_requests=2)), None)
        await completed(await runtime.agents.get().start("unlimited"), None)
        task = Task.from_agent("deadline.agent", runtime.agents.get())
        graph = TaskGraph("deadline-graph", (TaskNode("agent", task=task, input=AgentTaskInput("graph")),))
        run = await runtime_engine.with_definitions(task).start(graph, idempotency_key="graph", budget=RunBudget(deadline_at=_DEADLINE))
        assert (await run.wait()).result.status is TaskStatus.SUCCEEDED
        execution = await run.execution("agent")
        assert next(ctx for ctx in contexts if ctx.execution_id == execution.execution_id).deadline_at == _DEADLINE


@pytest.mark.asyncio
async def test_cold_recovery_restores_absolute_deadline_once(tmp_path: Path, monkeypatch) -> None:
    contexts: list[AgentContext[None]] = []
    features = _group(contexts)
    features.capability(CheckpointCapability(), scope="runtime")
    async with Persistence.open('deadline-recovery', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_114, Runtime.attach(persistence=persistence_114, features=model_features(RuntimeUsageModels(), features)) as runtime:
        async def interrupted_launch(*args, **kwargs):
            pass
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            run = await runtime.agents.get().start("recover", budget=RunBudget(deadline_at=_DEADLINE))
        execution_id = run.execution_id
    assert not contexts
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('deadline-recovery', storage=storage) as persistence_125, Runtime.attach(persistence=persistence_125, features=model_features(RuntimeUsageModels(), features), auto_recover=False) as runtime:
        reads: list[str] = []
        read = runtime._state.execution.budgets.read

        async def counted_read(scope_id: str):
            reads.append(scope_id)
            return await read(scope_id)

        monkeypatch.setattr(runtime._state.execution.budgets, "read", counted_read)
        await runtime.executions.recover(execution_id, principal=runtime.default_principal,
                                         idempotency_key="recover")
        result = (await runtime.executions.wait(execution_id, principal=runtime.default_principal)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert [context.deadline_at for context in contexts] == [_DEADLINE]
        assert reads == ["execution:" + execution_id]
