#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared budgets keep their scope across public execution and graph APIs."""

from pathlib import Path

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue, Principal, RunBudget, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
from ._runtime_test_helpers import RuntimeUsageModels


def _capabilities() -> CapabilityGroup:
    group = CapabilityGroup("budgets")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
async def test_execution_retries_and_default_forks_share_scope_but_explicit_fork_does_not() -> None:
    async with Runtime.open("budget-lineage", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory(), capabilities=(_capabilities(),)) as runtime:
        agent = runtime.agents.get()
        root = await agent.start("root", budget=RunBudget(model_requests=10), idempotency_key="budget-root")
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        retry = await root.retry("retry")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        fork = await root.fork("fork")
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED
        usage = await root.budget_usage()
        assert usage is not None
        assert usage.scope_id == "execution:" + root.execution_id
        assert usage.model_requests == 3
        assert await retry.budget_usage() == usage
        assert await fork.budget_usage() == usage

        fresh = await root.fork("fresh", budget=RunBudget(model_requests=1))
        assert (await fresh.wait()).result.status is ExecutionStatus.SUCCEEDED
        fresh_usage = await fresh.budget_usage()
        assert fresh_usage is not None
        assert fresh_usage.scope_id == "execution:" + fresh.execution_id
        assert fresh_usage.model_requests == 1
        assert (await root.budget_usage()).model_requests == 3
        with pytest.raises(AIError) as changed:
            await agent.start("root", budget=RunBudget(model_requests=11), idempotency_key="budget-root")
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError) as denied:
            await runtime.executions.budget_usage(root.execution_id, principal=Principal("stranger", "other-tenant"))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_session_new_turns_have_independent_budgets() -> None:
    async with Runtime.open("budget-session", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory(), capabilities=(_capabilities(),)) as runtime:
        session = runtime.agents.get().session("conversation")
        first = await session.start("first", budget=RunBudget(model_requests=1))
        assert (await first.wait()).result.status is ExecutionStatus.SUCCEEDED
        second = await session.start("second", budget=RunBudget(model_requests=1))
        assert (await second.wait()).result.status is ExecutionStatus.SUCCEEDED
        first_usage, second_usage = await first.budget_usage(), await second.budget_usage()
        assert first_usage is not None and second_usage is not None
        assert first_usage.scope_id != second_usage.scope_id
        assert first_usage.model_requests == second_usage.model_requests == 1
        third = await session.start("unlimited")
        assert (await third.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert await third.budget_usage() is None


@pytest.mark.asyncio
async def test_graph_budget_is_read_only_during_description_and_shared_by_mixed_nodes(tmp_path: Path) -> None:
    async def ordinary(context: TaskNodeContext[None]) -> JsonValue:
        return context.node_id

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("budget-graph", models=RuntimeUsageModels(), storage=storage,
                            capabilities=(_capabilities(),)) as runtime:
        agent_task = runtime.tasks.from_agent("budget.agent", runtime.agents.get())
        callable_task = Task("budget.callable", ordinary, effect_policy="none")
        engine = runtime.tasks.bind(agent_task, callable_task)
        graph = TaskGraph("shared-graph", (
            TaskNode("agent", task=agent_task, input=AgentTaskInput("hello")),
            TaskNode("callable", task=callable_task),
        ))
        budget = RunBudget(model_requests=4)
        submission = await engine.describe_submission(graph, idempotency_key="budget-graph", budget=budget)
        assert submission.admission.budget == budget
        assert submission.admission.budget_scope_id == "graph:shared-graph"
        with pytest.raises(AIError) as absent:
            await storage.execution.budgets.read("graph:shared-graph")
        assert absent.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        run = await engine.start(graph, idempotency_key="budget-graph", budget=budget)
        assert (await run.wait()).result.status is TaskStatus.SUCCEEDED
        usage = await run.budget_usage()
        assert usage is not None and usage.scope_id == "graph:shared-graph"
        assert usage.model_requests == 1
        for node_id in ("agent", "callable"):
            assert await (await run.execution(node_id)).budget_usage() == usage
        with pytest.raises(AIError) as changed:
            await engine.start(graph, idempotency_key="budget-graph", budget=RunBudget(model_requests=5))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError) as invalid:
            await engine.describe_submission(graph, idempotency_key="invalid-budget", budget={"model_requests": 1})
        assert invalid.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_zero_budget_stops_model_before_output_and_subagent_inherits_spent_scope() -> None:
    from linktools.ai.runtime import ExecutionRequest

    group = _capabilities()
    group.agent("child", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("budget-child", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory(), capabilities=(group,)) as runtime:
        denied = await runtime.agents.get().start("blocked", budget=RunBudget(model_requests=0))
        result = (await denied.wait()).result
        assert result.status is ExecutionStatus.FAILED
        assert result.output is None
        assert result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        assert (await denied.budget_usage()).model_requests == 0

        root = await runtime.agents.get().start("parent", budget=RunBudget(model_requests=1))
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        child_agent = runtime.agents.get("child")
        compiled = runtime._compiled_agent(child_agent.id, child_agent.revision, child_agent.compiled)
        binding = runtime._compiler.bind_subagent(compiled)
        child_handle = await runtime._execution_service.start_subagent(
            binding.binding_digest,
            ExecutionRequest("child", runtime.default_principal, "budget-child", None, "run", False, False),
            parent_execution_id=root.execution_id, root_execution_id=root.execution_id,
            parent_invocation_id="child-invocation", binding_contract=binding.binding_contract,
        )
        child = await runtime.executions.get(child_handle.execution_id)
        child_result = (await child.wait()).result
        assert child_result.status is ExecutionStatus.FAILED
        assert child_result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        assert await child.budget_usage() == await root.budget_usage()
        assert (await root.budget_usage()).model_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("expanded", (False, True))
async def test_graph_siblings_and_expanded_nodes_share_actual_model_admission(expanded: bool) -> None:
    from linktools.ai.task import TaskExpander, TaskExpansionContext

    async with Runtime.open("budget-concurrency", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory(), capabilities=(_capabilities(),)) as runtime:
        task = runtime.tasks.from_agent("budget.concurrent", runtime.agents.get())

        def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
            return (TaskNode("expanded", task=task, input=AgentTaskInput("second")),)

        expander = TaskExpander("budget.expand", expand)
        graph = TaskGraph("contended", (
            (TaskNode("first", task=task, input=AgentTaskInput("first"), expander=expander.ref),)
            if expanded else (
                TaskNode("first", task=task, input=AgentTaskInput("first")),
                TaskNode("second", task=task, input=AgentTaskInput("second")),
            )
        ))
        engine = runtime.tasks.bind(task, expander)
        run = await engine.start(graph, idempotency_key="budget-contended", budget=RunBudget(model_requests=1))
        assert (await run.wait()).result.status is TaskStatus.FAILED
        usage = await run.budget_usage()
        assert usage is not None and usage.model_requests == 1
        state = await run.state()
        statuses = [node.status for node in state.node_states]
        assert statuses.count(TaskStatus.SUCCEEDED) == 1
        assert statuses.count(TaskStatus.FAILED) == 1
        for node in state.node_states:
            if node.execution_id is not None:
                assert await (await run.execution(node.node_id)).budget_usage() == usage


@pytest.mark.asyncio
async def test_budget_scope_survives_restart_and_missing_inherited_scope_fails_closed(tmp_path: Path) -> None:
    from linktools.ai.runtime.state._store import RecordQuery, StateTransaction

    async with Runtime.open("budget-reopen", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(_capabilities(),)) as runtime:
        root = await runtime.agents.get().start("first", budget=RunBudget(model_requests=2))
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = root.execution_id
        before = await root.budget_usage()

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("budget-reopen", models=RuntimeUsageModels(), storage=storage,
                            capabilities=(_capabilities(),)) as runtime:
        restored = await runtime.executions.get(execution_id)
        assert await restored.budget_usage() == before
        retried = await restored.retry("second")
        assert (await retried.wait()).result.status is ExecutionStatus.SUCCEEDED
        usage = await retried.budget_usage()
        assert usage is not None and usage.model_requests == 2
        assert usage.scope_id == before.scope_id

        async def remove_scope(transaction: StateTransaction) -> None:
            records = await transaction.list_records(RecordQuery(kind="budget_scope"))
            assert len(records) == 1
            assert await transaction.delete_record(records[0].key_digest)

        await storage.execution.budgets.state_store.mutate(remove_scope)
        with pytest.raises(AIError) as missing:
            await retried.retry("must not reset")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as missing_fork:
            await retried.fork("must not reset either")
        assert missing_fork.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
