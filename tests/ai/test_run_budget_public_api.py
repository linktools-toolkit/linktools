#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task.engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Shared budgets keep their scope across public execution and graph APIs."""
from ._runtime_test_helpers import model_features

from pathlib import Path

import pytest

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, Principal, RunBudget, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
from ._runtime_test_helpers import RuntimeUsageModels


def _capabilities() -> RuntimeFeatures:
    group = RuntimeFeatures("budgets")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
async def test_execution_retries_and_default_forks_share_scope_but_explicit_fork_does_not() -> None:
    capabilities = _capabilities()
    capabilities.capability(TraceCapability(), scope="runtime")
    features = capabilities
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('budget-lineage', storage=PersistenceStorage.in_memory()) as persistence_31, Runtime.attach(persistence=persistence_31, features=model_features(RuntimeUsageModels(), features)) as runtime:
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
    features = _capabilities()
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('budget-session', storage=PersistenceStorage.in_memory()) as persistence_66, Runtime.attach(persistence=persistence_66, features=model_features(RuntimeUsageModels(), features)) as runtime:
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

    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('budget-graph', storage=storage) as persistence_88, Runtime.attach(persistence=persistence_88, features=model_features(RuntimeUsageModels(), _capabilities())) as runtime, TaskEngine.open(persistence=persistence_88, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        agent_task = Task.from_agent("budget.agent", runtime.agents.get())
        callable_task = Task("budget.callable", ordinary, effect_policy="none")
        engine = runtime_engine.with_definitions(agent_task, callable_task)
        graph = TaskGraph("shared-graph", (
            TaskNode("agent", task=agent_task, input=AgentTaskInput("hello")),
            TaskNode("callable", task=callable_task),
        ))
        budget = RunBudget(model_requests=4)
        submission = await engine.describe_submission(graph, idempotency_key="budget-graph", budget=budget)
        assert submission.admission.budget == budget
        assert submission.admission.budget_scope_id == "graph:shared-graph"
        with pytest.raises(AIError) as absent:
            await runtime._state.execution.budgets.read("graph:shared-graph")
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
    from linktools.ai.execution import ExecutionRequest

    group = _capabilities()
    group.agent("child", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('budget-child', storage=PersistenceStorage.in_memory()) as persistence_125, Runtime.attach(persistence=persistence_125, features=model_features(RuntimeUsageModels(), group)) as runtime:
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

    async with Persistence.open('budget-concurrency', storage=PersistenceStorage.in_memory()) as persistence_158, Runtime.attach(persistence=persistence_158, features=model_features(RuntimeUsageModels(), _capabilities())) as runtime, TaskEngine.open(persistence=persistence_158, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("budget.concurrent", runtime.agents.get())

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
        engine = runtime_engine.with_definitions(task, expander)
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
    from linktools.ai.persistence.state import RecordQuery, StateTransaction

    features = _capabilities()
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('budget-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_193, Runtime.attach(persistence=persistence_193, features=model_features(RuntimeUsageModels(), features)) as runtime:
        root = await runtime.agents.get().start("first", budget=RunBudget(model_requests=2))
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = root.execution_id
        before = await root.budget_usage()

    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('budget-reopen', storage=storage) as persistence_201, Runtime.attach(persistence=persistence_201, features=model_features(RuntimeUsageModels(), features)) as runtime:
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

        await runtime._state.execution.budgets.state_store.mutate(remove_scope)
        with pytest.raises(AIError) as missing:
            await retried.retry("must not reset")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as missing_fork:
            await retried.fork("must not reset either")
        assert missing_fork.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
