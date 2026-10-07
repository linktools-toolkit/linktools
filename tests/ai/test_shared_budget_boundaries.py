#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared budgets gate dispatched model and tool effects, including compaction."""

import asyncio
from dataclasses import replace

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from linktools.ai.core import RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._budget import RunBudgetContext, RunBudgetCapability
from linktools.ai.runtime._compaction import _ObservedCompactionModel
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from ._runtime_test_helpers import tool_run_context, tool_with_metadata


@pytest_asyncio.fixture
async def budgets():
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="budget-boundaries", tenant_id="tenant")
    try:
        yield storage.execution.budgets
    finally:
        await storage.close()


class _Bridge:
    def __init__(self, *, cached: bool = False) -> None:
        self.decision = ToolOperationDecision("operation", "owner", 1, False,
                                              "cached", cached)
        self.deferred = []
        self.completed = []

    async def begin(self, *args):
        return self.decision

    async def defer(self, decision):
        self.deferred.append(decision)
        return False

    async def renew(self, decision):
        return decision

    async def complete(self, decision, result):
        self.completed.append((decision, result))
        return False

    async def unknown(self, decision, error):
        raise AssertionError("no unknown effect may be recorded before dispatch") from error


def _boundary(budget, calls, *, bridge=None):
    async def perform() -> str:
        calls.append("called")
        return "done"

    descriptor = ManagedToolDescriptor(
        "none" if bridge is None else "tool_operation",
        "none" if bridge is None else "non_replay_safe", "business",
    )
    raw = FunctionToolset([tool_with_metadata(perform, descriptor)], id="raw")
    return BoundaryToolset((raw,), {"perform": descriptor}, id="test",
                           budget=budget, tool_operations=bridge)


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_denied_tool_never_dispatches_or_marks_effect_unknown(budgets, managed):
    await budgets.ensure("scope", RunBudget(tool_calls=0))
    budget = RunBudgetContext(budgets, "scope", "execution", "run")
    calls = []
    bridge = _Bridge() if managed else None
    boundary = _boundary(budget, calls, bridge=bridge)
    ctx = tool_run_context()
    tools = await boundary.get_tools(ctx)
    with pytest.raises(AIError) as raised:
        await boundary.call_tool("perform", {}, ctx, tools["perform"])
    assert raised.value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    assert calls == []
    assert (await budgets.read("scope")).tool_calls == 0
    if managed:
        assert bridge.deferred == [bridge.decision]
        assert bridge.completed == []


@pytest.mark.asyncio
async def test_cached_tool_result_does_not_consume_or_recheck_budget(budgets):
    await budgets.ensure("scope", RunBudget(tool_calls=0))
    bridge = _Bridge(cached=True)
    calls = []
    boundary = _boundary(RunBudgetContext(budgets, "scope", "execution", "run"),
                         calls, bridge=bridge)
    ctx = tool_run_context()
    tools = await boundary.get_tools(ctx)
    assert await boundary.call_tool("perform", {}, ctx, tools["perform"]) == "cached"
    assert (await budgets.read("scope")).tool_calls == 0
    assert calls == []


@pytest.mark.asyncio
async def test_new_tool_fence_counts_another_actual_dispatch(budgets):
    await budgets.ensure("scope", RunBudget(tool_calls=2))
    bridge = _Bridge()
    calls = []
    boundary = _boundary(RunBudgetContext(budgets, "scope", "execution", "run"),
                         calls, bridge=bridge)
    ctx = tool_run_context()
    tools = await boundary.get_tools(ctx)
    assert await boundary.call_tool("perform", {}, ctx, tools["perform"]) == "done"
    bridge.decision = replace(bridge.decision, fence=2)
    assert await boundary.call_tool("perform", {}, ctx, tools["perform"]) == "done"
    assert calls == ["called", "called"]
    assert (await budgets.read("scope")).tool_calls == 2
    with pytest.raises(AIError):
        await boundary.call_tool("perform", {}, ctx, tools["perform"])
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_capability_tool_budget_works_without_metrics(budgets):
    await budgets.ensure("scope", RunBudget(tool_calls=1))
    capability = RunBudgetCapability(RunBudgetContext(budgets, "scope", "execution", "run"))
    calls = []

    async def handler(args):
        calls.append(args)
        return "done"

    assert await capability.wrap_tool_execute(
        tool_run_context(), call=ToolCallPart("capability", {}, "first"),
        tool_def=ToolDefinition(name="capability"), args={}, handler=handler,
    ) == "done"
    with pytest.raises(AIError):
        await capability.wrap_tool_execute(
            tool_run_context(), call=ToolCallPart("capability", {}, "second"),
            tool_def=ToolDefinition(name="capability"), args={}, handler=handler,
        )
    assert calls == [{}]


@pytest.mark.asyncio
async def test_model_settlement_uses_own_tokens_without_adding_cache_counters(budgets):
    await budgets.ensure("scope", RunBudget(total_tokens=100))
    budget = RunBudgetContext(budgets, "scope", "execution", "run")

    async def handler():
        return ModelResponse([TextPart("done")], usage=RequestUsage(
            input_tokens=12, output_tokens=4, cache_read_tokens=7, cache_write_tokens=2,
        ))

    await budget.run_model("request", handler)
    usage = await budgets.read("scope")
    assert usage.total_tokens == 16
    assert usage.model_requests == 1
    assert usage.in_flight_model_requests == usage.unknown_model_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_model_marks_unknown_and_prevents_next_token_budget_effect(budgets, cancel):
    await budgets.ensure("scope", RunBudget(total_tokens=100))
    budget = RunBudgetContext(budgets, "scope", "execution", "run")

    async def handler():
        if cancel:
            raise asyncio.CancelledError
        raise RuntimeError("provider failed without usage")

    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await budget.run_model("request", handler)
    usage = await budgets.read("scope")
    assert usage.total_tokens == 0  # Known subtotal; the unknown counter prevents a zero-usage claim.
    assert usage.in_flight_model_requests == 0
    assert usage.unknown_model_requests == 1
    with pytest.raises(AIError) as raised:
        await budget.admit_tool("later")
    assert raised.value.safe_details["reason"] == "unknown_usage"


@pytest.mark.asyncio
async def test_compaction_request_is_budgeted_before_provider_dispatch(budgets):
    await budgets.ensure("scope", RunBudget(model_requests=0))
    ctx = tool_run_context()
    journal = ModelRequestJournal(source_namespace="budget-boundaries", tenant_id="tenant",
                                  execution_id="execution", agent_run_id="run")
    observed = []

    async def observer(ctx, fact, phase, model, response, error):
        observed.append((phase, fact.purpose))

    model = _ObservedCompactionModel(
        TestModel(), ctx=ctx, journal=journal, observer=observer, recorder=None,
        source_messages=(), budget=RunBudgetContext(budgets, "scope", "execution", "run"),
    )
    with pytest.raises(AIError) as raised:
        await model.request([], None, ModelRequestParameters())
    assert raised.value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    assert observed == [("started", "compaction"), ("failed", "compaction")]
    assert (await budgets.read("scope")).model_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_policy", ["none", "non_replay_safe"])
async def test_expired_graph_budget_blocks_callable_without_unknown_effect(effect_policy: str) -> None:
    from datetime import datetime, timedelta, timezone
    from linktools.ai.core import TaskStatus
    from linktools.ai.runtime import Runtime
    from linktools.ai.task import Task, TaskGraph, TaskNode
    from ._runtime_test_helpers import RuntimeUsageModels

    calls = []

    async def handler(context):
        calls.append(context.node_id)
        return "done"

    async with Runtime.open("budget-deadline", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        task = Task("deadline.task", handler, effect_policy=effect_policy)
        engine = runtime.tasks.bind(task)
        run = await engine.start(
            TaskGraph("expired", (TaskNode("node", task=task),)),
            idempotency_key="expired",
            budget=RunBudget(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        )
        result = (await run.wait()).result
        assert result.status is TaskStatus.FAILED
        execution = await run.execution("node")
        terminal = (await execution.wait()).result
        assert terminal.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        assert terminal.safe_error_details["reason"] == "deadline_exceeded"
        assert calls == []


@pytest.mark.asyncio
async def test_standalone_graph_service_rejects_unowned_budget_before_admission() -> None:
    from linktools.ai.core import Principal, TenantAuthorizationPolicy
    from linktools.ai.task import (
        DefaultTaskGraphService, TaskGraph, TaskGraphAdmission, TaskGraphRequest,
        TaskGraphSubmission, TaskNode,
    )
    from .test_task_submission_fence import _Launcher

    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="budget-owner", tenant_id="tenant")
    try:
        launcher = _Launcher()
        service = DefaultTaskGraphService(storage.task, TenantAuthorizationPolicy(), launcher)
        request = TaskGraphRequest(TaskGraph("graph", (TaskNode("node", input={}),)), Principal("user", "tenant"),
                                   "start", budget=RunBudget(model_requests=0))
        submission = TaskGraphSubmission("budget-owner", TaskGraphAdmission.from_request(request), request.graph)
        for operation in (
            lambda: service.start(request),
            lambda: service.prepare_described(submission),
            lambda: service.start_prepared(submission),
        ):
            with pytest.raises(AIError) as rejected:
                await operation()
            assert rejected.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert rejected.value.safe_details["reason"] == "run_budget_owner_missing"
        assert await storage.task.admissions.get("graph", tenant_id="tenant") is None
        assert await storage.task.admissions.submission_status(submission.ref) is None
        assert launcher.started == []

        # A durable admission from another composition is not permission to run unbudgeted.
        await storage.task.admissions.prepare(submission)
        await storage.task.admissions.admit_prepared(submission)
        with pytest.raises(AIError) as rejected:
            await service.recover_pending()
        assert rejected.value.safe_details["reason"] == "run_budget_owner_missing"
        assert launcher.started == []
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension", ["deadline", "tokens"])
async def test_custom_runner_is_gated_before_dispatch(dimension: str) -> None:
    from datetime import datetime, timedelta, timezone
    from linktools.ai.core import TaskStatus
    from linktools.ai.runtime import Runtime
    from linktools.ai.task import Task, TaskGraph, TaskNode
    from ._runtime_test_helpers import RuntimeUsageModels
    from .test_task_mixed_node_reliability import _ExecutionBackedContractRunner

    async with Runtime.open("runner-budget", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        runner = _ExecutionBackedContractRunner(runtime._execution_service)
        task = Task.from_runner("budget.runner", runner, contract={
            "version": 1, "type": "budget.runner", "effect_policy": "none",
            "output_contract": {"kind": "json"}, "reconcile": False,
        })
        budget = (RunBudget(total_tokens=0) if dimension == "tokens" else
                  RunBudget(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        run = await runtime.tasks.bind(task).start(
            TaskGraph("graph", (TaskNode("node", task=task),)),
            idempotency_key="runner", budget=budget,
        )
        assert (await run.wait()).result.status is TaskStatus.FAILED
        assert runner.calls == 0
        assert (await run.budget_usage()).model_requests == 0


@pytest.mark.asyncio
async def test_failed_budget_preparation_does_not_leak_scope_into_unbudgeted_graph(monkeypatch) -> None:
    from linktools.ai.core import TaskStatus
    from linktools.ai.runtime import Runtime
    from linktools.ai.task import Task, TaskGraph, TaskNode
    from ._runtime_test_helpers import RuntimeUsageModels

    calls = []

    async def handler(context):
        calls.append(context.node_id)
        return "done"

    async with Runtime.open("budget-prepare", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        task = Task("prepare.task", handler, effect_policy="none")
        engine = runtime.tasks.bind(task)
        graph = TaskGraph("same", (TaskNode("node", task=task),))
        store = runtime._task_node_runtime._binding_capture_store
        capture = store.capture

        async def fail_capture(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        monkeypatch.setattr(store, "capture", fail_capture)
        with pytest.raises(AIError):
            await engine.start(graph, idempotency_key="failed", budget=RunBudget(total_tokens=0))
        monkeypatch.setattr(store, "capture", capture)
        run = await engine.start(graph, idempotency_key="unbudgeted")
        assert (await run.wait()).result.status is TaskStatus.SUCCEEDED
        assert await run.budget_usage() is None
        assert await (await run.execution("node")).budget_usage() is None
        assert calls == ["node"]
