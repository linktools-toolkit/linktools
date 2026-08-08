#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent, Task, Observe, and Temporal boundary contracts."""

from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentDeps
from scripts.build.agent_bundle import build_bundle
from linktools.ai.core import ErrorCode, AIError
from linktools.ai.model import ModelRegistry, ModelRoute
from linktools.ai.observe import RunContext
from linktools.ai.observe import MiddlewarePipeline
from linktools.ai.observe import RunSnapshot, snapshot_digest
from linktools.ai.observe import InMemoryTraceRecorder, TraceItem
from linktools.ai.workspace import trusted_workspace_principal
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.spec import AgentFeatureRef, AgentSpec, PromptSpec
from linktools.ai.task import TaskGraph, TaskNode
from linktools.ai.temporal import WorkerActivities, WorkerRegistration, production_registration
from linktools.ai.temporal import WorkflowGateway
from linktools.ai.temporal import ActivityType, EvaluationActivity, ExecuteActivity, SessionActivity, TaskActivity, WorkflowType
from linktools.ai.temporal.workflow import EvaluationWorkflowInput, EvaluationWorkflowResult, ExecutionWorkflowInput, ExecutionWorkflowResult, SessionWorkflowInput, SessionWorkflowResult, TaskWorkflowInput, TaskWorkflowResult


def test_task_graph_rejects_cycles_and_agent_bundle_is_deterministic() -> None:
    with pytest.raises(ValueError):
        TaskGraph("cycle", (TaskNode("a", ("b",)), TaskNode("b", ("a",))))
    spec = AgentSpec("agent", 1, "route", (AgentFeatureRef("tool", "bash"),), "text", 1, ("answer",))
    prompt = PromptSpec("prompt", 1, "system", ("answer",), ())
    assert build_bundle(spec, prompt, "capabilities").digest == build_bundle(spec, prompt, "capabilities").digest


def test_model_registry_and_serializable_agent_deps_are_instance_owned() -> None:
    registry = ModelRegistry()
    snapshot = registry.prime({"route": ModelRoute("route", "openai", "model")})
    assert snapshot.routes["route"].model == "model"
    deps = AgentDeps(
        execution_id="execution",
        tenant_principal_ref="tenant:principal",
        model_plan_id="route",
        budget_id="budget",
        prompt_snapshot_id="prompt",
    )
    assert deps.model_dump()["execution_id"] == "execution"


def test_task_completion_checks_owner_fence_result_and_terminal_state() -> None:
    from linktools.ai.task import TaskCompletionLedger

    ledger = TaskCompletionLedger()
    first = ledger.complete("task", "owner", 1, "digest")
    assert ledger.complete("task", "owner", 1, "digest") == first
    with pytest.raises(AIError) as result_error:
        ledger.complete("task", "owner", 1, "other")
    assert result_error.value.code == ErrorCode.TASK_RESULT_CONFLICT
    with pytest.raises(AIError) as owner_error:
        ledger.complete("task", "other", 1, "digest")
    assert owner_error.value.code == ErrorCode.TASK_OWNER_CONFLICT
    with pytest.raises(AIError) as fence_error:
        ledger.complete("task", "owner", 0, "digest")
    assert fence_error.value.code == ErrorCode.TASK_FENCE_STALE
    with pytest.raises(AIError) as terminal_error:
        ledger.fail("task", "owner", 1, "FAILED", "error")
    assert terminal_error.value.code == ErrorCode.TASK_TERMINAL_CONFLICT


@pytest.mark.asyncio
async def test_middleware_order_and_failure_classification() -> None:
    events: list[str] = []

    class Observer:
        mutating = False

        async def before_run(self, context: RunContext) -> None:
            events.append("before")

        async def before_model(self, context: RunContext) -> None:
            events.append("model")

        async def after_model(self, context: RunContext) -> None:
            events.append("after-model")

        async def before_tool(self, context: RunContext) -> None:
            events.append("tool")

        async def after_tool(self, context: RunContext) -> None:
            events.append("after-tool")

        async def on_error(self, context: RunContext, error: BaseException) -> None:
            events.append("error")

        async def after_run(self, context: RunContext) -> None:
            events.append("after")

    context = RunContext("tenant", "principal", "execution", "session", "run", "agent")
    pipeline = MiddlewarePipeline((Observer(),))
    await pipeline.before_run(context)
    await pipeline.after_run(context)
    assert events == ["before", "after"]

    class FailingObserver(Observer):
        async def before_model(self, context: RunContext) -> None:
            raise RuntimeError("telemetry failure")

    await MiddlewarePipeline((FailingObserver(),)).before_model(context)

    class FailingMutation(FailingObserver):
        mutating = True

    with pytest.raises(AIError) as middleware_error:
        await MiddlewarePipeline((FailingMutation(),)).before_model(context)
    assert middleware_error.value.code == ErrorCode.MIDDLEWARE_FAILED


@pytest.mark.asyncio
async def test_trace_is_monotonic_and_snapshot_digest_is_verified() -> None:
    recorder = InMemoryTraceRecorder()
    item = TraceItem("execution", 1, "completed", datetime.now(timezone.utc), "done")
    assert await recorder.append(item) == item
    assert await recorder.append(item) == item
    values = {
        "snapshot_id": "snapshot",
        "execution_id": "execution",
        "binding_digest": "binding",
        "trace_digest": "trace",
        "result_digest": None,
    }
    snapshot = RunSnapshot(**values, digest=snapshot_digest(values))
    assert snapshot.verify()


def test_temporal_registration_has_one_explicit_worker_surface() -> None:
    class ExecutionOperation:
        async def execute(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult:
            return ExecutionWorkflowResult(request.execution_id, "SUCCEEDED", None, 0)

    class SessionOperation:
        async def execute(self, request: SessionWorkflowInput) -> SessionWorkflowResult:
            return SessionWorkflowResult(request.session_id, request.mutation_id, "SUCCEEDED")

    class TaskOperation:
        async def execute(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
            return TaskWorkflowResult(request.graph_id, "SUCCEEDED", request.task_ids)

    class EvaluationOperation:
        async def execute(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
            return EvaluationWorkflowResult(request.evaluation_id, "SUCCEEDED", request.case_ids)

    activities = WorkerActivities(
        execution=ExecuteActivity(ExecutionOperation()),
        session=SessionActivity(SessionOperation()),
        task=TaskActivity(TaskOperation()),
        evaluation=EvaluationActivity(EvaluationOperation()),
    )
    registration = production_registration(activities)
    assert isinstance(registration, WorkerRegistration)
    assert len(registration.workflows) == len(registration.activities) == 4
    assert tuple(item.__name__ for item in registration.workflows) == ("ExecutionWorkflow", "SessionWorkflow", "TaskWorkflow", "EvaluationWorkflow")
    assert tuple(type(item).__name__ for item in registration.activities) == ("ExecuteActivity", "SessionActivity", "TaskActivity", "EvaluationActivity")
    assert ExecuteActivity.run.__name__ == "run"
    assert ExecuteActivity.load_input.__name__ == "load_input"
    assert ExecuteActivity.fix_bundle_route.__name__ == "fix_bundle_route"
    assert ExecuteActivity.fix_binding.__name__ == "fix_binding"
    assert ExecuteActivity.load_prompt.__name__ == "load_prompt"
    assert ExecuteActivity.reserve_budget.__name__ == "reserve_budget"
    assert ExecuteActivity.run_agent.__name__ == "run_agent"
    assert ExecuteActivity.process_deferred.__name__ == "process_deferred"
    assert ExecuteActivity.commit_result.__name__ == "commit_result"
    assert ExecuteActivity.settle_budget.__name__ == "settle_budget"
    assert EvaluationActivity.run.__name__ == "run"
    assert SessionActivity.run.__name__ == "run"
    assert TaskActivity.run.__name__ == "run"

    class Worker:
        def configure(
            self,
            *,
            data_converter: str,
            payload_codec: str,
            interceptor: str,
            build_id: str,
            task_queue: str,
        ) -> None:
            self.configuration = (data_converter, payload_codec, interceptor, build_id, task_queue)

        def register_workflows(self, workflows: tuple[WorkflowType, ...]) -> None:
            self.workflows = workflows

        def register_activities(self, activities: tuple[ActivityType, ...]) -> None:
            self.activities = activities

    worker = Worker()
    registration.register(worker)
    assert worker.configuration == ("json", "asset", "linktools-ai", "linktools-ai", "linktools-ai-production")
    assert len(worker.workflows) == 4


@pytest.mark.asyncio
async def test_workflow_gateway_validates_contract_and_unknown_operations() -> None:
    class Client:
        async def start_workflow(self, workflow: str, request, *, workflow_id: str):
            return None

        async def start_task_graph(self, request, *, workflow_id: str):
            return None

        async def update_workflow(self, workflow_id: str, operation: str, payload):
            return None

        async def query_workflow(self, workflow_id: str, query: str):
            return None

        async def cancel_workflow(self, workflow_id: str):
            return None

    gateway = WorkflowGateway(Client())
    local = ExecutionRequest("prompt", trusted_workspace_principal("workspace"), idempotency_key="contract-key")
    await gateway.start_execution("execution", local)
    with pytest.raises(ValueError):
        await gateway.query_execution("execution", "unknown")
