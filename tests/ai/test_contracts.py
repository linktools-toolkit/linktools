#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent, Task, Observe, and Temporal boundary contracts."""

import warnings
from dataclasses import fields
from datetime import datetime, timezone

import pytest
from linktools.ai.asset import AssetRef
from linktools.ai.core import (
    OperationLedgerRecord,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.observe import (
    InMemoryTraceRecorder,
    MiddlewarePipeline,
    RecordedTraceItem,
    RunContext,
    RunSnapshot,
    snapshot_digest,
)
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state import (
    RuntimeStatePlan,
)
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
)
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.spec import AgentCapabilityRef, AgentSpec
from linktools.ai.storage import InMemoryObjectStore, StoredPayload
from linktools.ai.task import TaskGraph, TaskGraphLimits, TaskLease, TaskNode
from linktools.ai.temporal import (
    ActivityType,
    EvaluationActivity,
    ExecuteActivity,
    SessionActivity,
    TaskActivity,
    WorkerActivities,
    WorkerRegistration,
    WorkflowGateway,
    WorkflowType,
    production_registration,
)
from linktools.ai.temporal.workflow import (
    EvaluationWorkflowInput,
    EvaluationWorkflowResult,
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
    SessionWorkflow,
    SessionWorkflowInput,
    SessionWorkflowResult,
    TaskWorkflowInput,
)
from linktools.ai.workspace import trusted_workspace_principal
from scripts.build.agent_bundle import build_bundle


def test_task_graph_rejects_cycles_and_agent_bundle_is_deterministic() -> None:
    with pytest.raises(ValueError):
        TaskGraph("cycle", (TaskNode("a", ("b",)), TaskNode("b", ("a",))))
    spec = AgentSpec(
        "agent",
        1,
        "route",
        (AgentCapabilityRef("tool", "bash"),),
        "text",
        1,
        "system",
        ("answer",),
    )
    assert build_bundle(spec, "capabilities").digest == build_bundle(spec, "capabilities").digest


def test_model_registry_snapshot_is_instance_owned() -> None:
    registry = ModelRegistry()
    registry.register_openai("route", model="model")
    snapshot = registry.snapshot()
    assert snapshot.resolve("route").model_identity == "openai:model"
    assert snapshot.resolve("route").route_id == "route"


@pytest.mark.parametrize("value", ["", " spaced", "spaced ", "line\nbreak", "control\u0085value", "界" * 129])
def test_classification_fields_reject_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        AssetRef(value, "asset")
    with pytest.raises(ValueError):
        AgentCapabilityRef(value, "capability")
def test_runtime_state_plan_rejects_an_invalid_domain() -> None:
    with pytest.raises(ValueError):
        RuntimeStatePlan(conversation="invalid")


def test_recovery_checkpoint_enforces_attempt_sequence_invariants() -> None:
    now = datetime.now(timezone.utc)
    recovery_input = RecoveryExecutionInput(
        user_prompt="prompt",
        principal_id="principal",
        principal_kind="user",
        session_id=None,
        memory_scope=None,
        agent_id="default",
        binding_digest="binding",
        lineage_kind="RUN",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
    )

    def checkpoint(
        state: RecoveryCheckpointState,
        sequence: int,
        step_run_id: str | None,
        pending_operation_id: str | None = None,
    ) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(
            execution_id="execution",
            tenant_id="tenant",
            input=recovery_input,
            step_run_id=step_run_id,
            agent_run_sequence=sequence,
            state=state,
            handoff_phase=RecoveryHandoffPhase.NONE,
            terminal_handoff=None,
            handoff_contract_digest=None,
            pending_operation_id=pending_operation_id,
            revision=0,
            created_at=now,
            updated_at=now,
        )

    with pytest.raises(ValueError):
        checkpoint(RecoveryCheckpointState.ADMITTED, 1, None)
    with pytest.raises(ValueError):
        checkpoint(RecoveryCheckpointState.ADMITTED, 0, None, "operation")
    with pytest.raises(ValueError):
        checkpoint(RecoveryCheckpointState.ACTIVE, 0, None)
    assert checkpoint(RecoveryCheckpointState.COMPLETED, 0, None).agent_run_sequence == 0


def test_domain_codec_preserves_mapping_payloads_in_nullable_json_values() -> None:
    payload = StoredPayload.inline_json({"text": "你好！"})

    assert decode_domain(encode_domain(payload), StoredPayload) == payload


def test_domain_codec_rejects_mapping_as_a_list() -> None:
    with pytest.raises(TypeError):
        decode_domain({"value": "not-a-list"}, list[str])


def test_subagent_tool_schema_accepts_json_payload() -> None:
    from linktools.ai.agent._capabilities import _SubagentCapability

    async def delegate(**_kwargs: str) -> dict[str, object]:
        return {"execution_id": "child", "status": "SUCCEEDED", "output": {"value": True}}

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _SubagentCapability(delegate).get_toolset()

    assert not any("Could not generate return schema" in str(item.message) for item in captured)


def test_authorization_kinds_are_canonical() -> None:
    assert Principal("principal", "tenant", "service").kind == PrincipalKind.SERVICE.value
    assert Principal("principal", "tenant", "custom").kind == "custom"
    assert ResourceRef(ResourceKind.EXECUTION.value, "execution", "tenant").kind is ResourceKind.EXECUTION
    with pytest.raises(ValueError, match="principal identity is invalid"):
        Principal("principal", "tenant", " custom")
    with pytest.raises(ValueError, match="resource reference is incomplete"):
        ResourceRef("custom", "resource", "tenant")


def test_contextual_classification_fields_stay_concise() -> None:
    operation_fields = {field.name for field in fields(OperationLedgerRecord)}
    task_lease_fields = {field.name for field in fields(TaskLease)}
    tool_operation_fields = {field.name for field in fields(ToolOperationRecord)}
    assert "operation_kind" in operation_fields and "kind" not in operation_fields
    assert "owner" in task_lease_fields and "lease_owner" not in task_lease_fields
    assert "owner" in tool_operation_fields and "lease_owner" not in tool_operation_fields


@pytest.mark.parametrize("value", [" memory", "memory ", "memory\nvalue", "界" * 129])
def test_memory_scope_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(AIError) as error:
        ExecutionRequest("prompt", trusted_workspace_principal("workspace"), "request", value)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


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
    with pytest.raises(ValueError):
        ledger.complete("task", " owner", 2, "digest")


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
    item = RecordedTraceItem("execution", 1, "completed", datetime.now(timezone.utc), "done")
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
        async def prepare(self, request, node_id, dependency_results):
            raise AssertionError("not called")

        async def renew(self, lease):
            raise AssertionError("not called")

        async def settle(self, request, lease, result):
            raise AssertionError("not called")

    class EvaluationOperation:
        async def execute(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
            return EvaluationWorkflowResult(request.evaluation_id, "SUCCEEDED", request.case_ids)

    activities = WorkerActivities(
        execution=ExecuteActivity(ExecutionOperation()),
        session=SessionActivity(SessionOperation()),
        task=TaskActivity(TaskOperation()),
        evaluation=EvaluationActivity(EvaluationOperation()),
    )
    registration = production_registration(activities, build_id="test-build")
    assert isinstance(registration, WorkerRegistration)
    assert len(registration.workflows) == len(registration.activities) == 4
    assert tuple(item.__name__ for item in registration.workflows) == ("ExecutionWorkflow", "SessionWorkflow", "TaskWorkflow", "EvaluationWorkflow")

    assert tuple(type(item).__name__ for item in registration.activities) == ("ExecuteActivity", "SessionActivity", "TaskActivity", "EvaluationActivity")
    assert ExecuteActivity.run.__name__ == "run"
    assert ExecuteActivity.load_input.__name__ == "load_input"
    assert ExecuteActivity.fix_bundle_route.__name__ == "fix_bundle_route"
    assert ExecuteActivity.fix_binding.__name__ == "fix_binding"
    assert ExecuteActivity.reserve_budget.__name__ == "reserve_budget"
    assert ExecuteActivity.run_agent.__name__ == "run_agent"
    assert ExecuteActivity.process_deferred.__name__ == "process_deferred"
    assert ExecuteActivity.commit_result.__name__ == "commit_result"
    assert ExecuteActivity.settle_budget.__name__ == "settle_budget"
    assert EvaluationActivity.run.__name__ == "run"
    assert SessionActivity.run.__name__ == "run"
    assert TaskActivity.prepare.__name__ == "prepare"
    assert TaskActivity.renew.__name__ == "renew"
    assert TaskActivity.settle.__name__ == "settle"

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
    assert worker.configuration == ("json", "asset", "linktools-ai", "test-build", "linktools-ai-production")
    assert len(worker.workflows) == 4


@pytest.mark.asyncio
async def test_temporal_workflow_inputs_apply_canonical_tenant_validation() -> None:
    with pytest.raises(ValueError, match="task workflow graph is invalid"):
        TaskWorkflowInput("graph", " tenant", (), TaskGraphLimits(), "request", "worker")

    with pytest.raises(ValueError, match="execution workflow tenant is invalid"):
        await ExecutionWorkflow().run(
            ExecutionWorkflowInput("execution", "tenant ", "binding", "bundle", "request", "worker")
        )

    with pytest.raises(ValueError, match="session workflow tenant is invalid"):
        await SessionWorkflow().run(SessionWorkflowInput("session", "tenant\n", 0, "operation", "create"))


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

    gateway = WorkflowGateway(
        Client(),
        worker_build="test-build",
        request_store=InMemoryObjectStore("test-requests"),
        namespace="test-namespace",
    )
    local = ExecutionRequest("prompt", trusted_workspace_principal("workspace"), idempotency_key="contract-key", memory_scope="test")
    await gateway.start_execution("execution", local)
    with pytest.raises(ValueError):
        await gateway.query_execution("execution", "unknown")
