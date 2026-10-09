#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime policies share selected datasets without transferring resource ownership."""

from pathlib import Path

import pytest

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, JsonValue, Principal, ResourceKind, ResourceRef,
    TenantAuthorizationPolicy, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, StartEvaluationRequest,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import (
    CaptureGraphRequest, Runtime, RuntimeContext, RuntimeStorage,
)
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from .test_evaluation_consumers import (
    EVALUATION_COMPLETION_TIMEOUT_SECONDS, FixtureModels, echo, exact, rule_scorer,
)


TENANT = "shared-evaluation"
PUBLISHER = service_principal(TENANT, "system-publisher")
OPERATOR = service_principal(TENANT, "operator")
CONTEXT = RuntimeContext(None, tenant_id=TENANT)


class SharedDatasetAuthorization:
    """Application-owned grants scoped to the exact dataset and capture identities."""

    def __init__(self) -> None:
        self.default = TenantAuthorizationPolicy(TENANT)
        self.datasets: set[str] = set()
        self.input_captures: set[str] = set()
        self.graph_captures: set[str] = set()
        self.denied: set[AuthorizationAction] = set()
        self.calls: list[tuple[Principal, AuthorizationAction, ResourceRef]] = []

    def __bool__(self) -> bool:
        return False

    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        self.calls.append((principal, action, resource))
        if action in self.denied:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if (
            principal == OPERATOR
            and resource.tenant_id == TENANT
            and resource.owner_principal_id == PUBLISHER.principal_id
            and (
                action is AuthorizationAction.EVALUATION_DATASET_READ
                and resource.kind is ResourceKind.EVALUATION
                and resource.id in self.datasets
                or action is AuthorizationAction.EXECUTION_CAPTURE_INPUT
                and resource.kind is ResourceKind.EXECUTION
                and resource.id in self.input_captures
                or action is AuthorizationAction.TASK_CAPTURE_GRAPH
                and resource.kind is ResourceKind.TASK_GRAPH
                and resource.id in self.graph_captures
            )
        ):
            return
        await self.default.authorize(principal, action, resource)


def storage_for(backend: str, path: Path) -> RuntimeStorage:
    if backend == "filesystem":
        return RuntimeStorage.filesystem(path)
    return RuntimeStorage.sqlite(path / "state.sqlite")


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_none", (False, True))
async def test_runtime_default_authorization_keeps_owner_and_local_trusted_denials(
    tmp_path: Path, explicit_none: bool,
) -> None:
    options = {"authorization": None} if explicit_none else {}
    async with Runtime.open(
        "default-policy", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path),
        **options,
    ) as runtime:
        owner = service_principal("default", "owner")
        other = service_principal("default", "other")
        foreign = service_principal("foreign", "owner")
        spec = DatasetSpec(DatasetRef("private", 1), (
            CaseSpec.task(CaseRef("private", "one", 1), input={"answer": "yes"}),
        ))
        dataset = await runtime.evaluations.publish_dataset(
            spec, principal=owner, idempotency_key="publish",
        )
        assert (await runtime.evaluations.get_dataset(dataset, principal=owner)).ref == dataset
        for principal in (other, foreign, runtime.default_principal):
            with pytest.raises(AIError) as denied:
                await runtime.evaluations.get_dataset(dataset, principal=principal)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.publish_dataset(
                spec, principal=runtime.default_principal, idempotency_key="local-publish",
            )
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("input_kind", ("task", "agent"))
async def test_shared_dataset_reopens_with_scoped_policy_and_operator_owned_experiment(
    tmp_path: Path, backend: str, input_kind: str,
) -> None:
    models = FixtureModels()
    group = CapabilityGroup[None]("shared-agent")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    ref = DatasetRef("system-data", 1)
    case_ref = CaseRef(ref.id, "one", 1)
    spec = DatasetSpec(ref, (
        CaseSpec.task(case_ref, input={"answer": "yes"}, expected="yes")
        if input_kind == "task" else
        CaseSpec.agent(case_ref, prompt="approved system prompt", expected="fixture answer"),
    ))
    private = DatasetSpec(DatasetRef("private-data", 1), (
        CaseSpec.task(CaseRef("private-data", "one", 1), input={"answer": "private"}),
    ))
    async with Runtime.open(
        "shared-data", models=models, storage=storage_for(backend, tmp_path), context=CONTEXT,
        capabilities=(group,),
    ) as runtime:
        await runtime.evaluations.publish_dataset(spec, principal=PUBLISHER, idempotency_key="system")
        await runtime.evaluations.publish_dataset(private, principal=PUBLISHER, idempotency_key="private")
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.get_dataset(ref, principal=OPERATOR)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

    authorization = SharedDatasetAuthorization()
    authorization.datasets.add(ref.id)
    storage = storage_for(backend, tmp_path)
    async with Runtime.open(
        "shared-data", models=models, storage=storage, context=CONTEXT,
        capabilities=(group,), authorization=authorization,
    ) as runtime:
        assert (await runtime.evaluations.get_dataset(ref, principal=OPERATOR)).ref == ref
        cases = await runtime.evaluations.list_cases(ref, principal=OPERATOR)
        assert cases.items[0].ref == case_ref
        for dataset, principal in (
            (private.ref, OPERATOR),
            (ref, service_principal("foreign", OPERATOR.principal_id)),
        ):
            with pytest.raises(AIError) as denied:
                await runtime.evaluations.get_dataset(dataset, principal=principal)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.publish_dataset(spec, principal=OPERATOR, idempotency_key="overwrite")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

        target = (Task("shared.target", echo, effect_policy="none") if input_kind == "task" else
                  runtime.tasks.from_agent("shared.target", runtime.agents.get()))
        scorer = Task("shared.exact", exact, effect_policy="none")
        engine = runtime.tasks.bind(target, scorer)
        request = StartEvaluationRequest(EvaluationSpec(
            ref, (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),),
            policy=EvaluationPolicy(model_fixtures=(models.contract,)),
        ), OPERATOR, "evaluate")
        if input_kind == "agent":
            capture = cases.items[0].input
            assert isinstance(capture, AgentInputCaptureRef)
            with pytest.raises(AIError) as denied:
                await runtime.evaluations.start(request, engine=engine)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert models.prompts == []
            authorization.input_captures.add(capture.capture_id)

        run = await runtime.evaluations.start(request, engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        summary = (await run.create_report()).scores[0]
        assert summary.valid == 1
        if input_kind == "task":
            assert summary.mean == 1.0
        else:
            assert models.prompts == ["approved system prompt"]
        assert await storage.evaluation.records.dataset_owner(ref) == PUBLISHER.principal_id
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=TENANT)
        assert record is not None and record.manifest.principal == OPERATOR
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.get(run.experiment_id, principal=PUBLISHER)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

        trial = (await run.trials()).items[0]
        assert trial.execution_status is ExecutionStatus.SUCCEEDED
        graph = await engine.get(trial.graph_ref.graph_id, principal=OPERATOR)
        execution = await graph.execution("target")
        assert runtime.history is not None
        assert (await runtime.history.inspect_execution(execution.execution_id, principal=OPERATOR)).execution_id == execution.execution_id
        authorization.denied.add(AuthorizationAction.EXECUTION_READ)
        for inspect in (runtime.executions.inspect, runtime.history.inspect_execution):
            with pytest.raises(AIError) as denied:
                await inspect(execution.execution_id, principal=OPERATOR)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        authorization.denied.clear()

    async with Runtime.open(
        "shared-data", models=models, storage=storage_for(backend, tmp_path), context=CONTEXT,
        capabilities=(group,),
    ) as runtime:
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.get_dataset(ref, principal=OPERATOR)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert (await (await runtime.evaluations.get(run.experiment_id, principal=OPERATOR)).inspect()).completion == "complete"


@pytest.mark.asyncio
async def test_shared_graph_capture_requires_its_own_permission(tmp_path: Path) -> None:
    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        return context.input["answer"]

    authorization = SharedDatasetAuthorization()
    authorization.datasets.add("system-graph")
    target = Task("shared.graph-target", answer, effect_policy="none")
    scorer = Task("shared.graph-score", exact, effect_policy="none")
    async with Runtime.open(
        "shared-graph", models=ModelRegistry(), storage=RuntimeStorage.filesystem(tmp_path),
        context=CONTEXT, authorization=authorization,
    ) as runtime:
        engine = runtime.tasks.bind(target, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("target", task=target, input={"answer": "yes"}),
        )), principal=PUBLISHER, idempotency_key="source")
        await source.wait()
        capture = await runtime.tasks.capture_graph(
            "source", CaptureGraphRequest(PUBLISHER, "capture"),
        )
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("system-graph", 1), (
            CaseSpec.graph(CaseRef("system-graph", "one", 1), inputs={}, expected="yes"),
        )), principal=PUBLISHER, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("graph", graph_template=GraphTargetSpec(capture=capture, selector="terminal_sinks")),),
            (rule_scorer(scorer),),
        ), OPERATOR, "evaluate")
        with pytest.raises(AIError) as denied:
            await runtime.evaluations.start(request, engine=engine)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        authorization.graph_captures.add(capture.capture_id)
        run = await runtime.evaluations.start(request, engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        assert any(
            principal == OPERATOR and action is AuthorizationAction.TASK_CAPTURE_GRAPH
            and resource.owner_principal_id == PUBLISHER.principal_id
            for principal, action, resource in authorization.calls
        )
        with pytest.raises(AIError) as denied:
            await runtime.tasks.capture_graph(
                "source", CaptureGraphRequest(OPERATOR, "unapproved-source-capture"),
            )
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
