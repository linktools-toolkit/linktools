#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation launch intents own local, application-provided Runtime scopes."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    ExecutionStatus, JsonValue, OperationKind, OperationLedgerRecord, OperationStatus, Principal,
    TaskStatus, canonical_sha256, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationManifest, EvaluationPolicy, EvaluationSpec, HumanScoreRequest, RescoreRequest,
    ScoreAttemptView, ScoreBundle, ScorerSpec,
    ScoringInput, StartEvaluationRequest, TargetTrialRef,
)
from linktools.ai.migrate import provision_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import (
    EvaluationTrialScope, Runtime, RuntimeContext, RuntimeStorage, TaskEngine,
)
from linktools.ai.storage import FilesystemObjectStore
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._evaluation_records import EvaluationLaunchIntent
from linktools.ai.runtime._evaluation_scope import _EnteredTrialScope
from linktools.ai.task import (
    Task, TaskGraph, TaskGraphState, TaskNode, TaskNodeContext, TaskRef,
    TaskSubmissionCancellation, TaskSubmissionRef,
)
from linktools.ai.workspace import Workspace

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, FixtureModels
from ._runtime_test_helpers import _wait_for_committed


NAMESPACE = "evaluation-trial-scopes"
PRINCIPAL = service_principal("trial-scope-tenant", "admitted-owner")
CONTEXT = RuntimeContext(None, tenant_id=PRINCIPAL.tenant_id)
DIMENSION = DimensionContract("quality", "number", "higher", 0, 1)
_SCOPE_CONTEXT = ContextVar[EvaluationTrialScope | None]("evaluation_trial_scope", default=None)


_UNSCOPED_MANIFEST_WIRE = """
{"$dataclass":"evaluation_manifest","fields":{
  "candidates":{"$tuple":[{"$dataclass":"evaluation_candidate_contract","fields":{
    "definition_contracts":{"$tuple":[]},"graph_template":null,"slot_id":"target",
    "task":{"$dataclass":"task_ref","fields":{"id":"target","revision":1}}}}]},
  "dataset":{"$dataclass":"evaluation_dataset_ref","fields":{"id":"dataset","revision":1}},
  "experiment_id":"experiment","input_mode":"fixed_input","kind":"experiment",
  "policy":{"$dataclass":"evaluation_policy","fields":{
    "allow_volatile":false,"content_retention_seconds":null,"cost_limit":null,"currency":null,
    "environment":{"$mapping":[]},"external_effects":"deny","human_timeout_seconds":86400.0,
    "max_trials":1000,"metadata_retention_seconds":null,"model_fixtures":{"$tuple":[]},
    "model_mode":"fixture_only","price_table":null,"scorer_concurrency":2,
    "scorer_graph_limits":{"$dataclass":"task_graph_limits","fields":{
      "max_budget":1000,"max_concurrency":8,"max_depth":8,"max_nodes":128}},
    "scorer_timeout_seconds":300.0,"target_concurrency":4,
    "target_graph_limits":{"$dataclass":"task_graph_limits","fields":{
      "max_budget":1000,"max_concurrency":8,"max_depth":8,"max_nodes":128}},
    "token_limit":null,"trial_timeout_seconds":300.0,"unknown_usage":"stop"}},
  "principal":{"$dataclass":"principal","fields":{
    "kind":"service","principal_id":"owner","tenant_id":"tenant"}},
  "scorers":{"$tuple":[{"$dataclass":"evaluation_scorer_contract","fields":{
    "accepts_target_failure":false,"accepts_target_kinds":{"$tuple":["execution","graph"]},
    "config":{"$mapping":[]},"dimensions":{"$tuple":[{"$dataclass":"evaluation_dimension_contract",
      "fields":{"direction":"higher","maximum":null,"minimum":null,"name":"quality","unit":"number"}}]},
    "evidence_policy":{"$dataclass":"evaluation_evidence_policy","fields":{
      "include_attachments":false,"include_input":true,"include_output":true,"include_trace":false}},
    "input_projection":{"$mapping":[["kind","task"],["version",1]]},"output_contract":{"$mapping":[]},
    "required":true,"rubric":null,"slot_id":"score",
    "task":{"$dataclass":"task_ref","fields":{"id":"score","revision":1}},"task_contract":{"$mapping":[]}}}]},
  "source_experiment_id":null,"source_trials":{"$tuple":[]},
  "trials":{"$tuple":[{"$dataclass":"evaluation_trial_plan","fields":{
    "candidate_slot_id":"target","case_ref":{"$dataclass":"evaluation_case_ref","fields":{
      "case_id":"case","dataset_id":"dataset","revision":1}},"repetition":1,"trial_id":"trial"}}]}
}}
"""


def test_unscoped_manifest_wire_preserves_identity_and_scoped_mode_changes_it() -> None:
    wire = json.loads(_UNSCOPED_MANIFEST_WIRE)
    unscoped = decode_domain(wire, EvaluationManifest)
    assert not unscoped.trial_scope_required
    assert unscoped.digest == "bba3e1f0b12a4ed3432f9aed120b7faceb73b331817113d0826b0a9bb0b65eec"
    start = unscoped.to_mapping()
    start.pop("experiment_id")
    assert canonical_sha256(start) == "0580399011dde8221ae4856d8187620674736393c9d1ea34270792ca2901923b"

    scoped = replace(unscoped, trial_scope_required=True)
    assert scoped.digest != unscoped.digest
    scoped_start = scoped.to_mapping()
    scoped_start.pop("experiment_id")
    assert canonical_sha256(scoped_start) != canonical_sha256(start)
    assert decode_domain(encode_domain(scoped), EvaluationManifest) == scoped
    assert decode_domain(encode_domain(unscoped), EvaluationManifest) == unscoped

    wire["fields"]["trial_scope_required"] = False
    assert decode_domain(wire, EvaluationManifest) == unscoped
    wire["fields"]["trial_scope_required"] = "false"
    with pytest.raises(AIError) as malformed:
        decode_domain(wire, EvaluationManifest)
    assert malformed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def _until(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)


async def _score(context: TaskNodeContext[Path]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    assert context.principal == PRINCIPAL
    assert sample.target_output == sample.expected == {"answer": "private case input"}
    assert (context.app / "retained.txt").read_text(encoding="utf-8") == "application-owned workspace"
    return ScoreBundle(dimensions={"quality": 1.0}).to_mapping()


def _scorer(task: Task[Path]) -> ScorerSpec:
    return ScorerSpec("quality", task.ref, (DIMENSION,))


async def _request(
    runtime: Runtime[None], tasks: tuple[Task[Path], ...], *, key: str = "start",
    policy: EvaluationPolicy | None = None,
) -> StartEvaluationRequest:
    dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("scoped", 1), (
        CaseSpec.task(CaseRef("scoped", "one", 1), input={"answer": "private case input"},
                      expected={"answer": "private case input"}),
    )), principal=PRINCIPAL, idempotency_key="publish")
    return StartEvaluationRequest(EvaluationSpec(
        dataset, (CandidateSpec("target", task=tasks[0].ref),), (_scorer(tasks[1]),),
        policy=policy or EvaluationPolicy(),
    ), PRINCIPAL, key)


class Scopes:
    def __init__(
        self, root: Path, storage: RuntimeStorage, make_storage: Callable[[], RuntimeStorage],
        tasks: tuple[Task[Path], ...], *, namespace: str = NAMESPACE,
        tenant_id: str = PRINCIPAL.tenant_id,
    ) -> None:
        self.root = root
        self.storage = storage
        self.make_storage = make_storage
        self.tasks = tasks
        self.namespace = namespace
        self.tenant_id = tenant_id
        self.opened: list[EvaluationTrialScope] = []
        self.closed: list[EvaluationTrialScope] = []
        self.workspaces: list[Path] = []
        self.runtimes: list[Runtime[Path]] = []
        self.stores: list[RuntimeStorage] = []

    @asynccontextmanager
    async def __call__(self, scope: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        experiment_id = scope.experiment_id
        assert experiment_id == scope.submission.admission.correlation["evaluation_experiment"]
        record = await self.storage.evaluation.records.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record is not None
        intent = next(item for item in record.intents if item.slot_id == scope.slot_id)
        assert intent.submission == scope.submission
        assert intent.trial == scope.trial
        assert intent.scorer_slot_id == scope.scorer_slot_id
        assert scope.principal == record.manifest.principal == PRINCIPAL
        assert scope.submission.admission.principal == PRINCIPAL
        with pytest.raises(FrozenInstanceError):
            scope.slot_id = "changed"

        workspace = self.root / scope.experiment_id / scope.slot_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "retained.txt").write_text("application-owned workspace", encoding="utf-8")
        child_storage = self.make_storage()
        assert child_storage is not self.storage
        token = _SCOPE_CONTEXT.set(scope)
        try:
            async with Runtime.open(
                self.namespace, models=ModelRegistry(), storage=child_storage,
                context=RuntimeContext(workspace, tenant_id=self.tenant_id),
                capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(workspace)),),
                auto_recover=False,
            ) as child:
                self.opened.append(scope)
                self.workspaces.append(workspace)
                self.runtimes.append(child)
                self.stores.append(child_storage)
                yield child.tasks.bind(*self.tasks)
        finally:
            try:
                if scope in self.opened:
                    self.closed.append(scope)
            finally:
                _SCOPE_CONTEXT.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql"))
async def test_scoped_targets_and_rescores_share_durable_facts_but_own_local_workspaces(
    tmp_path: Path, backend: str,
) -> None:
    calls: list[Path] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        assert context.principal == PRINCIPAL
        assert context.input == {"answer": "private case input"}
        calls.append(context.app)
        assert (context.app / "retained.txt").read_text(encoding="utf-8") == "application-owned workspace"
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))
    sql = None
    disposed: list[bool] = []
    if backend == "sql":
        sql = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.sqlite'}")
        await provision_database(sql)
        event.listen(sql.sync_engine, "engine_disposed", lambda engine: disposed.append(True))

    def make_storage() -> RuntimeStorage:
        if sql is None:
            return RuntimeStorage.sqlite(tmp_path / "state.sqlite")
        return RuntimeStorage.sql(sql, object_store=FilesystemObjectStore(tmp_path / "objects"))

    storage = make_storage()
    try:
        async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
            scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)
            request = await _request(runtime, tasks)
            run = await runtime.evaluations.start(request, engine=runtime.tasks.bind(*tasks), trial_scope=scopes)
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            await _until(lambda: len(scopes.closed) == 2)
            assert len(calls) == 1
            assert [scope.scorer_slot_id for scope in scopes.opened] == [None, "quality"]
            assert all(scope.newly_prepared for scope in scopes.opened)
            assert len({id(child) for child in scopes.runtimes}) == 2
            assert len({id(store) for store in scopes.stores}) == 2
            assert len(set(scopes.workspaces)) == 2
            assert calls == scopes.workspaces[:1]
            assert all(not store.ready for store in scopes.stores)
            assert storage.ready
            assert (await run.scores()).items[0].score.dimensions == {"quality": 1.0}

            rescored = await run.rescore(RescoreRequest((_scorer(tasks[1]),), "rescore"),
                engine=runtime.tasks.bind(tasks[1]), trial_scope=scopes)
            assert (await rescored.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            await _until(lambda: len(scopes.closed) == 3)
            assert len(calls) == 1
            assert scopes.opened[-1].scorer_slot_id == "quality"
            assert scopes.opened[-1].newly_prepared
            assert scopes.opened[-1].experiment_id == rescored.experiment_id
            assert scopes.opened[-1].trial.target_experiment_id == run.experiment_id
            assert (await rescored.scores()).items[0].score.dimensions == {"quality": 1.0}
            assert await runtime.evaluations.get_dataset(request.spec.dataset, principal=PRINCIPAL)
            if sql is not None:
                assert disposed == []
                async with sql.connect() as connection:
                    assert await connection.scalar(text("SELECT 1")) == 1
        assert disposed == []
    finally:
        if sql is not None:
            await sql.dispose()


@pytest.mark.asyncio
async def test_child_agent_reads_captured_input_with_its_own_model_and_runtime_binding(
    tmp_path: Path,
) -> None:
    planning_models = FixtureModels()
    children: list[tuple[EvaluationTrialScope, FixtureModels]] = []
    closed: list[EvaluationTrialScope] = []
    group = CapabilityGroup[None]("scoped-agent")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    body = b"private captured attachment"

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target_output == sample.expected == {"text": "fixture answer"}
        assert "captured prompt" in json.dumps(sample.target_input)
        return ScoreBundle(dimensions={"quality": 1.0}).to_mapping()

    scorer = Task("scopes.agent-score", score, effect_policy="none")
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    async with Runtime.open(NAMESPACE, models=planning_models, storage=storage, context=CONTEXT,
                            capabilities=(group,)) as runtime:
        target = runtime.tasks.from_agent("scopes.agent", runtime.agents.get())
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("agent-input", 1), (
            CaseSpec.agent(CaseRef("agent-input", "one", 1),
                prompt=("captured prompt", BinaryContent(body, media_type="text/plain")),
                expected={"text": "fixture answer"}),
        )), principal=PRINCIPAL, idempotency_key="publish-agent")
        case = (await runtime.evaluations.list_cases(dataset, principal=PRINCIPAL)).items[0]
        assert isinstance(case.input, AgentInputCaptureRef)

        @asynccontextmanager
        async def scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[None]]:
            record = await storage.evaluation.records.get(descriptor.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            assert any(intent.submission == descriptor.submission for intent in record.intents)
            models = FixtureModels()
            child_storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
            try:
                async with Runtime.open(NAMESPACE, models=models, storage=child_storage, context=CONTEXT,
                                        capabilities=(group,), auto_recover=False) as child:
                    child_target = child.tasks.from_agent("scopes.agent", child.agents.get())
                    assert child_target.ref == target.ref
                    assert child_target.contract == target.contract
                    children.append((descriptor, models))
                    yield child.tasks.bind(child_target, scorer)
            finally:
                closed.append(descriptor)

        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=target.ref),), (ScorerSpec("quality", scorer.ref, (DIMENSION,)),),
            policy=EvaluationPolicy(model_fixtures=(planning_models.contract,))), PRINCIPAL, "start-agent"),
            engine=runtime.tasks.bind(target, scorer), trial_scope=scope)
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        await _until(lambda: len(closed) == 2)
        assert planning_models.prompts == planning_models.attachments == []
        target_models = [models for descriptor, models in children if descriptor.scorer_slot_id is None]
        assert len(target_models) == 1
        assert target_models[0].prompts == ["captured prompt"]
        assert target_models[0].attachments == [body]
        assert all(models.prompts == [] for descriptor, models in children if descriptor.scorer_slot_id is not None)
        trial = (await run.trials()).items[0]
        evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
        assert evidence.target.output.value == {"text": "fixture answer"}
        assert "captured prompt" in json.dumps(evidence.input.value)
        scores = (await run.scores()).items
        assert len(scores) == 1 and scores[0].status == "valid"
        assert scores[0].score.dimensions == {"quality": 1.0}
        report = await run.create_report()
        assert report.scores[0].valid == 1 and report.scores[0].mean == 1.0


@pytest.mark.asyncio
async def test_admitted_agent_reconciles_after_real_runtime_reopen_with_original_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_entered, stop_owner = asyncio.Event(), asyncio.Event()
    completed_outputs: list[str] = []
    opened: list[tuple[EvaluationTrialScope, Path]] = []
    closed: list[EvaluationTrialScope] = []
    child_stores: list[RuntimeStorage] = []
    model_prompts: list[str] = []
    interrupted = True
    group = CapabilityGroup[Path]("recoverable-scoped-agent")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    class RecoveringModels(FixtureModels):
        def __init__(self, *, interrupt: bool) -> None:
            super().__init__()
            self.interrupt = interrupt

        def materialize(self) -> FunctionModel:
            async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
                del info
                prompt = [part.content for message in messages for part in message.parts
                          if isinstance(part, UserPromptPart)][-1]
                assert isinstance(prompt, str)
                model_prompts.append(prompt)
                if self.interrupt:
                    model_entered.set()
                    await stop_owner.wait()
                    raise RuntimeError("injected stopped execution owner")
                completed_outputs.append("fixture answer")
                yield "fixture answer"

            return FunctionModel(stream_function=respond)

    async def score(context: TaskNodeContext[Path]) -> JsonValue:
        assert context.principal == PRINCIPAL
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target_output == sample.expected == {"text": "fixture answer"}
        assert "immutable recovery prompt" in json.dumps(sample.target_input)
        return ScoreBundle(dimensions={"quality": 1.0}).to_mapping()

    scorer = Task("scopes.recovery-score", score, effect_policy="none")

    @asynccontextmanager
    async def scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        workspace = tmp_path / "workspaces" / descriptor.experiment_id / descriptor.slot_id
        workspace.mkdir(parents=True, exist_ok=True)
        marker = workspace / "retained.txt"
        if not marker.exists():
            marker.write_text("application-owned recovery workspace", encoding="utf-8")
        models = RecoveringModels(interrupt=interrupted and descriptor.scorer_slot_id is None)
        child_storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
        token = _SCOPE_CONTEXT.set(descriptor)
        try:
            async with Runtime.open(NAMESPACE, models=models, storage=child_storage,
                context=RuntimeContext(workspace, tenant_id=PRINCIPAL.tenant_id),
                capabilities=(group, CapabilityGroup("workspace", workspace=Workspace.load(workspace))),
                auto_recover=False) as child:
                target = child.tasks.from_agent("scopes.recovery-agent", child.agents.get())
                opened.append((descriptor, workspace))
                child_stores.append(child_storage)
                with monkeypatch.context() as fault:
                    if models.interrupt:
                        backend = child._execution_service.runtime_backend()

                        async def commit_stopped_owner(
                            execution: ExecutionRecord, error: Exception, *,
                            agent_run_id: str | None = None, producer_generation: int | None = None,
                        ) -> ExecutionRecord:
                            del error, agent_run_id
                            return await backend._commit_recovery_required(
                                execution, AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED), (),
                                producer_generation=producer_generation,
                            )

                        fault.setattr(backend, "_commit_failure", commit_stopped_owner)
                    yield child.tasks.bind(target, scorer)
        finally:
            closed.append(descriptor)
            _SCOPE_CONTEXT.reset(token)

    planning_models = FixtureModels()
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    try:
        async with Runtime.open(NAMESPACE, models=planning_models, storage=storage, context=CONTEXT,
                                capabilities=(group,), auto_recover=False) as runtime:
            target = runtime.tasks.from_agent("scopes.recovery-agent", runtime.agents.get())
            dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("recoverable-agent", 1), (
                CaseSpec.agent(CaseRef("recoverable-agent", "one", 1), prompt="immutable recovery prompt",
                               expected={"text": "fixture answer"}),
            )), principal=PRINCIPAL, idempotency_key="publish-recoverable")
            capture = (await runtime.evaluations.list_cases(dataset, principal=PRINCIPAL)).items[0].input
            assert isinstance(capture, AgentInputCaptureRef)
            run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("target", task=target.ref),), (ScorerSpec("quality", scorer.ref, (DIMENSION,)),),
                policy=EvaluationPolicy(model_fixtures=(planning_models.contract,))), PRINCIPAL, "recoverable-start"),
                engine=runtime.tasks.bind(target, scorer), trial_scope=scope)
            await asyncio.wait_for(model_entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            record = await _wait_for_committed(
                lambda: storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id),
                lambda value: value is not None and bool(value.intents) and value.intents[0].confirmed,
                timeout=EVALUATION_COMPLETION_TIMEOUT_SECONDS,
            )
            original_submission = record.intents[0].submission
            graph_id = original_submission.graph.graph_id
            bound = await _wait_for_committed(
                lambda: storage.task.tasks.graph_state(graph_id, tenant_id=PRINCIPAL.tenant_id),
                lambda value: value is not None and value.node_states[0].execution_id is not None,
                timeout=EVALUATION_COMPLETION_TIMEOUT_SECONDS,
            )
            execution_id = bound.node_states[0].execution_id
            original = await storage.execution.executions.get(execution_id, tenant_id=PRINCIPAL.tenant_id)
            checkpoint = await storage.recovery.checkpoints.get(execution_id, tenant_id=PRINCIPAL.tenant_id)
            assert checkpoint is not None
            assert original.status is ExecutionStatus.STARTED
            stop_owner.set()
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "needs_attention"
            await _until(lambda: len(closed) == 1)
            stopped = await storage.execution.executions.get(execution_id, tenant_id=PRINCIPAL.tenant_id)
            assert stopped.status is ExecutionStatus.RECOVERY_REQUIRED
            assert completed_outputs == []
            experiment_id = run.experiment_id
        assert not storage.ready and not child_stores[0].ready

        interrupted = False
        reopened_models = FixtureModels()
        reopened = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
        async with Runtime.open(NAMESPACE, models=reopened_models, storage=reopened, context=CONTEXT,
                                capabilities=(group,), auto_recover=False) as runtime:
            target = runtime.tasks.from_agent("scopes.recovery-agent", runtime.agents.get())
            engine = runtime.tasks.bind(target, scorer)
            resumed = await runtime.evaluations.reconcile(experiment_id, engine=engine, principal=PRINCIPAL,
                idempotency_key="recover-admitted-agent", trial_scope=scope)
            view = (await resumed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            await _until(lambda: len(closed) == 3)
            recovered_targets = [(descriptor, workspace) for descriptor, workspace in opened
                                 if descriptor.scorer_slot_id is None]
            assert len(recovered_targets) == 2 and recovered_targets[0] == recovered_targets[1]
            assert recovered_targets[1][0].submission == original_submission
            assert all(descriptor.principal == PRINCIPAL for descriptor, _ in opened)
            assert all((workspace / "retained.txt").exists() for _, workspace in opened)
            final_graph = await reopened.task.tasks.graph_state(graph_id, tenant_id=PRINCIPAL.tenant_id)
            assert final_graph.node_states[0].execution_id == execution_id
            assert final_graph.nodes[0].input_capture == bound.nodes[0].input_capture
            current = await reopened.execution.executions.get(execution_id, tenant_id=PRINCIPAL.tenant_id)
            assert current.status is ExecutionStatus.SUCCEEDED
            assert current.principal_id == original.principal_id == PRINCIPAL.principal_id
            assert current.principal_kind == original.principal_kind == PRINCIPAL.kind
            assert current.stored_user_input == original.stored_user_input
            assert (await runtime.evaluations.list_cases(dataset, principal=PRINCIPAL)).items[0].input == capture
            trial = (await resumed.trials()).items[0]
            assert trial.subject.execution_id == execution_id
            assert trial.graph_ref.graph_id == graph_id
            evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
            assert evidence.target.output.value == {"text": "fixture answer"}
            scores = (await resumed.scores()).items
            assert len(scores) == 1 and scores[0].status == "valid"
            assert scores[0].score.dimensions == {"quality": 1.0}
            await runtime.evaluations.reconcile(experiment_id, engine=engine, principal=PRINCIPAL,
                idempotency_key="recover-admitted-agent", trial_scope=scope)
            assert completed_outputs == ["fixture answer"]
            assert model_prompts == ["immutable recovery prompt"] * 2
            assert planning_models.prompts == reopened_models.prompts == []
    finally:
        stop_owner.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("submitter", ("local", "foreign"))
async def test_human_score_resumes_only_in_the_coordinator_that_owns_its_scope(
    tmp_path: Path, submitter: str,
) -> None:
    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        return dict(context.input)

    task = Task("scopes.human-target", target, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "owner", storage, make_storage, (task,))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("human", 1), (
            CaseSpec.task(CaseRef("human", "one", 1), input={"answer": "private case input"}),
        )), principal=PRINCIPAL, idempotency_key="publish-human")
        request = StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=task.ref),), (scorer,)), PRINCIPAL, "start-human")
        run = await runtime.evaluations.start(request, engine=runtime.tasks.bind(task), trial_scope=scopes)

        async def waiting_score() -> ScoreAttemptView:
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_graph is not None and pending.scorer_execution is not None:
                    state = await storage.task.tasks.graph_state(pending.scorer_graph.graph_id,
                        tenant_id=PRINCIPAL.tenant_id)
                    record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                    if (any(intent.scorer_slot_id == "human" and intent.confirmed for intent in record.intents)
                            and any(node.node_id == "score" and node.status is TaskStatus.WAITING
                                    for node in state.node_states)):
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(waiting_score(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        decision = HumanScoreRequest(pending.trial.trial_id, "human", pending.evidence_ref,
            ScoreBundle(dimensions={"quality": 1.0}), "human-decision")
        if submitter == "local":
            await run.submit_human_score(decision)
        else:
            foreign_storage = make_storage()
            async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=foreign_storage,
                                    context=CONTEXT, auto_recover=False) as foreign:
                foreign_scopes = Scopes(tmp_path / "foreign", foreign_storage, make_storage, (task,))
                observed = await foreign.evaluations.start(request,
                    engine=foreign.tasks.bind(task), trial_scope=foreign_scopes)
                assert observed.experiment_id == run.experiment_id
                await observed.submit_human_score(decision)
                view = (await observed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
                assert view.completion == "complete", view.needs_attention
                assert (await observed.submit_human_score(decision)).status == "valid"
                assert foreign_scopes.opened == foreign_scopes.closed == []
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        await _until(lambda: len(scopes.closed) == 2)
        scores = (await run.scores()).items
        assert len(scores) == 1 and scores[0].status == "valid"
        assert scores[0].score.dimensions == {"quality": 1.0}
        assert (await run.submit_human_score(decision)).status == "valid"
        assert [descriptor.scorer_slot_id for descriptor in scopes.opened] == [None, "human"]
        assert scopes.closed == scopes.opened


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", (
    "target_error", "cancel", "cancel_admission", "retention", "coordinator_close",
))
async def test_local_scope_is_closed_on_target_error_cancellation_and_coordinator_shutdown(
    tmp_path: Path, termination: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        entered.set()
        await release.wait()
        raise ValueError("offline target failure")

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)
        engine = runtime.tasks.bind(*tasks)
        policy = (EvaluationPolicy(content_retention_seconds=60, metadata_retention_seconds=60)
                  if termination == "retention" else None)
        request = await _request(runtime, tasks, policy=policy)
        run = await runtime.evaluations.start(request, engine=engine, trial_scope=scopes)
        try:
            await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            if termination == "target_error":
                release.set()
                assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
                assert (await run.trials()).items[0].graph_status is TaskStatus.FAILED
            elif termination == "cancel":
                receipt_waiting, release_receipt, close_attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
                graph_reading, release_graph_read = asyncio.Event(), asyncio.Event()
                child_storage = scopes.stores[0]
                compare_and_swap = child_storage.task.operations.compare_and_swap
                close_scope = runtime.evaluations._close_trial_scope
                tick = runtime.evaluations._tick
                graph_state = runtime.evaluations._graph_state
                watcher = runtime.evaluations._watchers[run.experiment_id]
                inside_tick = ContextVar("evaluation_tick_observation", default=False)
                descriptor = scopes.opened[0]
                held_operations: list[str] = []

                async def observe_tick(*args: object, **kwargs: object) -> None:
                    token = inside_tick.set(True)
                    try:
                        await tick(*args, **kwargs)
                    finally:
                        inside_tick.reset(token)

                async def pause_graph_read(
                    intent: EvaluationLaunchIntent, principal: Principal,
                ) -> TaskGraphState | None:
                    if (asyncio.current_task() is watcher and inside_tick.get()
                            and intent.slot_id == descriptor.slot_id and not graph_reading.is_set()):
                        graph_reading.set()
                        await release_graph_read.wait()
                    return await graph_state(intent, principal)

                async def delay_cancel_receipt(
                    operation_id: str, *, tenant_id: str, expected_status: OperationStatus,
                    next_record: OperationLedgerRecord,
                ) -> OperationLedgerRecord:
                    if (next_record.operation_kind is OperationKind.TASK_CANCEL
                            and next_record.status is OperationStatus.SUCCEEDED):
                        held_operations.append(operation_id)
                        receipt_waiting.set()
                        await release_receipt.wait()
                    return await compare_and_swap(operation_id, tenant_id=tenant_id,
                        expected_status=expected_status, next_record=next_record)

                async def observe_close(key: tuple[str, str]) -> None:
                    if key == (descriptor.experiment_id, descriptor.slot_id):
                        close_attempted.set()
                    await close_scope(key)

                with monkeypatch.context() as pause:
                    pause.setattr(child_storage.task.operations, "compare_and_swap", delay_cancel_receipt)
                    pause.setattr(runtime.evaluations, "_close_trial_scope", observe_close)
                    pause.setattr(runtime.evaluations, "_tick", observe_tick)
                    pause.setattr(runtime.evaluations, "_graph_state", pause_graph_read)
                    cancelling = None
                    try:
                        await asyncio.wait_for(graph_reading.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
                        cancelling = asyncio.create_task(run.cancel(idempotency_key="cancel"))
                        await asyncio.wait_for(receipt_waiting.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
                        state = await storage.task.tasks.graph_state(descriptor.submission.graph.graph_id,
                            tenant_id=PRINCIPAL.tenant_id)
                        assert state.status is TaskStatus.CANCELLED
                        release_graph_read.set()
                        await asyncio.wait_for(close_attempted.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
                        assert scopes.closed == []
                        assert child_storage.ready
                        assert not cancelling.done()
                        receipt = await child_storage.task.operations.get(held_operations[0], tenant_id=PRINCIPAL.tenant_id)
                        assert receipt.status is OperationStatus.RUNNING
                        release_receipt.set()
                        await cancelling
                    finally:
                        release_graph_read.set()
                        release_receipt.set()
                        if cancelling is not None:
                            await asyncio.gather(cancelling, return_exceptions=True)
                view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
                assert view.completion == "cancelled", view.needs_attention
            elif termination == "cancel_admission":
                cancelled = await runtime.evaluations.cancel_admission(request, engine=engine, trial_scope=scopes)
                assert cancelled.experiment_id == run.experiment_id
                view = (await cancelled.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
                assert view.completion == "cancelled", view.needs_attention
            elif termination == "retention":
                record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                graph_id = record.intents[0].submission.graph.graph_id

                class Offline:
                    @asynccontextmanager
                    async def offline_exclusivity(self) -> AsyncIterator[None]:
                        state = await storage.task.tasks.graph_state(graph_id, tenant_id=PRINCIPAL.tenant_id)
                        assert state.status is TaskStatus.CANCELLED
                        await runtime.evaluations.close()
                        yield

                purged = await runtime.evaluations.purge_expired(principal=PRINCIPAL,
                    now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Offline())
                assert purged.evaluations == (run.experiment_id,)
            else:
                await runtime.evaluations.close()
            await _until(lambda: len(scopes.closed) == 1)
            await _until(lambda: all(not store.ready for store in scopes.stores))
            assert scopes.closed == scopes.opened
            assert all((workspace / "retained.txt").exists() for workspace in scopes.workspaces)
            assert storage.ready
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("interrupted_entry", "cancel_entry", "exit"))
async def test_scope_cleanup_failure_is_observable_without_repeating_context_exit(
    tmp_path: Path, boundary: str,
) -> None:
    class ScopeCleanupError(RuntimeError):
        pass

    entered, cleanup = asyncio.Event(), asyncio.Event()
    cleanup_attempts: list[EvaluationTrialScope] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    manager = Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT)
    runtime = await manager.__aenter__()
    scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)

    @asynccontextmanager
    async def broken_scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        try:
            async with scopes(descriptor) as engine:
                entered.set()
                if boundary != "exit":
                    await asyncio.Event().wait()
                yield engine
        finally:
            cleanup_attempts.append(descriptor)
            cleanup.set()
            raise ScopeCleanupError("application scope cleanup failed")

    try:
        run = await runtime.evaluations.start(await _request(runtime, tasks),
            engine=runtime.tasks.bind(*tasks), trial_scope=broken_scope)
        await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        cancel_error = None
        if boundary == "exit":
            await asyncio.wait_for(cleanup.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        elif boundary == "cancel_entry":
            try:
                await run.cancel(idempotency_key="cancel-during-entry")
            except ScopeCleanupError as error:
                cancel_error = error
        close_error = None
        try:
            await runtime.evaluations.close()
        except ScopeCleanupError as error:
            close_error = error
        view = await run.inspect()
        assert cancel_error is not None or close_error is not None or any(
            issue.code == "ScopeCleanupError" for issue in view.needs_attention)
        if boundary == "cancel_entry":
            with pytest.raises(ScopeCleanupError):
                await runtime.evaluations.close()
        assert len(cleanup_attempts) == 1
        assert cleanup_attempts == scopes.opened == scopes.closed
        assert all(not child_storage.ready for child_storage in scopes.stores)
        assert all((workspace / "retained.txt").exists() for workspace in scopes.workspaces)
    finally:
        try:
            await manager.__aexit__(None, None, None)
        except ScopeCleanupError:
            pass
        finally:
            await storage.close()
    assert len(cleanup_attempts) == 1


@pytest.mark.asyncio
async def test_reconcile_attempts_every_owned_scope_cleanup_when_one_exit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScopeCleanupError(RuntimeError):
        pass

    release = asyncio.Event()
    slots: list[str] = []
    exited: list[str] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        await release.wait()
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    manager = Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT)
    runtime = await manager.__aenter__()
    scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)

    @asynccontextmanager
    async def failing_scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        if descriptor.slot_id == slots[-1]:
            raise AIError(ErrorCode.BINDING_CONFLICT)
        try:
            async with scopes(descriptor) as engine:
                yield engine
        finally:
            exited.append(descriptor.slot_id)
            if descriptor.slot_id == slots[0]:
                raise ScopeCleanupError("first owned scope cleanup failed")

    try:
        engine = runtime.tasks.bind(*tasks)
        request = await _request(runtime, tasks)
        request = replace(request, spec=replace(request.spec, repetitions=3,
            policy=EvaluationPolicy(target_concurrency=3)))
        with monkeypatch.context() as initial:
            initial.setattr(runtime.evaluations, "_watch", lambda *args, **kwargs: None)
            run = await runtime.evaluations.start(request, engine=engine, trial_scope=failing_scope)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        for plan in record.manifest.trials:
            trial = TargetTrialRef(run.experiment_id, plan.trial_id)
            slot = f"target:{trial.trial_id}"
            slots.append(slot)
            graph = TaskGraph(canonical_sha256({"experiment": run.experiment_id, "slot": slot}), (
                TaskNode("target", task=tasks[0], input={"answer": "private case input"}),
            ))
            submission = await engine.describe_submission(graph, principal=PRINCIPAL,
                idempotency_key=f"evaluation:{run.experiment_id}:{slot}",
                correlation={"evaluation_experiment": run.experiment_id,
                             "evaluation_trial": trial.trial_id, "evaluation_slot": slot})
            await storage.evaluation.records.register_launch_intent(run.experiment_id,
                EvaluationLaunchIntent(slot, trial, None, submission, None), capacity=3)
        assert scopes.opened == []
        with pytest.raises((AIError, ScopeCleanupError)) as failure:
            await runtime.evaluations.reconcile(run.experiment_id, engine=engine, principal=PRINCIPAL,
                idempotency_key="reconcile-cleanup", trial_scope=failing_scope)
        if isinstance(failure.value, AIError):
            assert failure.value.code is ErrorCode.BINDING_CONFLICT
        assert {descriptor.slot_id for descriptor in scopes.opened} == set(slots[:2])
        assert {descriptor.slot_id for descriptor in scopes.closed} == set(slots[:2])
        assert sorted(exited) == sorted(slots[:2])
        assert all(not child_storage.ready for child_storage in scopes.stores)
        assert all((workspace / "retained.txt").exists() for workspace in scopes.workspaces)
    finally:
        release.set()
        try:
            await manager.__aexit__(None, None, None)
        except ScopeCleanupError:
            pass
        finally:
            await storage.close()
    assert sorted(exited) == sorted(slots[:2])


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_live", (False, True))
async def test_cancel_before_admission_never_opens_a_trial_scope(
    tmp_path: Path, shared_live: bool,
) -> None:
    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        pytest.fail("a cancelled admission must never execute")

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    opened: list[EvaluationTrialScope] = []

    @asynccontextmanager
    async def scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        opened.append(descriptor)
        pytest.fail("a cancelled admission must not acquire application resources")
        yield

    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage,
        context=RuntimeContext(tmp_path if shared_live else None, tenant_id=PRINCIPAL.tenant_id),
        capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(tmp_path)),)
                     if shared_live else ()) as runtime:
        engine = runtime.tasks.bind(*tasks)
        request = await _request(runtime, tasks,
            policy=EvaluationPolicy(external_effects="live" if shared_live else "deny"))
        cancelled = await runtime.evaluations.cancel_admission(request, engine=engine, trial_scope=scope)
        repeated = await runtime.evaluations.start(request, engine=engine, trial_scope=scope)
        assert cancelled.experiment_id == repeated.experiment_id
        if not shared_live:
            with pytest.raises(AIError) as mode_changed:
                await runtime.evaluations.start(request, engine=engine)
            assert mode_changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert (await repeated.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "cancelled"
        record = await storage.evaluation.records.get(cancelled.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record.intents == ()
        assert opened == []


@pytest.mark.asyncio
@pytest.mark.parametrize("child_environment", ("bare", "app", "workspace", "both"))
async def test_explicit_live_effects_run_in_each_trial_runtime_environment(
    tmp_path: Path, child_environment: str,
) -> None:
    calls: list[Path | None] = []
    opened: list[EvaluationTrialScope] = []
    closed: list[EvaluationTrialScope] = []
    child_stores: list[RuntimeStorage] = []
    output = tmp_path / "target-output.txt"

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        assert context.principal == PRINCIPAL
        calls.append(context.app)
        output.write_text(context.input["answer"], encoding="utf-8")
        return dict(context.input)

    async def score(context: TaskNodeContext[Path]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target_output == sample.expected == {"answer": "private case input"}
        return ScoreBundle(dimensions={"quality": 1.0}).to_mapping()

    tasks = (Task("scopes.target", target, effect_policy="replay_safe"),
             Task("scopes.score", score, effect_policy="none"))
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    workspace = CapabilityGroup("workspace", workspace=Workspace.load(tmp_path))

    @asynccontextmanager
    async def scope(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        child_storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
        child_stores.append(child_storage)
        try:
            async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=child_storage,
                context=RuntimeContext(tmp_path if child_environment in {"app", "both"} else None,
                                       tenant_id=PRINCIPAL.tenant_id),
                capabilities=(workspace,) if child_environment in {"workspace", "both"} else (),
                auto_recover=False) as child:
                opened.append(descriptor)
                yield child.tasks.bind(*tasks)
        finally:
            closed.append(descriptor)

    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage,
        context=RuntimeContext(tmp_path if child_environment == "bare" else None,
                               tenant_id=PRINCIPAL.tenant_id),
        capabilities=(workspace,) if child_environment == "bare" else ()) as runtime:
        request = await _request(runtime, tasks, policy=EvaluationPolicy(external_effects="live"))
        run = await runtime.evaluations.start(request, engine=runtime.tasks.bind(*tasks), trial_scope=scope)
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        await _until(lambda: bool(opened) and closed == opened)
        assert all(not child_storage.ready for child_storage in child_stores)
        assert storage.ready
        assert view.completion == "complete", view.needs_attention
        assert calls == [tmp_path if child_environment in {"app", "both"} else None]
        assert output.read_text(encoding="utf-8") == "private case input"
        assert [descriptor.scorer_slot_id for descriptor in opened] == [None, "quality"]
        assert (await run.scores()).items[0].score.dimensions == {"quality": 1.0}


@pytest.mark.asyncio
async def test_reconcile_preserves_unscoped_live_submission_with_app_and_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        assert context.principal == PRINCIPAL
        assert context.app == tmp_path
        calls.append("target")
        return dict(context.input)

    async def score(context: TaskNodeContext[Path]) -> JsonValue:
        assert context.principal == PRINCIPAL
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target_output == sample.expected == {"answer": "private case input"}
        return ScoreBundle(dimensions={"quality": 1.0}).to_mapping()

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", score, effect_policy="none"))
    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(*tasks)
        request = await _request(runtime, tasks, policy=EvaluationPolicy(external_effects="live"))
        with monkeypatch.context() as paused:
            paused.setattr(runtime.evaluations, "_watch", lambda *args, **kwargs: None)
            run = await runtime.evaluations.start(request, engine=engine)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        trial = TargetTrialRef(run.experiment_id, record.manifest.trials[0].trial_id)
        slot = f"target:{trial.trial_id}"
        graph = TaskGraph(canonical_sha256({"experiment": run.experiment_id, "slot": slot}), (
            TaskNode("target", task=tasks[0], input={"answer": "private case input"}),
        ))
        submission = await engine.describe_submission(graph, principal=PRINCIPAL,
            idempotency_key=f"evaluation:{run.experiment_id}:{slot}",
            correlation={"evaluation_experiment": run.experiment_id,
                         "evaluation_trial": trial.trial_id, "evaluation_slot": slot})
        await storage.evaluation.records.register_launch_intent(run.experiment_id,
            EvaluationLaunchIntent(slot, trial, None, submission, None), capacity=1)
        assert not record.manifest.trial_scope_required
        assert calls == []
        assert await storage.task.admissions.submission_status(submission.ref) is None

    reopened = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=reopened,
        context=RuntimeContext(tmp_path, tenant_id=PRINCIPAL.tenant_id),
        capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(tmp_path)),),
        auto_recover=False) as runtime:
        engine = runtime.tasks.bind(*tasks)
        resumed = await runtime.evaluations.reconcile(run.experiment_id, engine=engine,
            principal=PRINCIPAL, idempotency_key="resume-live")
        view = (await resumed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        current = await reopened.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        intent = next(item for item in current.intents if item.slot_id == slot)
        assert current.manifest == record.manifest
        assert intent.confirmed and intent.submission == submission
        assert intent.submission.admission.principal == PRINCIPAL
        completed_trial = (await resumed.trials()).items[0]
        assert completed_trial.graph_ref.graph_id == submission.graph.graph_id
        assert (await resumed.scores()).items[0].score.dimensions == {"quality": 1.0}
        graph = await reopened.task.tasks.graph_state(submission.graph.graph_id, tenant_id=PRINCIPAL.tenant_id)
        assert graph.node_states[0].execution_id == completed_trial.subject.execution_id
        await runtime.evaluations.reconcile(run.experiment_id, engine=engine,
            principal=PRINCIPAL, idempotency_key="resume-live")
        assert (await resumed.trials()).items[0].graph_ref == completed_trial.graph_ref
        assert calls == ["target"]


@pytest.mark.asyncio
async def test_cancellation_seals_pending_entry_before_awaiting_the_native_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release_entry = asyncio.Event(), asyncio.Event()
    entry_cancelled, scope_closed = asyncio.Event(), asyncio.Event()
    fence_entered, release_target = asyncio.Event(), asyncio.Event()
    calls: list[JsonValue] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        calls.append(dict(context.input))
        await release_target.wait()
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)

        @asynccontextmanager
        async def stubborn_entry(descriptor: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
            try:
                async with scopes(descriptor) as engine:
                    entered.set()
                    try:
                        await release_entry.wait()
                    except asyncio.CancelledError:
                        entry_cancelled.set()
                        await release_entry.wait()
                    yield engine
            finally:
                scope_closed.set()

        cancel_submission = runtime._graph_service.cancel_submission

        async def pause_fence(
            submission: TaskSubmissionRef, *, principal: Principal, idempotency_key: str,
        ) -> TaskSubmissionCancellation:
            fence_entered.set()
            release_entry.set()
            await asyncio.wait_for(scope_closed.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            return await cancel_submission(submission, principal=principal, idempotency_key=idempotency_key)

        run = await runtime.evaluations.start(await _request(runtime, tasks),
            engine=runtime.tasks.bind(*tasks), trial_scope=stubborn_entry)
        try:
            await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            with monkeypatch.context() as pause:
                pause.setattr(runtime._graph_service, "cancel_submission", pause_fence)
                await run.cancel(idempotency_key="cancel-pending-entry")
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "cancelled", view.needs_attention
            assert fence_entered.is_set() and entry_cancelled.is_set() and scope_closed.is_set()
            assert calls == []
            assert len(scopes.opened) == 1 and scopes.closed == scopes.opened
            assert not scopes.stores[0].ready
            assert await storage.task.admissions.submission_status(scopes.opened[0].submission.ref) == "cancelled"
        finally:
            release_entry.set()
            release_target.set()


@pytest.mark.asyncio
async def test_scope_creation_permission_is_local_once_and_failure_remains_cancellable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    joined = asyncio.Event()
    release = asyncio.Event()
    scopes: list[EvaluationTrialScope] = []
    calls: list[str] = []
    waiters = 0
    original_engine = _EnteredTrialScope.engine

    async def observe_waiter(self: _EnteredTrialScope) -> TaskEngine:
        nonlocal waiters
        waiters += 1
        if waiters == 2:
            joined.set()
        return await original_engine(self)

    monkeypatch.setattr(_EnteredTrialScope, "engine", observe_waiter)

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("target")
        return context.input

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    @asynccontextmanager
    async def missing_workspace(scope: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
        scopes.append(scope)
        entered.set()
        await release.wait()
        raise FileNotFoundError("trial_workspace_unavailable")
        yield  # pragma: no cover

    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage,
                            context=CONTEXT, auto_recover=False) as runtime:
        engine = runtime.tasks.bind(*tasks)
        run = await runtime.evaluations.start(await _request(runtime, tasks),
            engine=engine, trial_scope=missing_workspace)
        await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        assert len(scopes) == 1 and scopes[0].newly_prepared
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert len(record.intents) == 1
        submission = record.intents[0].submission
        assert await storage.task.admissions.submission_status(submission.ref) == "prepared"
        assert await storage.task.admissions.get(submission.graph.graph_id, tenant_id=PRINCIPAL.tenant_id) is None

        # A concurrent waiter uses the same reserved local entry, including
        # the one-time creator disposition, rather than opening another scope.
        reconcile = asyncio.create_task(runtime.evaluations.reconcile(run.experiment_id,
            engine=engine, principal=PRINCIPAL, idempotency_key="concurrent", trial_scope=missing_workspace))
        await asyncio.wait_for(joined.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        release.set()
        with pytest.raises(FileNotFoundError, match="trial_workspace_unavailable"):
            await reconcile
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "needs_attention"
        assert len(scopes) == 1

        with pytest.raises(FileNotFoundError, match="trial_workspace_unavailable"):
            await runtime.evaluations.reconcile(run.experiment_id, engine=engine,
                principal=PRINCIPAL, idempotency_key="retry", trial_scope=missing_workspace)
        assert len(scopes) == 2 and not scopes[1].newly_prepared
        assert scopes[1].submission == scopes[0].submission
        await run.cancel(idempotency_key="cancel-missing-workspace")
        cancelled = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert cancelled.gate == "closed_cancel" and all(intent.released for intent in cancelled.intents)
        assert len(scopes) == 2 and calls == []
        assert await storage.task.admissions.submission_status(submission.ref) == "cancelled"
        resumed = await runtime.evaluations.reconcile(run.experiment_id, engine=engine,
            principal=PRINCIPAL, idempotency_key="settle-cancellation", trial_scope=missing_workspace)
        assert (await resumed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "cancelled"
        assert len(scopes) == 2 and calls == []


@pytest.mark.asyncio
async def test_interrupted_scope_entry_closes_local_resources_and_reconciles_the_same_intent(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    calls: list[Path] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        calls.append(context.app)
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)

        @asynccontextmanager
        async def suspended(scope: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
            async with scopes(scope) as engine:
                entered.set()
                await asyncio.Event().wait()
                yield engine

        run = await runtime.evaluations.start(await _request(runtime, tasks),
            engine=runtime.tasks.bind(*tasks), trial_scope=suspended)
        await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        await runtime.evaluations.close()
        assert len(scopes.closed) == 1
        assert scopes.closed == scopes.opened
        assert calls == []
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert len(record.intents) == 1
        pending = record.intents[0]
        assert not pending.confirmed and not pending.released
        assert await storage.task.admissions.submission_status(pending.submission.ref) == "prepared"
        assert scopes.opened[0].newly_prepared
        experiment_id = run.experiment_id

    reopened = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=reopened, context=CONTEXT,
                            auto_recover=False) as runtime:
        resumed_scopes = Scopes(tmp_path / "workspaces", reopened, make_storage, tasks)
        with pytest.raises(AIError) as missing_scope:
            await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(*tasks),
                principal=PRINCIPAL, idempotency_key="missing-scope")
        assert missing_scope.value.code is ErrorCode.EVALUATION_INCOMPATIBLE
        assert calls == []
        resumed = await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(*tasks),
            principal=PRINCIPAL, idempotency_key="resume", trial_scope=resumed_scopes)
        view = (await resumed.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        await _until(lambda: len(resumed_scopes.closed) == 2)
        assert len(calls) == 1
        assert resumed_scopes.opened[0].submission == pending.submission
        assert resumed_scopes.opened[0].principal == PRINCIPAL
        assert not resumed_scopes.opened[0].newly_prepared
        assert resumed_scopes.opened[0] == replace(scopes.opened[0], newly_prepared=False)
        assert resumed_scopes.workspaces[0] == scopes.workspaces[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ("namespace", "tenant", "definition_policy", "backing_store"))
async def test_incompatible_scope_engine_is_closed_before_target_execution(
    tmp_path: Path, mismatch: str,
) -> None:
    called: list[str] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        called.append("target")
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))
    child_tasks = ((Task("scopes.target", target, effect_policy="replay_safe"), tasks[1])
                   if mismatch == "definition_policy" else tasks)

    def make_child_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / ("other-state.sqlite" if mismatch == "backing_store" else "state.sqlite"))

    storage = RuntimeStorage.sqlite(tmp_path / "state.sqlite")
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "workspaces", storage, make_child_storage, child_tasks,
            namespace="wrong-namespace" if mismatch == "namespace" else NAMESPACE,
            tenant_id="wrong-tenant" if mismatch == "tenant" else PRINCIPAL.tenant_id)
        run = await runtime.evaluations.start(await _request(runtime, tasks),
            engine=runtime.tasks.bind(*tasks), trial_scope=scopes)
        await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        await _until(lambda: bool(scopes.closed))
        await _until(lambda: all(not store.ready for store in scopes.stores))
        assert called == []
        assert scopes.closed == scopes.opened
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert record.dispositions
        assert any(item.disposition.reason_code in {
            str(ErrorCode.BINDING_CONFLICT), str(ErrorCode.EVALUATION_INCOMPATIBLE),
        } for item in record.dispositions)
        assert await storage.task.admissions.submission_status(scopes.opened[0].submission.ref) in {"prepared", "cancelled"}


@pytest.mark.asyncio
async def test_closing_one_coordinator_does_not_stop_another_experiments_child(
    tmp_path: Path,
) -> None:
    started_a, started_b = asyncio.Event(), asyncio.Event()
    release_a, release_b = asyncio.Event(), asyncio.Event()
    finished_a = asyncio.Event()

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        if context.app.is_relative_to(tmp_path / "a"):
            started_a.set()
            try:
                await release_a.wait()
            finally:
                finished_a.set()
        else:
            started_b.set()
            await release_b.wait()
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage_a, storage_b = make_storage(), make_storage()
    async with (
        Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage_a, context=CONTEXT) as first,
        Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage_b, context=CONTEXT,
                     auto_recover=False) as second,
    ):
        scopes_a = Scopes(tmp_path / "a", storage_a, make_storage, tasks)
        scopes_b = Scopes(tmp_path / "b", storage_b, make_storage, tasks)
        run_a = await first.evaluations.start(await _request(first, tasks, key="first"),
            engine=first.tasks.bind(*tasks), trial_scope=scopes_a)
        run_b = await second.evaluations.start(await _request(second, tasks, key="second"),
            engine=second.tasks.bind(*tasks), trial_scope=scopes_b)
        try:
            await asyncio.wait_for(asyncio.gather(started_a.wait(), started_b.wait()),
                                   EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            await second.evaluations.close()
            assert scopes_b.closed == scopes_b.opened
            assert len(scopes_b.closed) == 1
            assert scopes_a.closed == []
            assert not finished_a.is_set()
            assert storage_a.ready and storage_b.ready
            record_b = await storage_b.evaluation.records.get(run_b.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            assert len(record_b.intents) == 1
            release_a.set()
            assert (await run_a.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            await _until(lambda: len(scopes_a.closed) == 2)
            assert (await run_a.scores()).items[0].status == "valid"
        finally:
            release_a.set()
            release_b.set()


@pytest.mark.asyncio
async def test_remote_logical_release_does_not_claim_another_coordinators_scope_is_closed(
    tmp_path: Path,
) -> None:
    target_started, release_target = asyncio.Event(), asyncio.Event()
    teardown_started, release_teardown = asyncio.Event(), asyncio.Event()
    target_calls: list[Path] = []

    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        target_calls.append(context.app)
        target_started.set()
        await release_target.wait()
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    storage_a, storage_b = make_storage(), make_storage()
    async with (
        Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage_a, context=CONTEXT) as first,
        Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage_b, context=CONTEXT,
                     auto_recover=False) as second,
    ):
        scopes_a = Scopes(tmp_path / "a", storage_a, make_storage, tasks)
        scopes_b = Scopes(tmp_path / "b", storage_b, make_storage, tasks)

        @asynccontextmanager
        async def delayed_teardown(scope: EvaluationTrialScope) -> AsyncIterator[TaskEngine[Path]]:
            async with scopes_a(scope) as engine:
                try:
                    yield engine
                finally:
                    if scope.scorer_slot_id is None:
                        teardown_started.set()
                        await release_teardown.wait()

        request = await _request(first, tasks)
        run_a = await first.evaluations.start(request,
            engine=first.tasks.bind(*tasks), trial_scope=delayed_teardown)
        try:
            await asyncio.wait_for(target_started.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)

            async def confirmed() -> None:
                while True:
                    record = await storage_a.evaluation.records.get(run_a.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                    if any(intent.scorer_slot_id is None and intent.confirmed for intent in record.intents):
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(confirmed(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            run_b = await second.evaluations.start(request,
                engine=second.tasks.bind(*tasks), trial_scope=scopes_b)
            assert run_b.experiment_id == run_a.experiment_id
            release_target.set()
            await asyncio.wait_for(teardown_started.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            view = (await run_b.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            record = await storage_b.evaluation.records.get(run_b.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            assert all(intent.released for intent in record.intents)
            assert scopes_a.closed == []
            assert scopes_a.stores[0].ready
            assert len(target_calls) == 1
            assert all(scope.scorer_slot_id is not None for scope in scopes_b.opened)
            await second.evaluations.close()
            assert scopes_a.closed == []
        finally:
            release_target.set()
            release_teardown.set()
        await _until(lambda: len(scopes_a.closed) == 1)


@pytest.mark.asyncio
async def test_retention_never_deletes_application_owned_trial_workspaces(tmp_path: Path) -> None:
    async def target(context: TaskNodeContext[Path]) -> JsonValue:
        return dict(context.input)

    tasks = (Task("scopes.target", target, effect_policy="none"),
             Task("scopes.score", _score, effect_policy="none"))

    def make_storage() -> RuntimeStorage:
        return RuntimeStorage.sqlite(tmp_path / "state.sqlite")

    class Offline:
        @asynccontextmanager
        async def offline_exclusivity(self) -> AsyncIterator[None]:
            yield

    storage = make_storage()
    async with Runtime.open(NAMESPACE, models=ModelRegistry(), storage=storage, context=CONTEXT) as runtime:
        scopes = Scopes(tmp_path / "workspaces", storage, make_storage, tasks)
        policy = EvaluationPolicy(content_retention_seconds=60, metadata_retention_seconds=60)
        run = await runtime.evaluations.start(await _request(runtime, tasks, policy=policy),
            engine=runtime.tasks.bind(*tasks), trial_scope=scopes)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        await _until(lambda: len(scopes.closed) == 2)
        record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
        purged = await runtime.evaluations.purge_expired(principal=PRINCIPAL,
            now=record.metadata_expires_at + timedelta(seconds=1), exclusive=Offline())
        assert purged.evaluations == (run.experiment_id,)
        assert all((workspace / "retained.txt").read_text(encoding="utf-8") == "application-owned workspace"
                   for workspace in scopes.workspaces)
        assert scopes.closed == scopes.opened
