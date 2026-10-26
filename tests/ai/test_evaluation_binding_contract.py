#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation preserves historical bindings and rejects changed source contracts."""

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations
from ._runtime_test_helpers import model_features

from pathlib import Path

import pytest

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSlotRef, CandidateSpec, CaseRef, CaseSpec, ComparisonSpec,
    DatasetRef, DatasetSpec, EvaluationPolicy, EvaluationSpec, GatePolicy, GraphTargetSpec,
    ScoreBundle, ScoreComparisonSelection, ScoreSelection, ScoringInput, StartEvaluationRequest, TaskCaseInput,
)
from linktools.ai.inputs import CaptureInputRequest
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskGraphTemplate, TaskNode, TaskNodeContext, TaskNodeResultRef

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, PRINCIPAL, FixtureModels, echo, exact, rule_scorer


@pytest.mark.asyncio
async def test_historical_agent_capture_restores_original_structured_output_binding(tmp_path: Path) -> None:
    models = FixtureModels()
    group = RuntimeFeatures[None]("historical-consumer")
    group.agent("default", model="default", system_prompt="Preserve the original behavior.",
                allow_tools=(), allow_skills=(), allow_subagents=())
    scorer = Task("consumer.historical-score", exact, effect_policy="none")
    async with (
        Persistence.open('historical-consumers', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_32,
        Runtime.open(persistence=persistence_32, context=CONTEXT, features=model_features(models, group)) as runtime,
    ):
        source = await runtime.agents.get().start("the original question", output=ScoreBundle,
            principal=PRINCIPAL, idempotency_key="historical-source")
        source_result = (await source.wait(timeout_seconds=10)).result
        assert source_result.status is ExecutionStatus.SUCCEEDED
        capture = await runtime.executions.capture_input(source.execution_id,
            CaptureInputRequest(PRINCIPAL, "capture-historical-source", "clean"))
        assert isinstance(capture, AgentInputCaptureRef)
        async with TaskEngine.open(scorer, persistence=persistence_32, context=CONTEXT) as engine:
            captured_input = await engine.inputs.read_agent(capture, principal=PRINCIPAL)
            assert captured_input.binding is not None
            historical = Task.from_agent("consumer.historical", runtime.agents.restore(captured_input.binding))
            async with Evaluations.open(engine.with_definitions(historical)) as evaluations:
                dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("historical", 1), cases=(
                    CaseSpec.from_capture(CaseRef("historical", "original", 1), capture=capture, expected=source_result.output),
                )), principal=PRINCIPAL, idempotency_key="publish-historical")
                run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                    (CandidateSpec("historical", task=historical.ref),), (rule_scorer(scorer),),
                    policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "start-historical"))
                view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
                assert view.completion == "complete", view.needs_attention
                report = await run.create_report()
                assert report.scores[0].valid == 1 and report.scores[0].mean == 1.0
                trial = (await run.trials()).items[0]
                assert trial.subject.execution_id != source.execution_id
                evidence = await evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
                assert evidence.target.output.value == source_result.output
                assert models.prompts == ["the original question", "the original question"]
                assert models.schemas[0] == models.schemas[1]
                assert (await source.wait()).result.output == source_result.output


@pytest.mark.asyncio
async def test_reconcile_rejects_same_revision_task_contract_drift_before_rerunning(tmp_path: Path) -> None:
    calls: list[str] = []

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("target")
        return context.input["answer"]

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    original = Task("consumer.named", target, effect_policy="none")
    changed = Task("consumer.named", target, effect_policy="none", cancel=cancel)
    scorer = Task("consumer.named-score", exact, effect_policy="none")
    async with (
        Persistence.open('drift-consumers', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_76,
        TaskEngine.open(original, scorer, persistence=persistence_76, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("drift", 1), cases=(
            CaseSpec.task(CaseRef("drift", "one", 1), input={"answer": "yes"}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="publish-drift")
        run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("current", task=original.ref),), (rule_scorer(scorer),)), PRINCIPAL, "start-drift"))
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        async with (
            TaskEngine.open(changed, scorer, persistence=persistence_76, context=CONTEXT) as changed_engine,
            Evaluations.open(changed_engine) as changed_evaluations,
        ):
            with pytest.raises(AIError) as raised:
                await changed_evaluations.reconcile(run.experiment_id,
                principal=PRINCIPAL, idempotency_key="reconcile-drift")
        assert raised.value.code is ErrorCode.BINDING_CONFLICT
        assert calls == ["target"]
        assert (await run.inspect()).completion == "complete"


@pytest.mark.asyncio
async def test_strict_comparison_cannot_mix_datasets_even_when_both_scores_pass(tmp_path: Path) -> None:
    target = Task("consumer.dataset-target", echo, effect_policy="none")
    scorer = Task("consumer.dataset-score", exact, effect_policy="none")
    async with (
        Persistence.open('different-dataset-consumers', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_97,
        TaskEngine.open(target, scorer, persistence=persistence_97, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        runs = []
        for name in ("before", "after"):
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef(name, 1), cases=(
                CaseSpec.task(CaseRef(name, "one", 1), input={"answer": "yes"}, expected="yes"),
            )), principal=PRINCIPAL, idempotency_key=f"publish-{name}")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("current", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, f"start-{name}"))
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            assert (await run.create_report()).scores[0].mean == 1.0
            runs.append(run)
        selection = ScoreComparisonSelection(ScoreSelection("exact", "exact_match"), ScoreSelection("exact", "exact_match"))
        comparison = await evaluations.create_comparison_report(ComparisonSpec(
            CandidateSlotRef(runs[0].experiment_id, "current"), CandidateSlotRef(runs[1].experiment_id, "current"),
            (selection,), allowed_changes=("task_definition",), gate_policy=GatePolicy()), principal=PRINCIPAL)
        assert comparison.compatibility == "incompatible" and comparison.gate == "inconclusive"
        assert any(item.path == "dataset" and not item.allowed for item in comparison.differences)
        assert comparison.dimensions[0].complete_pairs == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rerun", (False, True), ids=("frozen", "rerun"))
async def test_two_graph_cases_keep_frozen_or_rerun_dependency_values_separate(rerun: bool) -> None:
    async def source(context: TaskNodeContext[None]) -> JsonValue:
        return context.input["value"]

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("prepared")

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        return ScoreBundle(dimensions={"exact_match": float(sample.target_output["answer"]["value"] == sample.expected)}).to_mapping()

    producer = Task("consumer.captured-source", source, effect_policy="none")
    consumer = Task("consumer.captured-answer", answer, effect_policy="none")
    scorer = Task("consumer.captured-graph-score", score, effect_policy="none")
    async with (
        Persistence.open('captured-graph-consumers', storage=PersistenceStorage.in_memory(), tenant_id='evaluation-consumers') as persistence_135,
    ):
        async with (
            TaskEngine.open(producer, consumer, scorer, persistence=persistence_135, context=CONTEXT) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            cases = []
            for name in ("first", "second"):
                original = await engine.start(TaskGraph(f"original-{name}", nodes=(
                    TaskNode("A", task=producer, input={"value": f"old-{name}"}),
                    TaskNode("B", ("A",), task=consumer, input_refs={"prepared": TaskNodeResultRef("A")}),
                )), principal=PRINCIPAL, idempotency_key=f"original-graph-{name}")
                assert (await original.wait()).result.wait_status is TaskStatus.SUCCEEDED
                capture = await engine.inputs.capture_input((await original.execution("B")).execution_id,
                    CaptureInputRequest(PRINCIPAL, f"capture-{name}", "clean"))
                cases.append(CaseSpec.graph(CaseRef("captured-graphs", name, 1), inputs={
                    "A": TaskCaseInput(input={"value": f"new-{name}"}),
                    "B": TaskCaseInput(capture=capture),
                }, expected=f"{'new' if rerun else 'old'}-{name}"))
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("captured-graphs", 1), cases=tuple(cases)),
                principal=PRINCIPAL, idempotency_key="publish-captured-graphs")
            template = TaskGraphTemplate(nodes=(
                TaskNode("A", task=producer),
                TaskNode("B", ("A",), task=consumer, input_refs={"prepared": TaskNodeResultRef("A")} if rerun else {}),
            ))
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("workflow", graph_template=GraphTargetSpec(template=template, outputs={"answer": "B"})),),
                (rule_scorer(scorer),), policy=EvaluationPolicy(allow_volatile=True)), PRINCIPAL, "start-captured-graphs"))
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            trials = (await run.trials()).items
            assert all(trial.execution_status is TaskStatus.SUCCEEDED for trial in trials), [(trial.case_ref.case_id, trial.execution_status, trial.disposition.reason_code if trial.disposition else trial.error_code) for trial in trials]
            summary = (await run.create_report()).scores[0]
            assert summary.planned == summary.valid == 2 and summary.mean == 1.0
            observed = {}
            for trial in trials:
                evidence = await evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
                observed[trial.case_ref.case_id] = evidence.target.outputs["answer"].value.value
            assert observed == {name: f"{'new' if rerun else 'old'}-{name}" for name in ("first", "second")}


@pytest.mark.asyncio
async def test_rerun_dependency_failure_never_falls_back_to_captured_success(tmp_path: Path) -> None:
    observed: list[TaskStatus] = []

    async def source(context: TaskNodeContext[None]) -> JsonValue:
        if context.input["fail"]:
            raise ValueError("new dependency failed")
        return "old successful body"

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        status = context.dependency_states["prepared"].status
        observed.append(status)
        if status is TaskStatus.FAILED:
            with pytest.raises(AIError) as raised:
                await context.read_dependency("prepared")
            assert raised.value.code is ErrorCode.TASK_DEPENDENCY_FAILED
            return {"status": status.value, "body": "unavailable"}
        return {"status": status.value, "body": await context.read_dependency("prepared")}

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target_status == "succeeded"
        assert sample.target.node_statuses == {"A": "failed", "B": "succeeded"}
        return ScoreBundle(dimensions={"exact_match": float(sample.target_output["answer"]["value"] == sample.expected)}).to_mapping()

    producer = Task("consumer.rerun-failure", source, effect_policy="none")
    consumer = Task("consumer.rerun-observer", answer, effect_policy="none")
    scorer = Task("consumer.rerun-score", score, effect_policy="none")
    async with (
        Persistence.open('rerun-failure-consumers', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_200,
    ):
        async with (
            TaskEngine.open(producer, consumer, scorer, persistence=persistence_200, context=CONTEXT) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            original = await engine.start(TaskGraph("successful-source", nodes=(
                TaskNode("A", task=producer, input={"fail": False}),
                TaskNode("B", ("A",), task=consumer, input_refs={"prepared": TaskNodeResultRef("A")}),
            )), principal=PRINCIPAL, idempotency_key="original-successful-source")
            assert (await original.wait()).result.wait_status is TaskStatus.SUCCEEDED
            capture = await engine.inputs.capture_input((await original.execution("B")).execution_id,
                CaptureInputRequest(PRINCIPAL, "capture-successful-consumer", "clean"))
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("rerun-failure", 1), cases=(
                CaseSpec.graph(CaseRef("rerun-failure", "one", 1), inputs={
                    "A": TaskCaseInput(input={"fail": True}), "B": TaskCaseInput(capture=capture),
                }, expected={"status": "FAILED", "body": "unavailable"}),
            )), principal=PRINCIPAL, idempotency_key="publish-rerun-failure")
            template = TaskGraphTemplate(nodes=(
                TaskNode("A", task=producer, failure_policy="isolate"),
                TaskNode("B", ("A",), task=consumer, input_refs={"prepared": TaskNodeResultRef("A")}, dependency_policy="all_terminal"),
            ))
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("workflow", graph_template=GraphTargetSpec(template=template, outputs={"answer": "B"})),),
                (rule_scorer(scorer),)), PRINCIPAL, "start-rerun-failure"))
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            assert observed == [TaskStatus.SUCCEEDED, TaskStatus.FAILED]
            report = await run.create_report()
            assert report.candidates[0].succeeded == 1
            assert report.scores[0].valid == 1 and report.scores[0].mean == 1.0
