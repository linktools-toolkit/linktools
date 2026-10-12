#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation completion preserves execution outcomes and graph recovery independently."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    RescoreRequest, StartEvaluationRequest, evaluation_completion,
)
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.task import Task
from .test_evaluation_consumers import (
    CONTEXT, PRINCIPAL, FixtureModels, echo, exact, rule_scorer,
)


@pytest.mark.asyncio
async def test_target_graph_recovery_does_not_hide_successful_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Task("completion.target", echo, effect_policy="none")
    scorer = Task("completion.score", exact, effect_policy="none")
    async with Runtime.open("completion", models=FixtureModels(),
                            storage=RuntimeStorage.filesystem(tmp_path), context=CONTEXT) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("completion", 1), (
            CaseSpec.task(CaseRef("completion", "one", 1), input={"answer": "yes"}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, "start"),
            engine=runtime.tasks.bind(target, scorer))
        assert (await run.wait(timeout_seconds=30)).result.completion == "complete"
        rescored = await run.rescore(RescoreRequest((rule_scorer(scorer),), "rescore"),
                                     engine=runtime.tasks.bind(scorer))
        assert (await rescored.wait(timeout_seconds=30)).result.completion == "complete"
        owner = runtime.evaluations
        read_state = owner._graph_state

        async def recovery_state(intent, principal):
            state = await read_state(intent, principal)
            return replace(state, status=TaskStatus.RECOVERY_REQUIRED) if intent.scorer_slot_id is None else state

        monkeypatch.setattr(owner, "_graph_state", recovery_state)
        trials = (await run.trials()).items
        assert trials[0].execution_status is ExecutionStatus.SUCCEEDED
        assert trials[0].graph_status is TaskStatus.RECOVERY_REQUIRED
        assert trials[0].terminal
        view = await run.inspect()
        assert view.completion == "needs_attention"
        assert view.progress.terminal_trials == 1
        assert any(issue.code == "recovery_required" and issue.trial == trials[0].trial
                   for issue in view.needs_attention)
        report = await run.create_report()
        assert report.completion == "needs_attention"
        assert report.candidates[0].succeeded == 1
        assert (await runtime.evaluations.get_report(report.report_id, principal=PRINCIPAL)).trials == trials
        rescored_view = await rescored.inspect()
        assert rescored_view.completion == "complete"
        assert rescored_view.needs_attention == ()
        assert (await rescored.create_report()).completion == "complete"


@pytest.mark.parametrize("cancellation_requested,budget_stopped,pending,expected", (
    (False, False, False, "complete"),
    (False, False, True, "running"),
    (True, False, False, "cancelled"),
    (True, False, True, "cancelling"),
    (False, True, False, "cancelled"),
    (False, True, True, "running"),
))
def test_completion_preserves_cancellation_and_budget_gate_meanings(
    cancellation_requested: bool, budget_stopped: bool, pending: bool, expected: str,
) -> None:
    assert evaluation_completion((), (), pending_launches=pending,
        cancellation_requested=cancellation_requested, budget_stopped=budget_stopped) == expected
    assert evaluation_completion((), (), pending_launches=pending, blocked=True,
        cancellation_requested=cancellation_requested, budget_stopped=budget_stopped) == "needs_attention"


@pytest.mark.asyncio
@pytest.mark.parametrize("released,boundary", ((False, "storage"), (True, "storage"),
                                               (False, "author"), (False, "score_node"),
                                               (False, "recovery")))
async def test_inspect_preserves_scorer_graph_read_boundaries_without_saved_score(
    monkeypatch: pytest.MonkeyPatch, released: bool, boundary: str,
) -> None:
    from linktools.ai.core import service_principal
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.evaluation import TargetTrialRef, TrialView
    from linktools.ai.runtime._evaluation import RuntimeEvaluations
    from linktools.ai.runtime.state._evaluation_records import EvaluationLaunchIntent
    from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskGraphState, TaskGraphSubmission, TaskNode, TaskNodeView, TaskRef
    from .test_evaluation_persistence import experiment

    service = object.__new__(RuntimeEvaluations)
    service._namespace = "evaluation"
    record = experiment()
    principal = record.manifest.principal
    caller = service_principal(principal.tenant_id, "viewer") if boundary == "author" else principal
    trial = TargetTrialRef(record.experiment_id, "trial")
    graph = TaskGraph("score-graph", (TaskNode("score", task=TaskRef("score", 1), input={
        "evidence_ref": {"namespace": "evaluation", "tenant_id": principal.tenant_id,
                         "evidence_id": "evidence", "digest": "a" * 64},
    }),))
    admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, principal, "score-start"))
    launch = EvaluationLaunchIntent("score:trial:score", trial, "score",
        TaskGraphSubmission("evaluation", admission, graph), None, confirmed=True, released=released)
    record = replace(record, intents=(launch,))

    async def read_record(experiment_id, actor):
        assert experiment_id == record.experiment_id and actor == caller
        return record

    async def read_trials(value, actor):
        return (TrialView(trial, value.manifest.trials[0].case_ref, "target", 1),), {}

    failure = AIError(ErrorCode.AUTHORIZATION_DENIED if boundary == "author" else ErrorCode.STORAGE_INTEGRITY_ERROR)
    graph_reads = 0

    async def read_graph(intent, actor):
        nonlocal graph_reads
        graph_reads += 1
        assert intent == launch
        if boundary == "storage" or boundary == "author" and actor == principal:
            raise failure
        if boundary == "recovery":
            return TaskGraphState(graph.graph_id, TaskStatus.RECOVERY_REQUIRED, graph.nodes,
                (TaskNodeView(graph.graph_id, "score", (), TaskStatus.RECOVERY_REQUIRED, None, 1, None, None,
                              "recovery_required", "a" * 64),))
        return TaskGraphState(graph.graph_id, TaskStatus.RUNNING,
            (TaskNode("other", task=TaskRef("score", 1)),),
            (TaskNodeView(graph.graph_id, "other", (), TaskStatus.RUNNING, None, 0, None, None, None, None),))

    monkeypatch.setattr(service, "_record", read_record)
    monkeypatch.setattr(service, "_trials", read_trials)
    monkeypatch.setattr(service, "_graph_state", read_graph)
    if boundary == "recovery":
        view = await service._inspect(record.experiment_id, caller)
        assert view.completion == "needs_attention"
        assert view.progress.planned_scores == 1 and view.progress.terminal_scores == 0
        assert view.needs_attention[0].code == "recovery_required"
        assert view.needs_attention[0].trial == trial
        assert graph_reads <= 1
    elif boundary == "score_node":
        with pytest.raises(RuntimeError, match="coroutine raised StopIteration"):
            await service._inspect(record.experiment_id, caller)
    else:
        with pytest.raises(AIError) as raised:
            await service._inspect(record.experiment_id, caller)
        assert raised.value is failure
