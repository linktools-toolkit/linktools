#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation completion preserves execution outcomes and graph recovery independently."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    StartEvaluationRequest, evaluation_completion,
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
