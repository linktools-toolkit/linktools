#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation reads persist only snapshots needed by durable consumers."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSlotRef, CandidateSpec, CaseRef, CaseSpec, ComparisonReport, ComparisonSpec, DatasetRef,
    DatasetSpec, EvaluationPolicy, EvaluationReport, EvaluationSpec, ScoreComparisonSelection, ScoreFilter, ScoreSelection,
    StartEvaluationRequest, TrialFilter,
)
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskNodeContext

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, FixtureModels, exact, rule_scorer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem"])
async def test_pages_save_only_resumable_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    released = asyncio.Event()

    async def target(context: TaskNodeContext[None]) -> JsonValue:
        await released.wait()
        return context.input["answer"]

    task = Task("paging.target", target, effect_policy="none")
    scorer = Task("paging.scorer", exact, effect_policy="none")
    storage = RuntimeStorage.in_memory() if backend == "memory" else RuntimeStorage.filesystem(tmp_path)
    published: list[EvaluationReport | ComparisonReport] = []
    async with Runtime.open("paging", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        state = runtime.evaluations._state
        publish = state.publish_report

        async def save(report: EvaluationReport | ComparisonReport) -> EvaluationReport | ComparisonReport:
            result = await publish(report)
            assert await state.get_report(report.report_id) == report
            published.append(report)
            return result

        monkeypatch.setattr(state, "publish_report", save)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("paging", 1), cases=tuple(
            CaseSpec.task(CaseRef("paging", str(index), 1), input={"answer": str(index)}, expected=str(index))
            for index in range(3)
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=task.ref),), (rule_scorer(scorer),),
            policy=EvaluationPolicy(allow_volatile=backend == "memory"),
        ), PRINCIPAL, "run"), engine=runtime.tasks.bind(task, scorer))
        pages = {}
        for kind in ("trials", "scores"):
            read = run.trials if kind == "trials" else run.scores
            all_pending = await read()
            assert len(all_pending.items) == 3 and all_pending.next_cursor is None
            assert len(published) == len(pages)
            filters = TrialFilter(terminal=False) if kind == "trials" else ScoreFilter(statuses=("pending",))
            first = await read(filters=filters, limit=1)
            assert first.next_cursor is not None and len(first.items) == 1
            assert len(published) == len(pages) + 1
            snapshot = published[-1]
            assert isinstance(snapshot, EvaluationReport)
            frozen = snapshot.trials if kind == "trials" else snapshot.score_attempts
            assert first.items == frozen[:1]
            # A cutoff can match both endpoints once their reports are published.
            if not pages:
                assert await state.get_report_at(snapshot.cutoff) == snapshot
            pages[kind] = (filters, first, snapshot, all_pending)
        released.set()
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        final_cursors = {}
        for kind, (filters, first, snapshot, all_pending) in pages.items():
            read = run.trials if kind == "trials" else run.scores
            frozen = snapshot.trials if kind == "trials" else snapshot.score_attempts
            second = await read(filters=filters, cursor=first.next_cursor, limit=1)
            assert second.next_cursor is not None and second.items == frozen[1:2]
            assert len(published) == 2
            changed_filters = TrialFilter(terminal=True) if kind == "trials" else ScoreFilter(statuses=("valid",))
            with pytest.raises(AIError) as raised:
                await read(filters=changed_filters, cursor=second.next_cursor, limit=1)
            assert raised.value.code is ErrorCode.CURSOR_INVALID
            complete = await read()
            assert len(complete.items) == 3 and complete.items != all_pending.items
            one_filter = (TrialFilter(case_refs=(complete.items[0].case_ref,)) if kind == "trials"
                          else ScoreFilter(trial_ids=(complete.items[0].trial.trial_id,)))
            for _ in range(5):
                assert (await read()).items == complete.items
                single = await read(filters=one_filter, limit=1)
                assert single.items == complete.items[:1] and single.next_cursor is None
                empty = await read(filters=filters, limit=1)
                assert empty.items == () and empty.next_cursor is None
            assert len(published) == 2
            final_cursors[kind] = second.next_cursor
        report = await run.create_report()
        assert len(published) == 3 and published[-1] == report
        assert await runtime.evaluations.get_report(report.report_id, principal=PRINCIPAL) == report
        spec = ComparisonSpec(CandidateSlotRef(run.experiment_id, "current"),
                              CandidateSlotRef(run.experiment_id, "current"),
                              (ScoreComparisonSelection(ScoreSelection("exact", "exact_match"),
                                                        ScoreSelection("exact", "exact_match")),))
        comparison = await runtime.evaluations.create_comparison_report(spec, principal=PRINCIPAL)
        assert len(published) == 5
        assert published[-2].cutoff == report.cutoff
        assert published[-1] == comparison
        assert await runtime.evaluations.get_report(comparison.report_id, principal=PRINCIPAL) == comparison
        repeated = await runtime.evaluations.create_comparison_report(replace(spec, cutoff=comparison.cutoff), principal=PRINCIPAL)
        assert repeated.cutoff == comparison.cutoff and len(published) == 6
        experiment_id = run.experiment_id
        if backend == "memory":
            for kind, (filters, first, snapshot, all_pending) in pages.items():
                read = run.trials if kind == "trials" else run.scores
                frozen = snapshot.trials if kind == "trials" else snapshot.score_attempts
                final = await read(filters=filters, cursor=final_cursors[kind], limit=1)
                assert final.next_cursor is None and final.items == frozen[2:]
                assert len(published) == 6
            return

    # Reopening the backend must recover the original cursors even after live state changes.
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("paging", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        run = await runtime.evaluations.get(experiment_id, principal=PRINCIPAL)
        state = runtime.evaluations._state
        publish = state.publish_report

        async def save_after_reopen(report: EvaluationReport | ComparisonReport) -> EvaluationReport | ComparisonReport:
            published.append(report)
            return await publish(report)

        monkeypatch.setattr(state, "publish_report", save_after_reopen)
        for kind, (filters, first, snapshot, all_pending) in pages.items():
            read = run.trials if kind == "trials" else run.scores
            frozen = snapshot.trials if kind == "trials" else snapshot.score_attempts
            final = await read(filters=filters, cursor=final_cursors[kind], limit=1)
            assert final.next_cursor is None and final.items == frozen[2:]
            assert len(published) == 6
            assert await runtime.evaluations.get_report(snapshot.report_id, principal=PRINCIPAL) == snapshot
