#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selected usage and derived evaluation lifetimes follow retained sources."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSlotRef, CandidateSpec, CaseRef, CaseSpec, ComparisonSpec, DatasetRef,
    DatasetSpec, EvaluationPolicy, EvaluationReadCutoff, EvaluationSpec, GatePolicy, RescoreRequest,
    ScoreBundle, ScoreComparisonSelection, ScoreSelection, ScorerSpec, ScoringInput, StartEvaluationRequest,
)
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime import _evaluation as evaluation_module
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.task import Task, TaskNodeContext, TaskRef

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, DIMENSION, PRINCIPAL, FixtureModels, echo, rule_scorer
from .test_evaluation_retention import Offline
from .test_evaluation_scoring_semantics import UsageModels, capabilities


class Clock:
    def __init__(self) -> None:
        self.value = datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


async def score(context: TaskNodeContext[None]) -> JsonValue:
    return ScoreBundle(dimensions={"exact_match": 1.0}, rationale="retained private explanation").to_mapping()


def test_usage_cutoff_observations_are_immutable_and_round_trip() -> None:
    targets = {"trial": True}
    scorers = {"selected": {"trial": True}, "other": {"trial": False}}
    cutoff = EvaluationReadCutoff("experiment", "a" * 64,
                                  target_usage_complete=targets, scorer_usage_complete=scorers)
    targets["trial"] = False
    scorers["selected"]["trial"] = False
    assert cutoff.target_usage_complete["trial"] and cutoff.scorer_usage_complete["selected"]["trial"]
    assert not cutoff.usage_complete
    assert decode_domain(encode_domain(cutoff), EvaluationReadCutoff) == cutoff
    with pytest.raises(TypeError):
        cutoff.scorer_usage_complete["selected"]["trial"] = False
    with pytest.raises(ValueError):
        EvaluationReadCutoff("experiment", "a" * 64, target_usage_complete={"trial": 1})


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ("rescore", "rule", "failing"))
async def test_usage_gate_uses_only_selected_scoring_work(monkeypatch: pytest.MonkeyPatch, selected: str) -> None:
    models = UsageModels(True)
    target = Task("selection.target", echo, effect_policy="none")
    rule = Task("selection.rule", score, effect_policy="none")
    async with Runtime.open("selected-usage", models=models, storage=RuntimeStorage.in_memory(),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        judge = runtime.tasks.from_agent("selection.judge", runtime.agents.get())
        engine = runtime.tasks.bind(target, rule, judge)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("usage", 1), (
            CaseSpec.task(CaseRef("usage", "bad", 1), input={"answer": "bad"}, expected="bad"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            tuple(CandidateSpec(name, task=target.ref) for name in ("baseline", "candidate")),
            (rule_scorer(judge, slot="failing"), rule_scorer(rule, slot="rule")),
            policy=EvaluationPolicy(allow_volatile=True, model_fixtures=(models.contract,))), PRINCIPAL, "run"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        scoring = await run.rescore(RescoreRequest((rule_scorer(rule, slot="rule"),), "rescore"), engine=engine)
        assert (await scoring.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        selection = ScoreSelection("failing" if selected == "failing" else "rule", "exact_match",
                                   scoring.experiment_id if selected == "rescore" else None)
        spec = ComparisonSpec(CandidateSlotRef(run.experiment_id, "baseline"), CandidateSlotRef(run.experiment_id, "candidate"),
            (ScoreComparisonSelection(selection, selection),), gate_policy=GatePolicy(require_complete_usage=True))
        report = await runtime.evaluations.compare(spec, principal=PRINCIPAL)
        assert ("unknown_required_usage" in report.gate_reasons) is (selected == "failing")
        assert report.gate == ("inconclusive" if selected == "failing" else "pass")

        async def no_new_usage(*args, **kwargs):
            raise AssertionError("a saved usage cutoff cannot observe later requests")

        monkeypatch.setattr(runtime.history, "graph_usage", no_new_usage)
        repeated = await runtime.evaluations.compare(replace(spec, cutoff=report.cutoff), principal=PRINCIPAL)
        assert repeated.gate == report.gate and repeated.cutoff == report.cutoff
        assert await runtime.evaluations.get_report(report.report_id, principal=PRINCIPAL) == report


@pytest.mark.asyncio
async def test_selected_rescore_cannot_hide_unknown_target_usage() -> None:
    class FailedTarget(FixtureModels):
        def materialize(self) -> FunctionModel:
            async def respond(messages, info):
                raise RuntimeError("target provider failed without usage")
                yield "unreachable"
            return FunctionModel(stream_function=respond)

    models = FailedTarget()
    rule = Task("selection.failed-target-score", score, effect_policy="none")
    async with Runtime.open("target-usage", models=models, storage=RuntimeStorage.in_memory(),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        target = runtime.tasks.from_agent("selection.failed-target", runtime.agents.get())
        engine = runtime.tasks.bind(target, rule)
        scorer = replace(rule_scorer(rule), accepts_target_failure=True)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("target-usage", 1), (
            CaseSpec.agent(CaseRef("target-usage", "one", 1), prompt="question"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            tuple(CandidateSpec(name, task=target.ref) for name in ("baseline", "candidate")), (scorer,),
            policy=EvaluationPolicy(allow_volatile=True, model_fixtures=(models.contract,))), PRINCIPAL, "run"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        rescored = await run.rescore(RescoreRequest((scorer,), "rescore"), engine=engine)
        assert (await rescored.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        selection = ScoreSelection("exact", "exact_match", rescored.experiment_id)
        report = await runtime.evaluations.compare(ComparisonSpec(CandidateSlotRef(run.experiment_id, "baseline"),
            CandidateSlotRef(run.experiment_id, "candidate"), (ScoreComparisonSelection(selection, selection),),
            gate_policy=GatePolicy(require_complete_usage=True)), principal=PRINCIPAL)
        assert report.dimensions[0].complete_pairs == 1
        assert report.gate == "inconclusive" and "unknown_required_usage" in report.gate_reasons


@pytest.mark.asyncio
async def test_usage_gate_excludes_an_unselected_candidate() -> None:
    class SelectiveJudge(FixtureModels):
        def materialize(self) -> FunctionModel:
            async def respond(messages, info):
                prompt = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)][-1]
                sample = ScoringInput.from_mapping(json.loads(prompt.split("\nDATA\n", 1)[1]))
                if sample.target_output == "bad":
                    raise RuntimeError("unselected candidate scorer has unknown usage")
                yield {0: DeltaToolCall(name=info.output_tools[0].name,
                    json_args=json.dumps({"dimensions": {"exact_match": 1.0}}))}
            return FunctionModel(stream_function=respond)

    async def good(context: TaskNodeContext[None]) -> JsonValue:
        return "good"

    models = SelectiveJudge()
    target, unrelated = Task("selection.good", good, effect_policy="none"), Task("selection.bad", echo, effect_policy="none")
    async with Runtime.open("candidate-usage", models=models, storage=RuntimeStorage.in_memory(),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        judge = runtime.tasks.from_agent("selection.selective-judge", runtime.agents.get())
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("candidate-usage", 1), (
            CaseSpec.task(CaseRef("candidate-usage", "one", 1), input={"answer": "bad"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("baseline", task=target.ref), CandidateSpec("candidate", task=target.ref),
             CandidateSpec("unselected", task=unrelated.ref)), (rule_scorer(judge),),
            policy=EvaluationPolicy(allow_volatile=True, model_fixtures=(models.contract,))), PRINCIPAL, "run"),
            engine=runtime.tasks.bind(target, unrelated, judge))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        assert not (await run.report()).cutoff.usage_complete
        selection = ScoreSelection("exact", "exact_match")
        report = await runtime.evaluations.compare(ComparisonSpec(CandidateSlotRef(run.experiment_id, "baseline"),
            CandidateSlotRef(run.experiment_id, "candidate"), (ScoreComparisonSelection(selection, selection),),
            gate_policy=GatePolicy(require_complete_usage=True)), principal=PRINCIPAL)
        assert report.gate == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", ("metadata", "content"))
async def test_saved_rescore_reports_expire_with_the_source_and_are_purged(
    monkeypatch: pytest.MonkeyPatch, deadline: str,
) -> None:
    clock = Clock()
    monkeypatch.setattr(evaluation_module, "_now", clock.now)
    target, rule = Task("retained.target", echo, effect_policy="none"), Task("retained.score", score, effect_policy="none")
    storage = RuntimeStorage.in_memory()
    policy = EvaluationPolicy(allow_volatile=True, **{deadline + "_retention_seconds": 60})
    async with Runtime.open("derived-retention", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target, rule)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("retained", 1), (
            CaseSpec.task(CaseRef("retained", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(rule),), policy=policy), PRINCIPAL, "run"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        clock.advance(30)
        scoring = await run.rescore(RescoreRequest((rule_scorer(rule),), "rescore"), engine=engine)
        assert (await scoring.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        saved = await scoring.report()
        assert await runtime.evaluations.get_report(saved.report_id, principal=PRINCIPAL) == saved
        clock.advance(31)
        for read in (scoring.report, lambda: runtime.evaluations.get_report(saved.report_id, principal=PRINCIPAL)):
            with pytest.raises(AIError) as unavailable:
                await read()
            assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        await runtime.evaluations.purge_expired(principal=PRINCIPAL, now=clock.now(), exclusive=Offline())
        with pytest.raises(AIError) as unavailable:
            await runtime.evaluations.get_report(saved.report_id, principal=PRINCIPAL)
        assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        with pytest.raises(AIError):
            await storage.evaluation.records.get_report(saved.report_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", ("metadata", "content"))
async def test_waiting_rescore_settles_when_its_source_lifetime_ends(
    monkeypatch: pytest.MonkeyPatch, deadline: str,
) -> None:
    clock = Clock()
    monkeypatch.setattr(evaluation_module, "_now", clock.now)
    target, rule = Task("source.target", echo, effect_policy="none"), Task("source.score", score, effect_policy="none")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("waiting-source", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target, rule)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("source", 1), (
            CaseSpec.task(CaseRef("source", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(rule),),
            policy=EvaluationPolicy(allow_volatile=True, **{deadline + "_retention_seconds": 60})), PRINCIPAL, "run"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        clock.advance(30)
        rescored = await run.rescore(RescoreRequest((ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,)),), "human"), engine=engine)

        async def ready():
            while True:
                pending = (await rescored.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    if next(node for node in (await graph.state()).node_states if node.node_id == "score").status is TaskStatus.WAITING:
                        return graph
                await asyncio.sleep(0.01)

        graph = await asyncio.wait_for(ready(), 10)
        saved = await rescored.report()
        clock.advance(31)

        async def settled():
            while True:
                record = await storage.evaluation.records.get(rescored.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                if (record.gate != "open" and all(intent.released for intent in record.intents) and
                        any(item.disposition.terminal for item in record.dispositions)):
                    return record
                await asyncio.sleep(0.01)

        record = await asyncio.wait_for(settled(), 2)
        assert not record.scores
        assert (await graph.state()).status is TaskStatus.CANCELLED
        with pytest.raises(AIError) as unavailable:
            await runtime.evaluations.get_report(saved.report_id, principal=PRINCIPAL)
        assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        await runtime.evaluations.purge_expired(principal=PRINCIPAL, now=clock.now(), exclusive=Offline())
        with pytest.raises(AIError):
            await storage.evaluation.records.get_report(saved.report_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("waiting", "running", "terminal"))
async def test_expired_launched_scoring_reaches_a_terminal_disposition_without_evidence(
    monkeypatch: pytest.MonkeyPatch, state: str,
) -> None:
    clock = Clock()
    monkeypatch.setattr(evaluation_module, "_now", clock.now)
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(context: TaskNodeContext[None]) -> JsonValue:
        entered.set()
        await release.wait()
        return await score(context)

    target, rule = Task("expiry.target", echo, effect_policy="none"), Task("expiry.score", hold, effect_policy="none")
    scorer = ScorerSpec("exact", TaskRef.deferred_input(), (DIMENSION,)) if state == "waiting" else rule_scorer(rule)
    async with Runtime.open("launched-expiry", models=FixtureModels(), storage=RuntimeStorage.in_memory(), context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target, rule)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("expiry", 1), (
            CaseSpec.task(CaseRef("expiry", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,),
            policy=EvaluationPolicy(allow_volatile=True, content_retention_seconds=60)), PRINCIPAL, "run"), engine=engine)

        async def ready():
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    native = await graph.state()
                    score_node = next(node for node in native.node_states if node.node_id == "score")
                    if score_node.status is (TaskStatus.WAITING if state == "waiting" else TaskStatus.RUNNING):
                        return graph
                await asyncio.sleep(0.01)

        graph = await asyncio.wait_for(ready(), 10)
        if state == "terminal":
            release.set()
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        clock.advance(61)

        async def no_expired_evidence(*args, **kwargs):
            raise AssertionError("expiry coordination must not read expired evidence")

        monkeypatch.setattr(runtime.evaluations, "read_evidence", no_expired_evidence)
        try:
            view = await run.wait(timeout_seconds=1)
            assert view.completion == ("complete" if state == "terminal" else "cancelled")
            assert view.progress.terminal_scores == view.progress.planned_scores == 1
            if state != "terminal":
                assert (await graph.state()).status is TaskStatus.CANCELLED
        finally:
            release.set()
