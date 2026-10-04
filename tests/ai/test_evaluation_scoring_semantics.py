#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoring evidence, human decisions and usage gates through native execution."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSlotRef, CandidateSpec, CaseRef, CaseSpec, ComparisonSpec, DatasetRef,
    DatasetSpec, EvaluationPolicy, EvaluationSpec, EvidencePolicy, GatePolicy,
    GraphTargetSpec, HumanScoreRequest, RescoreRequest, ScoreBundle, ScoreComparisonSelection,
    ScoreSelection, ScorerSpec, ScoringInput, StartEvaluationRequest, TaskCaseInput,
)
from linktools.ai.runtime import AgentTaskInput, AgentTaskInputContext, Runtime, RuntimeStorage, TaskEngine
from linktools.ai.task import (
    Task, TaskGraph, TaskGraphTemplate, TaskInputSupplyRequest, TaskNode, TaskNodeContext, TaskNodeResultRef, TaskRef,
)

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, DIMENSION, PRINCIPAL, FixtureModels, echo, rule_scorer


def capabilities() -> CapabilityGroup[None]:
    group = CapabilityGroup[None]("scoring-semantics")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize("include_input,judge_include_input", ((True, True), (False, False), (True, False)))
async def test_agent_input_evidence_reaches_rule_and_model_scorers_without_labels(
    tmp_path: Path, include_input: bool, judge_include_input: bool,
) -> None:
    models = FixtureModels()
    question = "The original question visible only when input is granted"
    expected = "Private reference answer"
    rubric = "Private scoring rubric"
    body = b"Do not expose binary bodies through input evidence"
    received: list[ScoringInput] = []
    async with Runtime.open("scoring-input", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        async def score(context: TaskNodeContext[None]) -> JsonValue:
            sample = ScoringInput.from_mapping(context.input)
            received.append(sample)
            evidence = await runtime.evaluations.read_evidence(sample.evidence_ref, principal=context.principal)
            projected = None if evidence.input is None else evidence.input.value
            assert (question in json.dumps(projected)) is include_input
            assert sample.to_mapping().get("target_input") == projected
            assert expected not in json.dumps(projected) and rubric not in json.dumps(projected)
            assert body.decode() not in json.dumps(projected)
            assert not evidence.attachments
            return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

        target = runtime.tasks.from_agent("semantics.target", runtime.agents.get())
        judge = runtime.tasks.from_agent("semantics.judge", runtime.agents.get())
        rule = Task("semantics.rule", score, effect_policy="none")
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("inputs", 1), (
            CaseSpec.agent(CaseRef("inputs", "one", 1), prompt=(question, BinaryContent(body, media_type="text/plain")),
                           expected=expected),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        scorers = tuple(replace(rule_scorer(task, slot=slot),
                               evidence_policy=EvidencePolicy(include_input=granted), rubric=rubric)
                        for task, slot, granted in ((rule, "rule", include_input),
                                                    (judge, "judge", judge_include_input)))
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), scorers,
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "start"),
            engine=runtime.tasks.bind(target, rule, judge))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        assert len(received) == 1
        scores = (await run.scores()).items
        assert all(item.status == "valid" for item in scores)
        assert expected not in models.prompts[0] and rubric not in models.prompts[0]
        judge_input = json.loads(models.prompts[-1].split("\nDATA\n", 1)[1])
        assert (question in json.dumps(judge_input.get("target_input"))) is judge_include_input
        assert judge_input["expected"] == expected and judge_input["rubric"] == rubric
        assert ScoringInput.from_mapping(judge_input).to_mapping() == judge_input
        if not include_input and not judge_include_input:
            with pytest.raises(AIError) as raised:
                await run.rescore(RescoreRequest((rule_scorer(rule),), "input-not-retained"),
                                  engine=runtime.tasks.bind(rule))
            assert raised.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE


@pytest.mark.asyncio
async def test_agent_evidence_uses_the_accepted_projected_prompt(tmp_path: Path) -> None:
    models = FixtureModels()
    received: list[JsonValue] = []

    async def project(context: AgentTaskInputContext) -> str:
        return "Prepared question: " + context.input["question"]

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        received.append(ScoringInput.from_mapping(context.input).target_input)
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    scorer = Task("semantics.projected-score", score, effect_policy="none")
    async with Runtime.open("projected-input", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        target = runtime.tasks.from_agent("semantics.projected-target", runtime.agents.get(), build_input=project)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("projected", 1), (
            CaseSpec.task(CaseRef("projected", "one", 1), input=dict(
                AgentTaskInput("Unprojected question", parameters={"question": "accepted input"}))),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),),
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "start"),
            engine=runtime.tasks.bind(target, scorer))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        assert received == [{"kind": "agent_input", "prompt": {"kind": "text", "text": models.prompts[0]}}]
        assert models.prompts == ["Prepared question: accepted input"]


@pytest.mark.asyncio
async def test_graph_evidence_retains_merged_task_inputs_and_dependency_values(tmp_path: Path) -> None:
    received: list[JsonValue] = []

    async def prepare(context: TaskNodeContext[None]) -> JsonValue:
        return context.input["question"].upper()

    async def finish(context: TaskNodeContext[None]) -> JsonValue:
        return await context.read_dependency("prepared")

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        received.append(ScoringInput.from_mapping(context.input).target_input)
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    first = Task("semantics.graph-prepare", prepare, effect_policy="none")
    second = Task("semantics.graph-finish", finish, effect_policy="none")
    scorer = Task("semantics.graph-score", score, effect_policy="none")
    template = TaskGraphTemplate((TaskNode("prepare", task=first, input={"fixed": "template data"}),
        TaskNode("finish", ("prepare",), task=second, input_refs={"prepared": TaskNodeResultRef("prepare")})))
    async with Runtime.open("graph-input", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("graph", 1), (
            CaseSpec.graph(CaseRef("graph", "one", 1), inputs={"prepare": TaskCaseInput(input={"question": "original"})}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", graph_template=GraphTargetSpec(template=template, outputs={"answer": "finish"})),),
            (rule_scorer(scorer),)), PRINCIPAL, "start"), engine=runtime.tasks.bind(first, second, scorer))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        assert len(received) == 1
        assert received[0]["inputs"]["prepare"]["input"] == {"fixed": "template data", "question": "original"}
        assert set(received[0]["inputs"]["finish"]["dependencies"]) == {"prepared"}
        assert received[0]["inputs"]["finish"]["dependencies"]["prepared"] == {
            "status": "succeeded", "value": "ORIGINAL", "reason": None,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("before_graph", "before_waiting"))
async def test_accepted_human_decision_is_consumed_after_deferred_input_becomes_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    target = Task("semantics.human-target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    entered, release = asyncio.Event(), asyncio.Event()
    async with Runtime.open("early-human", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        if stage == "before_waiting":
            original = runtime._execution_service.defer_task_input

            async def pause(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(runtime._execution_service, "defer_task_input", pause)
        else:
            original = TaskEngine.start_prepared

            async def pause_graph(engine, submission):
                if submission.graph.nodes[0].task == TaskRef.deferred_input():
                    entered.set()
                    await release.wait()
                return await original(engine, submission)

            monkeypatch.setattr(TaskEngine, "start_prepared", pause_graph)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("human", 1), (
            CaseSpec.task(CaseRef("human", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start"),
            engine=runtime.tasks.bind(target))
        await asyncio.wait_for(entered.wait(), 10)
        pending = (await run.scores()).items[0]
        request = HumanScoreRequest(pending.trial.trial_id, "human", pending.evidence_ref,
                                   ScoreBundle(dimensions={"exact_match": 1.0}), "decision")
        try:
            assert (await run.submit_human_score(request)).status == "pending"
            assert (await run.submit_human_score(request)).status == "pending"
            with pytest.raises(AIError) as raised:
                await run.submit_human_score(replace(request, score=ScoreBundle(dimensions={"exact_match": 0.0})))
            assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        finally:
            release.set()
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        accepted = (await run.scores()).items[0]
        assert accepted.status == "valid" and accepted.decision_id is not None
        assert (await run.submit_human_score(request)).decision_id == accepted.decision_id


class UsageModels(FixtureModels):
    def __init__(self, fail: bool) -> None:
        super().__init__()
        self.fail = fail

    def materialize(self) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            prompt = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)][-1]
            sample = json.loads(prompt.split("\nDATA\n", 1)[1])
            if self.fail and sample["expected"] == "bad":
                raise RuntimeError("Offline provider failed before reporting usage")
            yield {0: DeltaToolCall(name=info.output_tools[0].name,
                                   json_args=json.dumps({"dimensions": {"exact_match": 1.0}}))}
        return FunctionModel(stream_function=respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("graph", "execution"))
async def test_same_human_decision_handles_concurrent_native_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    target = Task("semantics.concurrent-human-target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    async with Runtime.open("concurrent-human", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("concurrent", 1), (
            CaseSpec.task(CaseRef("concurrent", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start"), engine=engine)

        async def ready():
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    state = await graph.state()
                    if next(node for node in state.node_states if node.node_id == "score").status is TaskStatus.WAITING:
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(ready(), 10)
        entered, completed = asyncio.Event(), asyncio.Event()
        calls = 0

        if boundary == "graph":
            launcher = runtime.tasks._graph_service._launcher
            original = launcher.supply_input

            async def interleave(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    await asyncio.wait_for(entered.wait(), 10)
                    try:
                        return await original(*args, **kwargs)
                    finally:
                        completed.set()
                entered.set()
                await completed.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(launcher, "supply_input", interleave)
        else:
            executions = runtime._execution_service._state.executions
            original = executions.commit_terminal
            second_returned = asyncio.Event()

            async def interleave_commit(commit, **kwargs):
                nonlocal calls
                if commit.execution.execution_id != pending.scorer_execution.execution_id:
                    return await original(commit, **kwargs)
                calls += 1
                if calls == 1:
                    await asyncio.wait_for(entered.wait(), 10)
                    result = await original(commit, **kwargs)
                    completed.set()
                    await second_returned.wait()
                    return result
                entered.set()
                await completed.wait()
                try:
                    return await original(commit, **kwargs)
                finally:
                    second_returned.set()

            monkeypatch.setattr(executions, "commit_terminal", interleave_commit)
        request = HumanScoreRequest(pending.trial.trial_id, "human", pending.evidence_ref,
                                   ScoreBundle(dimensions={"exact_match": 1.0}), "decision")
        await run.submit_human_score(request)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        accepted = (await run.scores()).items[0]
        assert accepted.status == "valid" and accepted.decision_id is not None
        assert (await run.submit_human_score(request)).decision_id == accepted.decision_id


@pytest.mark.asyncio
async def test_concurrent_deferred_inputs_do_not_accept_a_conflicting_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Runtime.open("conflicting-input", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        graph = await runtime.tasks.bind().start(TaskGraph("conflicting-score", (
            TaskNode("score", task=TaskRef.deferred_input(), output_type=ScoreBundle),
        )), principal=PRINCIPAL, idempotency_key="graph")

        async def ready():
            while True:
                state = (await graph.state()).node_states[0]
                if state.status is TaskStatus.WAITING:
                    return state.execution_id
                await asyncio.sleep(0.01)

        execution_id = await asyncio.wait_for(ready(), 10)
        executions = runtime._execution_service._state.executions
        original = executions.commit_terminal
        first_entered, both_entered, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = 0

        async def interleave(commit, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_entered.set()
                await asyncio.wait_for(both_entered.wait(), 10)
                result = await original(commit, **kwargs)
                committed.set()
                return result
            both_entered.set()
            await committed.wait()
            return await original(commit, **kwargs)

        monkeypatch.setattr(executions, "commit_terminal", interleave)
        accepted = ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()
        first = asyncio.create_task(graph.resume("score", TaskInputSupplyRequest(
            PRINCIPAL, execution_id, accepted, "first-value")))
        await asyncio.wait_for(first_entered.wait(), 10)
        try:
            with pytest.raises(AIError) as raised:
                await graph.resume("score", TaskInputSupplyRequest(PRINCIPAL, execution_id,
                    ScoreBundle(dimensions={"exact_match": 0.0}).to_mapping(), "other-value"))
            assert raised.value.code is ErrorCode.EXECUTION_RESULT_CONFLICT
        finally:
            await first
        assert await runtime.history.task_result("conflicting-score", "score", principal=PRINCIPAL) == accepted


@pytest.mark.asyncio
async def test_new_human_decision_rejects_native_cancellation_before_reconciliation(tmp_path: Path) -> None:
    target = Task("semantics.cancelled-human-target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    async with Runtime.open("cancelled-human", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(target)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("cancelled", 1), (
            CaseSpec.task(CaseRef("cancelled", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start"), engine=engine)

        async def waiting_score():
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    if any(node.node_id == "score" and node.status is TaskStatus.WAITING
                           for node in (await graph.state()).node_states):
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(waiting_score(), 10)
        experiment_id = run.experiment_id

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("cancelled-human", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        run = await runtime.evaluations.get(experiment_id, principal=PRINCIPAL)
        graph = await runtime.tasks.bind(target).get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
        assert (await graph.cancel(idempotency_key="native-cancel")).status is TaskStatus.CANCELLED
        with pytest.raises(AIError) as raised:
            await run.submit_human_score(HumanScoreRequest(pending.trial.trial_id, "human", pending.evidence_ref,
                ScoreBundle(dimensions={"exact_match": 1.0}), "too-late"))
        assert raised.value.code is ErrorCode.TASK_NOT_READY
        record = await storage.evaluation.records.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
        assert not record.human_decisions


@pytest.mark.asyncio
async def test_new_human_decision_is_rejected_after_its_slot_times_out(tmp_path: Path) -> None:
    target = Task("semantics.expired-human-target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    async with Runtime.open("expired-human", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("expired", 1), (
            CaseSpec.task(CaseRef("expired", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,),
            policy=EvaluationPolicy(human_timeout_seconds=0.2)), PRINCIPAL, "start"),
            engine=runtime.tasks.bind(target))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        expired = (await run.scores()).items[0]
        assert expired.status == "error" and expired.reason == "unanswered"
        with pytest.raises(AIError) as raised:
            await run.submit_human_score(HumanScoreRequest(expired.trial.trial_id, "human", expired.evidence_ref,
                ScoreBundle(dimensions={"exact_match": 1.0}), "too-late"))
        assert raised.value.code is ErrorCode.TASK_NOT_READY
        assert (await run.scores()).items[0] == expired


@pytest.mark.asyncio
@pytest.mark.parametrize(("score_only", "unknown_usage"), (
    pytest.param(False, False, id="initial-known"),
    pytest.param(False, True, marks=pytest.mark.merge, id="initial-unknown"),
    pytest.param(True, False, marks=pytest.mark.merge, id="rescore-known"),
    pytest.param(True, True, id="rescore-unknown"),
))
async def test_comparison_usage_requirement_accounts_for_scorer_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, score_only: bool, unknown_usage: bool,
) -> None:
    models = UsageModels(unknown_usage)
    target = Task("semantics.usage-target", echo, effect_policy="none")

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    rule = Task("semantics.initial-score", score, effect_policy="none")
    async with Runtime.open("scorer-usage", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT, capabilities=(capabilities(),)) as runtime:
        judge = runtime.tasks.from_agent("semantics.usage-judge", runtime.agents.get())
        engine = runtime.tasks.bind(target, rule, judge)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("usage", 1), tuple(
            CaseSpec.task(CaseRef("usage", answer, 1), input={"answer": answer}, expected=answer)
            for answer in (("good", "bad") if unknown_usage else ("good",))
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            tuple(CandidateSpec(slot, task=target.ref) for slot in ("baseline", "candidate")),
            (rule_scorer(rule if score_only else judge),),
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "start"), engine=engine)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        scoring = await run.rescore(RescoreRequest((rule_scorer(judge),), "rescore"), engine=engine) if score_only else run
        assert (await scoring.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        scorer_usage = [await runtime.history.graph_usage(item.scorer_graph.graph_id, principal=PRINCIPAL)
                        for item in (await scoring.scores()).items]
        assert any(item.unknown_usage_requests for item in scorer_usage) is unknown_usage
        selection = ScoreSelection("exact", "exact_match", scoring.experiment_id if score_only else None)
        spec = ComparisonSpec(CandidateSlotRef(run.experiment_id, "baseline"),
            CandidateSlotRef(run.experiment_id, "candidate"), (ScoreComparisonSelection(selection, selection),),
            gate_policy=GatePolicy(minimum_coverage=0.5, require_complete_usage=True))
        report = await runtime.evaluations.compare(spec, principal=PRINCIPAL)
        assert report.gate == ("inconclusive" if unknown_usage else "pass")
        assert ("unknown_required_usage" in report.gate_reasons) is unknown_usage
        assert await runtime.evaluations.get_report(report.report_id, principal=PRINCIPAL) == report
        async def changed_usage(*args, **kwargs):
            raise AssertionError("A fixed comparison cutoff cannot query current scorer usage")

        with monkeypatch.context() as patch:
            patch.setattr(runtime.history, "graph_usage", changed_usage)
            repeated = await runtime.evaluations.compare(replace(spec, cutoff=report.cutoff), principal=PRINCIPAL)
        assert repeated.gate == report.gate and repeated.cutoff == report.cutoff
        permissive = await runtime.evaluations.compare(replace(spec,
            gate_policy=replace(spec.gate_policy, require_complete_usage=False)), principal=PRINCIPAL)
        assert permissive.gate == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ("result", "cancel", "expire"))
async def test_budget_stops_new_scoring_without_cancelling_accepted_work(
    monkeypatch: pytest.MonkeyPatch, finish: str,
) -> None:
    from datetime import timedelta
    from linktools.ai.runtime import _evaluation as evaluation_module

    entered, release, budget_reached = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        entered.set()
        await release.wait()
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    async def observed_budget(*args: object) -> bool:
        return budget_reached.is_set()

    target, rule = Task("semantics.budget-target", echo, effect_policy="none"), Task(
        "semantics.budget-score", score, effect_policy="none")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("soft-budget", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        monkeypatch.setattr(runtime.evaluations, "_budget_exhausted", observed_budget)
        engine = runtime.tasks.bind(target, rule)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("budget", 1), tuple(
            CaseSpec.task(CaseRef("budget", name, 1), input={"answer": name}) for name in ("first", "second")
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (rule_scorer(rule),),
            policy=EvaluationPolicy(allow_volatile=True, scorer_concurrency=1, content_retention_seconds=60)),
            PRINCIPAL, "start"), engine=engine)
        await asyncio.wait_for(entered.wait(), 10)
        budget_reached.set()

        async def admission_stopped():
            while True:
                record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                if record.gate == "closed_budget" and any(
                    item.disposition.reason_code == "closed_budget" for item in record.dispositions
                ):
                    return record
                await asyncio.sleep(0.01)

        record = await asyncio.wait_for(admission_stopped(), 10)
        try:
            assert (await run.inspect()).completion == "running"
            pending = next(item for item in (await run.scores()).items if item.status == "pending")
            graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
            assert (await graph.state()).status is TaskStatus.RUNNING
            if finish == "cancel":
                await run.cancel(idempotency_key="cancel-after-budget")
            elif finish == "expire":
                monkeypatch.setattr(evaluation_module, "_now", lambda: record.content_expires_at + timedelta(seconds=1))
            else:
                release.set()
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "cancelled"
            record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
            if finish == "result":
                scores = (await run.scores()).items
                assert [item.status for item in scores] == ["valid", "not_attempted"]
                assert scores[1].reason == "closed_budget"
                assert (await graph.state()).status is TaskStatus.SUCCEEDED
            else:
                assert record.gate == "closed_cancel"
                assert (await graph.state()).status is TaskStatus.CANCELLED
        finally:
            release.set()


@pytest.mark.asyncio
async def test_budget_allows_reserved_human_score_to_complete_and_pass_a_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = asyncio.Event()

    async def observed_budget(*args: object) -> bool:
        return reached.is_set()

    target = Task("semantics.budget-human-target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("budget-human", models=FixtureModels(), storage=storage, context=CONTEXT) as runtime:
        monkeypatch.setattr(runtime.evaluations, "_budget_exhausted", observed_budget)
        engine = runtime.tasks.bind(target)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("human-budget", 1), (
            CaseSpec.task(CaseRef("human-budget", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,),
            policy=EvaluationPolicy(allow_volatile=True)), PRINCIPAL, "start"), engine=engine)

        async def waiting_score():
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    if any(node.node_id == "score" and node.status is TaskStatus.WAITING
                           for node in (await graph.state()).node_states):
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(waiting_score(), 10)
        reached.set()

        async def closed_budget():
            while True:
                record = await storage.evaluation.records.get(run.experiment_id, tenant_id=PRINCIPAL.tenant_id)
                if record.gate == "closed_budget":
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(closed_budget(), 10)
        await runtime.evaluations.reconcile(run.experiment_id, engine=engine, principal=PRINCIPAL,
                                             idempotency_key="resume-budget")
        request = HumanScoreRequest(pending.trial.trial_id, "human", pending.evidence_ref,
                                    ScoreBundle(dimensions={"exact_match": 1.0}), "accepted")
        await run.submit_human_score(request)
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).completion == "complete"
        assert (await run.submit_human_score(request)).status == "valid"
        report = await run.report()
        assert report.completion == "complete" and report.scores[0].coverage == 1
        selection = ScoreSelection("human", "exact_match")
        spec = ComparisonSpec(CandidateSlotRef(run.experiment_id, "candidate"),
            CandidateSlotRef(run.experiment_id, "candidate"), (ScoreComparisonSelection(selection, selection),),
            gate_policy=GatePolicy())
        comparison = await runtime.evaluations.compare(spec, principal=PRINCIPAL)
        assert comparison.gate == "pass"
        repeated = await runtime.evaluations.compare(replace(spec, cutoff=comparison.cutoff), principal=PRINCIPAL)
        assert repeated.gate == "pass" and repeated.cutoff == comparison.cutoff
