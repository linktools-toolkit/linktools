#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative case context and durable scoring submission boundaries."""

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import JsonValue, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    LabelProvenance, RescoreRequest, ScoreBundle, ScorerContract, ScoringInput,
    StartEvaluationRequest, TargetTrialRef,
)
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence._evaluation_records import EvaluationRecord
from linktools.ai.task import Task, TaskGraph, TaskGraphSubmission, TaskNodeContext, TaskSubmissionResult

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, echo, rule_scorer,
)


@pytest.mark.asyncio
async def test_program_scorer_uses_frozen_case_context_for_identical_inputs_and_rescore(tmp_path: Path) -> None:
    received: list[ScoringInput] = []

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        received.append(ScoringInput.from_mapping(context.input))
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    target = Task("context.target", echo, effect_policy="none")
    scorer = Task("context.score", score, effect_policy="none")
    untrusted = {
        "answer": "same answer", "case_ref": CaseRef("spoofed", "case", 99).to_mapping(),
        "label_provenance": LabelProvenance("spoofed", "intruder", "human-confirmed", "99").to_mapping(),
    }
    labels = (
        LabelProvenance("assessment:42", "reviewer", "human-confirmed", "3"),
        LabelProvenance("import", "author", "custom-import", "1"),
        LabelProvenance(),
    )
    cases = tuple(CaseSpec.task(CaseRef("context", f"case-{index}", 1), input=untrusted,
                               expected="same answer", label_provenance=label)
                  for index, label in enumerate(labels))
    async with (
        Persistence.open('scoring-context', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_51,
        TaskEngine.open(target, scorer, persistence=persistence_51, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("context", 1), cases),
            principal=PRINCIPAL, idempotency_key="dataset")
        spec = replace(rule_scorer(scorer), config={"policy": "constant"})
        run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (spec,)), PRINCIPAL, "start"))
        assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        assert len(received) == len(cases)
        revised = tuple(replace(case, ref=replace(case.ref, revision=2),
                                label_provenance=LabelProvenance("later-assessment")) for case in cases)
        await evaluations.publish_dataset(DatasetSpec(DatasetRef("context", 2), revised),
            principal=PRINCIPAL, idempotency_key="revised-dataset")
        rescored = await run.rescore(RescoreRequest((spec,), "rescore"))
        assert (await rescored.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        for samples in (received[:len(cases)], received[len(cases):]):
            assert {sample.case_ref: sample.label_provenance for sample in samples} == {
                case.ref: case.label_provenance for case in cases}
            for sample in samples:
                assert sample.config == {"policy": "constant"}
                assert sample.expected == sample.target_output == "same answer"
                assert sample.target_input == {"kind": "task_input", "input": untrusted, "dependencies": {}}
                assert sample.case_ref.revision == 1
                assert sample.label_provenance.source != "spoofed"
                assert sample.to_mapping()["label_provenance"] == sample.label_provenance.to_mapping()
        for selected_run in (run, rescored):
            first = await selected_run.current_trials(limit=2)
            second = await selected_run.current_trials(cursor=first.next_cursor, limit=2)
            current = (*first.items, *second.items)
            assert second.next_cursor is None
            assert {item.case_ref for item in current} == {case.ref for case in cases}
            assert await selected_run.read_trial(current[0].trial.trial_id) == current[0]
            frozen = await selected_run.trials(limit=3)
            assert tuple(item.trial for item in frozen.items) == tuple(item.trial for item in current)
            for item in current:
                case = await evaluations.get_case(item.case_ref, principal=PRINCIPAL)
                original = next(original for original in cases if original.ref == item.case_ref)
                assert case.ref == original.ref
                assert case.label_provenance == original.label_provenance
        assert len(received) == len(cases) * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("include_case_context", (False, True), ids=("without-case-context", "with-case-context"))
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_scoring_intent_replays_original_input_and_rejects_changed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_case_context: bool, backend: str,
) -> None:
    target_calls: list[JsonValue] = []
    received: list[ScoringInput] = []

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        target_calls.append(context.input["answer"])
        return context.input["answer"]

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        received.append(ScoringInput.from_mapping(context.input))
        return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()

    target = Task("replay.target", answer, effect_policy="none")
    scorer = Task("replay.score", score, effect_policy="none")
    case = CaseSpec.task(CaseRef("replay", "case", 1), input={"answer": "yes"}, expected="yes",
                         label_provenance=LabelProvenance("assessment:42", "reviewer", "human-confirmed", "3"))
    def storage() -> PersistenceStorage:
        return (PersistenceStorage.filesystem(tmp_path) if backend == "filesystem"
                else PersistenceStorage.sqlite(tmp_path / "state.sqlite"))

    initial = storage()
    async with (
        Persistence.open('scoring-replay', storage=initial, tenant_id='evaluation-consumers') as persistence_120,
        TaskEngine.open(target, scorer, persistence=persistence_120, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        build_graph = evaluations._scoring_graph
        captured: list[ScoringInput] = []

        def project(record: EvaluationRecord, trial: TargetTrialRef, contract: ScorerContract,
                    sample: ScoringInput) -> TaskGraph:
            captured.append(sample)
            payload = sample.to_mapping()
            if not include_case_context:
                payload.pop("case_ref")
                payload.pop("label_provenance")
            return build_graph(record, trial, contract, ScoringInput.from_mapping(payload))

        entered = asyncio.Event()
        start_prepared = TaskEngine.start_prepared

        async def interrupt(selected: TaskEngine, submission: TaskGraphSubmission) -> TaskSubmissionResult:
            if any(node.node_id == "score" for node in submission.graph.nodes):
                entered.set()
                await asyncio.Event().wait()
            return await start_prepared(selected, submission)

        with monkeypatch.context() as patch:
            patch.setattr(evaluations, "_scoring_graph", project)
            patch.setattr(TaskEngine, "start_prepared", interrupt)
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("replay", 1), (case,)),
                principal=PRINCIPAL, idempotency_key="dataset")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),)), PRINCIPAL, "start"))
            await asyncio.wait_for(entered.wait(), EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            experiment_id = run.experiment_id
            repository = persistence_120.evaluation.records
            record = await repository.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
            saved = next(intent for intent in record.intents if intent.scorer_slot_id is not None)
            assert not saved.confirmed and not saved.released
            original_bytes = canonical_json_bytes(encode_domain(saved.submission))
            assert (await repository.register_launch_intent(experiment_id, saved, capacity=1)) == record

            alternate = captured[0].to_mapping()
            if include_case_context:
                alternate.pop("case_ref")
                alternate.pop("label_provenance")
            graph = build_graph(record, saved.trial, record.manifest.scorers[0], ScoringInput.from_mapping(alternate))
            submission = await engine.with_definitions(evaluations._recorder).describe_submission(
                graph, principal=PRINCIPAL, idempotency_key=f"evaluation:{experiment_id}:{saved.slot_id}",
                limits=saved.submission.admission.limits, correlation=saved.submission.admission.correlation)
            assert submission.ref.operation_id == saved.submission.ref.operation_id
            assert submission.ref.graph_id == saved.submission.ref.graph_id
            assert submission.ref.request_digest != saved.submission.ref.request_digest
            with pytest.raises(AIError) as conflict:
                await repository.register_launch_intent(experiment_id, replace(saved, submission=submission), capacity=1)
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert await repository.get(experiment_id, tenant_id=PRINCIPAL.tenant_id) == record
            await evaluations.close()
        assert received == [] and target_calls == ["yes"]

    reopened = storage()
    async with (
        Persistence.open('scoring-replay', storage=reopened, tenant_id='evaluation-consumers') as persistence_180,
        TaskEngine.open(target, scorer, persistence=persistence_180, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        def reject_rebuild(*args: object, **kwargs: object) -> TaskGraph:
            pytest.fail("an existing scoring intent must replay its saved submission")

        monkeypatch.setattr(evaluations, "_scoring_graph", reject_rebuild)
        recovered = await evaluations.reconcile(experiment_id,
            principal=PRINCIPAL, idempotency_key="resume")
        assert (await recovered.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        assert target_calls == ["yes"] and len(received) == 1
        sample = received[0]
        assert sample.case_ref == (case.ref if include_case_context else None)
        assert sample.label_provenance == (case.label_provenance if include_case_context else None)
        record = await persistence_180.evaluation.records.get(experiment_id, tenant_id=PRINCIPAL.tenant_id)
        restored = next(intent for intent in record.intents if intent.scorer_slot_id is not None)
        assert canonical_json_bytes(encode_domain(restored.submission)) == original_bytes
        assert sample.to_mapping() == saved.submission.graph.nodes[0].input
