#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete planned denominators and strict paired evaluation statistics."""

import csv
import io
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import ExecutionStatus, TaskStatus, service_principal
from linktools.ai.evaluation import (
    CandidateContract, CaseContract, CaseRef, DatasetContract, DatasetRef,
    DimensionContract, EvaluationManifest, EvaluationPolicy, ExecutionSubjectRef,
    ScoreAttemptView, ScoreBundle, ScoreNotApplicable, ScorerContract, TargetTrialRef,
    TaskCaseInput, TrialPlan, TrialView,
)
from linktools.ai.evaluation._reports import (
    CandidateSlotRef, ComparisonReadCutoff, ComparisonSpec, DimensionBound,
    EvaluationReadCutoff, GatePolicy, ScoreComparisonSelection, ScoreSelection,
    build_comparison_report, build_evaluation_report, export_report,
)
from linktools.ai.task import TaskRef

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)
DATASET = DatasetRef("report", 1)
CASES = tuple(CaseContract(CaseRef(DATASET.id, name, 1), TaskCaseInput(input={"text": name}), weight=weight)
              for name, weight in (("a", 1), ("b", 3), ("c", 5)))
DATA = DatasetContract(DATASET, tuple(case.ref for case in CASES), "task_input")
SCORER = ScorerContract("quality", TaskRef("judge", 1), {"type": "function"},
                        (DimensionContract("quality", "points", "higher"),), {"kind": "json"})
SELECTION = ScoreComparisonSelection(ScoreSelection("quality", "quality"), ScoreSelection("quality", "quality"))


def _manifest(identity: str, *, scorer: ScorerContract = SCORER) -> EvaluationManifest:
    candidate = CandidateContract("model", TaskRef("target", 1), None,
                                  ({"id": "target", "revision": 1, "type": "function"},))
    return EvaluationManifest(identity, "experiment", None, DATASET, (candidate,), (scorer,),
        tuple(TrialPlan(f"{case.ref.case_id}-{repetition}", case.ref, "model", repetition)
              for case in CASES for repetition in (1, 2)), (), EvaluationPolicy(), "fixed_input",
        service_principal("tenant", "evaluation"))


def _trials(manifest: EvaluationManifest) -> tuple[TrialView, ...]:
    return tuple(TrialView(TargetTrialRef(manifest.experiment_id, plan.trial_id), plan.case_ref,
                           plan.candidate_slot_id, plan.repetition,
                           subject=ExecutionSubjectRef("test", "tenant", f"{manifest.experiment_id}-{plan.trial_id}"),
                           execution_status=ExecutionStatus.SUCCEEDED) for plan in manifest.trials)


def _scores(manifest: EvaluationManifest, values: tuple[float | None | ScoreNotApplicable, ...], *,
            scoring_id: str | None = None, status: str = "not_attempted") -> tuple[ScoreAttemptView, ...]:
    return tuple(ScoreAttemptView(scoring_id or manifest.experiment_id, f"score-{index}" if value is not None else None,
                                  TargetTrialRef(manifest.experiment_id, plan.trial_id), "quality", SCORER.task,
                                  "valid" if value is not None else status,
                                  ScoreBundle(dimensions={"quality": value}) if value is not None else None,
                                  reason="unavailable" if value is None else None)
                 for index, (plan, value) in enumerate(zip(manifest.trials, values)))


def _cutoff(manifest: EvaluationManifest) -> EvaluationReadCutoff:
    return EvaluationReadCutoff(manifest.experiment_id, manifest.digest, {"experiment": 1})


def _compare(*, left: EvaluationManifest | None = None, right: EvaluationManifest | None = None,
             left_values: tuple = (10, 20, 40, 50, None, None),
             right_values: tuple = (10, 22, 50, None, None, None),
             policy: GatePolicy | None = None, **kwargs: object):
    left, right = left or _manifest("base"), right or _manifest("candidate")
    spec = kwargs.pop("spec", ComparisonSpec(CandidateSlotRef(left.experiment_id, "model"),
        CandidateSlotRef(right.experiment_id, "model"), (SELECTION,), gate_policy=policy))
    defaults = {"trials": _trials(left) + _trials(right), "scores": _scores(left, left_values) + _scores(right, right_values),
                "completions": {left.experiment_id: "complete", right.experiment_id: "complete"},
                "baseline_cases": CASES, "candidate_cases": CASES,
                "baseline_dataset": DATA, "candidate_dataset": DATA,
                "cutoff": ComparisonReadCutoff(_cutoff(left), _cutoff(right), tuple(_cutoff(item) for item in kwargs.get("scoring_manifests", ()))), "report_id": "comparison", "created_at": NOW}
    defaults.update(kwargs)
    return build_comparison_report(spec, left, right, **defaults)


def test_pair_statistics_weight_cases_after_matched_repetitions() -> None:
    report = _compare()
    summary, = report.dimensions
    assert report.compatibility == "compatible"
    assert summary.planned_pairs == 6
    assert (summary.complete_pairs, summary.baseline_missing, summary.candidate_missing, summary.both_missing,
            summary.not_comparable) == (3, 0, 1, 2, 0)
    assert summary.mean_difference == pytest.approx(7.75)
    assert summary.baseline_mean == pytest.approx(33.75)
    assert summary.candidate_mean == pytest.approx(41.5)
    assert summary.complete_cases == 2
    assert summary.weight_sum == 4
    assert report.gate == "not_configured"


def test_reports_preserve_missing_failed_cancelled_and_never_started_slots() -> None:
    manifest = _manifest("run")
    trials = _trials(manifest)
    trials = (trials[0], replace(trials[1], execution_status=ExecutionStatus.FAILED, error_code="target_error"),
              replace(trials[2], execution_status=ExecutionStatus.CANCELLED),
              replace(trials[3], subject=None, execution_status=None))
    scores = _scores(manifest, (0, ScoreNotApplicable(reason="no reference"), None, None), status="error")
    report = build_evaluation_report(manifest, CASES, trials, scores,
                                     cutoff=_cutoff(manifest), report_id="report", created_at=NOW)
    candidate, = report.candidates
    assert (candidate.planned, candidate.started, candidate.succeeded, candidate.failed, candidate.cancelled) == (6, 3, 1, 1, 1)
    assert candidate.scoring_complete_trials == 1
    summary, = report.scores
    assert (summary.planned, summary.valid, summary.not_applicable, summary.error, summary.not_attempted, summary.pending) == (6, 1, 1, 2, 2, 0)
    assert summary.mean == 0
    assert summary.coverage == pytest.approx(1 / 6)
    assert {item.code: item.count for item in summary.failure_counts} == {"no reference": 1, "unavailable": 2, "score_missing": 2}
    assert {item.code: item.count for item in report.failures} == {"target_error": 1}


def test_terminal_missing_can_pass_an_explicit_reduced_coverage_gate() -> None:
    policy = GatePolicy(minimum_coverage=0.5, bounds=(DimensionBound(SELECTION, minimum=40, max_regression=0),))
    assert _compare(policy=policy).gate == "pass"
    assert _compare(policy=GatePolicy()).gate == "inconclusive"
    assert _compare(policy=replace(policy, minimum_cases=3)).gate == "inconclusive"
    assert _compare(policy=replace(policy, minimum_coverage=0), left_values=(None,) * 6, right_values=(None,) * 6).gate == "inconclusive"


@pytest.mark.parametrize(("changes", "reason"), [
    ({"completions": {"base": "complete", "candidate": "running"}}, "pending_or_recovery_required"),
    ({"completions": {"base": "complete", "candidate": "cancelled"}}, "cancelled"),
    ({"usage_complete": None}, "unknown_required_usage"),
])
def test_gate_incomplete_precedence_is_not_relaxed_by_coverage(changes: dict, reason: str) -> None:
    report = _compare(policy=GatePolicy(minimum_coverage=0.1, require_complete_usage=True), **changes)
    assert report.gate == "inconclusive"
    assert reason in report.gate_reasons


@pytest.mark.parametrize("usage_complete", (True, False, None))
@pytest.mark.parametrize("required", (False, True))
def test_usage_completeness_gate_requires_positive_confirmation_only_when_requested(
    usage_complete: bool | None, required: bool,
) -> None:
    report = _compare(
        policy=GatePolicy(minimum_coverage=0.1, require_complete_usage=required),
        usage_complete=usage_complete,
    )
    blocked = required and usage_complete is not True
    assert report.gate == ("inconclusive" if blocked else "pass")
    assert ("unknown_required_usage" in report.gate_reasons) is blocked


def test_pending_scores_and_recovery_required_prevent_a_gate_pass() -> None:
    left, right = _manifest("base"), _manifest("candidate")
    scores = _scores(left, (10,) * 6) + _scores(right, (11, None, None, None, None, None), status="pending")
    report = _compare(left=left, right=right, scores=scores, policy=GatePolicy(minimum_coverage=0.1))
    assert report.gate_reasons == ("pending_or_recovery_required",)
    trials = list(_trials(left) + _trials(right))
    trials[0] = replace(trials[0], execution_status=ExecutionStatus.RECOVERY_REQUIRED)
    assert _compare(trials=trials, policy=GatePolicy(minimum_coverage=0.1)).gate == "inconclusive"


def test_gate_regression_uses_dimension_direction() -> None:
    scorer = replace(SCORER, dimensions=(DimensionContract("quality", "points", "lower"),))
    report = _compare(left=_manifest("base", scorer=scorer), right=_manifest("candidate", scorer=scorer),
                      policy=GatePolicy(minimum_coverage=0.5, bounds=(DimensionBound(SELECTION, max_regression=1),)))
    assert report.gate == "fail"
    assert report.gate_reasons == ("quality:regression",)


def test_rescore_selection_keeps_original_pair_denominator_and_never_selects_latest() -> None:
    left, right = _manifest("base"), _manifest("candidate")
    left_source = replace(left, experiment_id="rescore-base", kind="score_only", source_experiment_id="base", trials=(),
                          source_trials=(TargetTrialRef("base", "a-1"),))
    right_source = replace(right, experiment_id="rescore-candidate", kind="score_only", source_experiment_id="candidate", trials=(),
                           source_trials=(TargetTrialRef("candidate", "a-1"),))
    scores = _scores(left, (100,), scoring_id="rescore-base") + _scores(right, (110,), scoring_id="rescore-candidate")
    initial = _compare(left=left, right=right, scores=scores, scoring_manifests=(left_source, right_source))
    assert initial.dimensions[0].complete_pairs == 0
    selection = ScoreComparisonSelection(ScoreSelection("quality", "quality", "rescore-base"),
                                         ScoreSelection("quality", "quality", "rescore-candidate"))
    spec = ComparisonSpec(CandidateSlotRef("base", "model"), CandidateSlotRef("candidate", "model"), (selection,),
                          gate_policy=GatePolicy(minimum_coverage=0.1))
    report = _compare(left=left, right=right, scores=scores, scoring_manifests=(left_source, right_source), spec=spec)
    assert report.compatibility == "compatible"
    assert report.dimensions[0].planned_pairs == 6
    assert report.dimensions[0].complete_pairs == 1
    assert report.dimensions[0].both_missing == 5
    assert report.dimensions[0].mean_difference == 10
    assert report.gate == "pass"
    assert _compare(left=left, right=right, scores=scores, scoring_manifests=(right_source,), spec=spec).compatibility == "incompatible"


def test_incompatible_scorer_contract_never_drops_planned_pairs() -> None:
    right = _manifest("candidate", scorer=replace(SCORER, config={"rubric_version": 2}))
    report = _compare(right=right, policy=GatePolicy(minimum_coverage=0))
    assert report.compatibility == "incompatible"
    assert report.dimensions[0].not_comparable == report.dimensions[0].planned_pairs == 6
    assert report.dimensions[0].mean_difference is None
    assert report.gate_reasons == ("incompatible",)
    assert any(item.path == "scores.0.contract" for item in report.differences)


def test_named_contract_drift_cannot_be_waived_as_candidate_change() -> None:
    right = _manifest("candidate")
    changed = replace(right.candidates[0], definition_contracts=({"id": "target", "revision": 1, "type": "different"},))
    right = replace(right, candidates=(changed,))
    spec = ComparisonSpec(CandidateSlotRef("base", "model"), CandidateSlotRef("candidate", "model"), (SELECTION,),
                          allowed_changes=("task_definition",))
    report = _compare(right=right, spec=spec)
    assert report.compatibility == "incompatible"
    assert any(item.path == "candidate.contract_drift" and not item.allowed for item in report.differences)
    changed = replace(changed, task=TaskRef("target", 2),
                      definition_contracts=({"id": "target", "revision": 2, "type": "different"},))
    assert _compare(right=replace(right, candidates=(changed,)), spec=spec).compatibility == "compatible"


def test_snapshot_cutoff_and_exports_preserve_zero_missing_and_na() -> None:
    revisions = {"experiment": 1}
    left = _manifest("base")
    cutoff = EvaluationReadCutoff("base", left.digest, revisions)
    revisions["experiment"] = 9
    assert cutoff.state_revisions["experiment"] == 1
    report = _compare(left_values=(0, ScoreNotApplicable(reason="not relevant"), None, None, None, None),
                      right_values=(0, 1, None, None, None, None))
    rows = list(csv.DictReader(io.StringIO(export_report(report, format="csv"))))
    assert rows[0]["baseline_value"] == "0.0"
    assert rows[1]["baseline_value"] == "null"
    assert rows[1]["baseline_reason"] == "not relevant"
    assert json.loads(export_report(report))["dimensions"][0]["complete_pairs"] == 1
    assert "1 / 6" in export_report(report, format="markdown")
    summary = build_evaluation_report(left, CASES, _trials(left), _scores(left, (0,)),
                                     cutoff=cutoff, report_id="report", created_at=NOW)
    assert json.loads(export_report(summary))["score_attempts"][0]["score"]["dimensions"]["quality"] == 0
    with pytest.raises(ValueError, match="cutoff"):
        build_evaluation_report(left, CASES, (), (), cutoff=replace(cutoff, manifest_digest="wrong"),
                                report_id="report", created_at=NOW)


def test_full_report_materializes_missing_slots_and_excludes_other_scoring_sources() -> None:
    manifest = _manifest("run")
    unrelated = _scores(manifest, (99,), scoring_id="another-score-run")
    report = build_evaluation_report(manifest, CASES, (), unrelated,
                                     cutoff=_cutoff(manifest), report_id="report", created_at=NOW)
    assert len(report.trials) == len(report.score_attempts) == 6
    assert all(item.scoring_experiment_id == "run" and item.score_attempt_id is None for item in report.score_attempts)
    assert report.scores[0].not_attempted == report.scores[0].planned == 6
    rows = list(csv.DictReader(io.StringIO(export_report(report, format="csv"))))
    assert len(rows) == 6
    assert {item["status"] for item in rows} == {"not_attempted"}


def test_score_only_report_denominator_is_its_selected_source_trials() -> None:
    target = _manifest("source")
    manifest = replace(target, experiment_id="rescore", kind="score_only", source_experiment_id="source", trials=(),
                       source_trials=(TargetTrialRef("source", "a-1"), TargetTrialRef("source", "b-1")))
    scores = _scores(target, (1, 999, 3), scoring_id="rescore")
    report = build_evaluation_report(manifest, CASES, _trials(target), scores,
                                     cutoff=_cutoff(manifest), report_id="report", created_at=NOW)
    assert report.scores[0].planned == 2
    assert report.scores[0].mean == pytest.approx(2.5)
    assert tuple(item.trial for item in report.trials) == manifest.source_trials
    assert tuple(item.trial for item in report.score_attempts) == manifest.source_trials


def test_trial_failure_threshold_uses_candidate_planned_trials() -> None:
    left, right = _manifest("base"), _manifest("candidate")
    trials = tuple(replace(item, execution_status=ExecutionStatus.FAILED) for item in _trials(left)) + _trials(right)
    report = _compare(left=left, right=right, trials=trials, policy=GatePolicy(minimum_coverage=0.5, maximum_failure_rate=0))
    assert report.gate == "pass"
    trials = _trials(left) + tuple(replace(item, execution_status=ExecutionStatus.FAILED) for item in _trials(right))
    report = _compare(left=left, right=right, trials=trials, policy=GatePolicy(minimum_coverage=0.5, maximum_failure_rate=0.9))
    assert report.gate == "fail"
    assert report.gate_reasons == ("maximum_failure_rate_exceeded",)


def test_rescore_cannot_claim_unselected_trial_results() -> None:
    left, right = _manifest("base"), _manifest("candidate")
    source = replace(left, experiment_id="rescore", kind="score_only", source_experiment_id="base", trials=(),
                     source_trials=(TargetTrialRef("base", "a-1"),))
    selection = ScoreComparisonSelection(ScoreSelection("quality", "quality", "rescore"), SELECTION.candidate)
    spec = ComparisonSpec(CandidateSlotRef("base", "model"), CandidateSlotRef("candidate", "model"), (selection,))
    report = _compare(left=left, right=right, scores=_scores(left, (1, 2), scoring_id="rescore") + _scores(right, (1, 2)),
                      scoring_manifests=(source,), spec=spec)
    assert report.compatibility == "incompatible"
    assert any(item.path == "scores.0.baseline.binding" for item in report.differences)
    with pytest.raises(ValueError, match="scoring manifest"):
        _compare(left=left, right=right, scoring_manifests=(source,), spec=spec,
                 cutoff=ComparisonReadCutoff(_cutoff(left), _cutoff(right)))


def test_contract_difference_values_are_deeply_immutable() -> None:
    right = _manifest("candidate", scorer=replace(SCORER, config={"nested": {"list": [1]}}))
    report = _compare(right=right)
    difference = next(item for item in report.differences if item.path == "scores.0.contract")
    with pytest.raises(TypeError):
        difference.candidate["config"]["nested"]["list"][0] = 2
    assert json.loads(export_report(report))["differences"][0]["candidate"]["config"]["nested"]["list"] == [1]


def test_saved_reports_round_trip_current_durable_codec() -> None:
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain

    target = _manifest("run")
    evaluation = build_evaluation_report(target, CASES, _trials(target),
        _scores(target, (0, ScoreNotApplicable(reason="no reference"), None, None, None, None)),
        cutoff=_cutoff(target), report_id="report", created_at=NOW)
    comparison = _compare(policy=GatePolicy(minimum_coverage=0, bounds=(DimensionBound(SELECTION, minimum=0, max_regression=1),)))
    incompatible = _compare(right=_manifest("candidate", scorer=replace(SCORER, config={"nested": [1, None]})))
    for report in (evaluation, comparison, incompatible):
        restored = decode_domain(encode_domain(report), type(report))
        assert restored == report
        assert export_report(restored) == export_report(report)


def test_allowed_model_change_does_not_allow_prompt_or_environment_changes() -> None:
    from linktools.ai.agent import AgentBindingContract
    from linktools.ai.spec import AgentSpec

    def candidate(revision: int, model: str, prompt: str = "Answer") -> CandidateContract:
        binding = AgentBindingContract(AgentSpec("agent", model=model, system_prompt=prompt, revision=revision),
            {"model_identity": model}, (), (), "text", {"type": "string"})
        return CandidateContract("model", TaskRef("target", revision), None,
            ({"id": "target", "revision": revision, "type": "agent", "config": {
                "binding_contract": binding.to_payload(), "agent_id": "agent", "agent_revision": revision}},))

    left = replace(_manifest("base"), candidates=(candidate(1, "old"),))
    right = replace(_manifest("candidate"), candidates=(candidate(2, "new"),))
    spec = ComparisonSpec(CandidateSlotRef("base", "model"), CandidateSlotRef("candidate", "model"), (SELECTION,),
                          allowed_changes=("task_definition", "model_binding"))
    assert _compare(left=left, right=right, spec=spec).compatibility == "compatible"
    changed_prompt = replace(right, candidates=(candidate(2, "new", "Different answer"),))
    report = _compare(left=left, right=changed_prompt, spec=spec)
    assert any(item.path == "agent_instructions" and not item.allowed for item in report.differences)
    assert report.compatibility == "incompatible"
    assert _compare(left=left, right=replace(right, policy=EvaluationPolicy(environment={"fixture": "changed"})),
                    spec=spec).compatibility == "incompatible"


def test_identical_graph_templates_compare_by_content_and_preserve_scope() -> None:
    from linktools.ai.evaluation import GraphTargetContract
    from linktools.ai.task import TaskGraphLimits, TaskGraphTemplate, TaskNode

    left = _manifest("base")
    graph = TaskGraphTemplate((TaskNode("target", task=left.candidates[0].task),), limits=TaskGraphLimits())
    template = GraphTargetContract(graph, "test", "tenant", {"answer": "target"}, None, graph.limits)
    candidate = replace(left.candidates[0], task=None, graph_template=template)
    left = replace(left, candidates=(candidate,))
    other_capture = replace(template, template=replace(graph))
    right = replace(_manifest("candidate"), candidates=(replace(candidate, graph_template=other_capture),))
    report = _compare(left=left, right=right, policy=GatePolicy(minimum_coverage=0.5))
    assert report.compatibility == "compatible"
    assert report.gate == "pass"
    assert report.differences == ()
    for different in (
        replace(other_capture, template=replace(graph, nodes=(TaskNode("target", task=graph.nodes[0].task, input={"changed": "PRIVATE GRAPH INPUT"}),))),
        replace(other_capture, outputs={"different": "target"}),
        replace(other_capture, outputs={}, selector="terminal_sinks"),
        replace(other_capture, template=replace(graph, limits=TaskGraphLimits(max_concurrency=1)),
                limits=TaskGraphLimits(max_concurrency=1)),
    ):
        changed = replace(right, candidates=(replace(candidate, graph_template=different),))
        report = _compare(left=left, right=changed)
        assert report.compatibility == "incompatible"
        assert any(item.path == "graph_definition" and not item.allowed for item in report.differences)
        assert "PRIVATE GRAPH INPUT" not in export_report(report)
    allowed = ComparisonSpec(CandidateSlotRef("base", "model"), CandidateSlotRef("candidate", "model"),
                             (SELECTION,), allowed_changes=("graph_definition",))
    for scope in ({"namespace": "another"}, {"tenant_id": "another"}):
        wrong_scope = replace(other_capture, **scope)
        changed = replace(right, candidates=(replace(candidate, graph_template=wrong_scope),))
        report = _compare(left=left, right=changed, spec=allowed)
        assert report.compatibility == "incompatible"
        assert any(item.path == "graph_scope" and not item.allowed for item in report.differences)


def test_report_snapshot_keeps_pending_completion_after_current_work_finishes() -> None:
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain

    manifest = _manifest("run")
    running_trials = [replace(trial, execution_status=ExecutionStatus.STARTED) for trial in _trials(manifest)]
    pending_scores = list(_scores(manifest, (None,) * 6, status="pending"))
    first = build_evaluation_report(manifest, CASES, running_trials, pending_scores,
                                    cutoff=_cutoff(manifest), report_id="pending", created_at=NOW)
    saved = encode_domain(first)
    running_trials[:] = _trials(manifest)
    pending_scores[:] = _scores(manifest, (1,) * 6)
    current = build_evaluation_report(manifest, CASES, running_trials, pending_scores,
        cutoff=replace(_cutoff(manifest), state_revisions={"experiment": 2}), report_id="complete", created_at=NOW)
    historical = decode_domain(saved, type(first))
    assert historical.completion == first.completion == "running"
    assert current.completion == "complete"
    assert all(item.status == "pending" for item in historical.score_attempts)
    assert all(item.execution_status == ExecutionStatus.STARTED for item in historical.trials)
    spec = ComparisonSpec(CandidateSlotRef("run", "model"), CandidateSlotRef("run", "model"),
                          (SELECTION,), gate_policy=GatePolicy(minimum_coverage=0))
    for source, expected_gate in ((historical, "inconclusive"), (current, "pass")):
        comparison = build_comparison_report(spec, manifest, manifest, CASES, CASES,
            source.trials, source.score_attempts, baseline_dataset=DATA, candidate_dataset=DATA,
            cutoff=ComparisonReadCutoff(source.cutoff, source.cutoff), report_id="comparison", created_at=NOW,
            completions={"run": source.completion})
        assert comparison.gate == expected_gate


def test_blocked_native_trial_is_terminal_failure_not_pending() -> None:
    manifest = _manifest("blocked")
    trials = tuple(replace(trial, execution_status=TaskStatus.BLOCKED) for trial in _trials(manifest))
    scores = _scores(manifest, (1.0,) * len(trials))
    report = build_evaluation_report(manifest, CASES, trials, scores, cutoff=_cutoff(manifest),
                                     report_id="blocked-report", created_at=NOW)
    assert report.completion == "complete"
    assert report.candidates[0].failed == len(trials)
    assert report.candidates[0].scoring_complete_trials == len(trials)
    baseline = _manifest("baseline")
    comparison = _compare(left=baseline, right=manifest,
        trials=_trials(baseline) + trials, left_values=(1.0,) * len(trials),
        right_values=(1.0,) * len(trials), policy=GatePolicy(maximum_failure_rate=0))
    assert comparison.gate == "fail"
    assert "pending_or_recovery_required" not in comparison.gate_reasons


def test_recoverable_disposition_requires_attention_in_pure_report() -> None:
    from linktools.ai.evaluation import SlotDispositionView

    manifest = _manifest("recoverable")
    trials = tuple(replace(trial, execution_status=None, disposition=SlotDispositionView(
        "recoverable_blocked", "temporarily_unavailable", False, True, NOW,
    )) for trial in _trials(manifest))
    report = build_evaluation_report(manifest, CASES, trials,
        _scores(manifest, (None,) * len(trials), status="pending"), cutoff=_cutoff(manifest),
        report_id="recoverable-report", created_at=NOW)
    assert report.completion == "needs_attention"


def test_successful_trial_with_graph_recovery_blocks_reports_and_comparison() -> None:
    left, right = _manifest("base"), _manifest("candidate")
    trials = tuple(replace(trial, graph_status=TaskStatus.RECOVERY_REQUIRED) for trial in _trials(right))
    report = build_evaluation_report(right, CASES, trials, _scores(right, (1,) * len(trials)),
        cutoff=_cutoff(right), report_id="graph-recovery", created_at=NOW)
    assert report.completion == "needs_attention"
    assert report.candidates[0].succeeded == len(trials)
    comparison = _compare(left=left, right=right, trials=_trials(left) + trials,
        policy=GatePolicy(minimum_coverage=0.1))
    assert comparison.gate == "inconclusive"
    assert "pending_or_recovery_required" in comparison.gate_reasons
