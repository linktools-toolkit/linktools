#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed evaluation summaries, paired comparisons and portable report exports."""

import csv
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, cast

from ..core import JsonValue, normalize_json_value
from ._contracts import (
    CandidateContract, CaseContract, CaseRef, DatasetContract, DatasetRef,
    EvaluationManifest, ScorerContract, TargetTrialRef,
)
from ._evidence import EvidenceRef, ScoreBundle, ScoreNotApplicable
from ._views import ScoreAttemptView, TrialView


@dataclass(frozen=True, slots=True)
class CandidateSlotRef:
    experiment_id: str
    slot_id: str


@dataclass(frozen=True, slots=True)
class ScoreSelection:
    scorer_slot_id: str
    dimension: str
    scoring_experiment_id: str | None = None
    decision_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScoreComparisonSelection:
    baseline: ScoreSelection
    candidate: ScoreSelection


@dataclass(frozen=True, slots=True)
class DimensionBound:
    selection: ScoreComparisonSelection
    minimum: float | None = None
    maximum: float | None = None
    max_regression: float | None = None

    def __post_init__(self) -> None:
        for value in (self.minimum, self.maximum, self.max_regression):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("gate bounds must be finite")
        for name in ("minimum", "maximum", "max_regression"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("gate bounds are inverted")
        if self.max_regression is not None and self.max_regression < 0:
            raise ValueError("maximum regression cannot be negative")


@dataclass(frozen=True, slots=True)
class GatePolicy:
    minimum_coverage: float = 1.0
    minimum_cases: int = 1
    bounds: tuple[DimensionBound, ...] = ()
    maximum_failure_rate: float | None = None
    require_complete_usage: bool = False

    def __post_init__(self) -> None:
        for value in (self.minimum_coverage, self.maximum_failure_rate):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("gate rates must be between zero and one")
        if isinstance(self.minimum_cases, bool) or not isinstance(self.minimum_cases, int) or self.minimum_cases < 1:
            raise ValueError("minimum cases must be a positive integer")
        object.__setattr__(self, "minimum_coverage", float(self.minimum_coverage))
        if self.maximum_failure_rate is not None:
            object.__setattr__(self, "maximum_failure_rate", float(self.maximum_failure_rate))
        object.__setattr__(self, "bounds", tuple(self.bounds))
        if len({item.selection for item in self.bounds}) != len(self.bounds):
            raise ValueError("gate contains duplicate dimension bounds")


@dataclass(frozen=True, slots=True)
class EvaluationReadCutoff:
    experiment_id: str
    manifest_digest: str
    state_revisions: Mapping[str, int] = field(default_factory=dict)
    score_selections: tuple[ScoreSelection, ...] = ()
    source_evidence_refs: tuple[EvidenceRef, ...] = ()
    target_usage_complete: Mapping[str, bool] = field(default_factory=dict)
    scorer_usage_complete: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.target_usage_complete, *self.scorer_usage_complete.values())
        if any(not isinstance(name, str) or not name for name in self.scorer_usage_complete) or any(
            not isinstance(name, str) or not name or not isinstance(complete, bool)
            for value in values for name, complete in value.items()
        ):
            raise ValueError("usage completeness requires named boolean observations")
        object.__setattr__(self, "state_revisions", MappingProxyType(dict(self.state_revisions)))
        object.__setattr__(self, "score_selections", tuple(self.score_selections))
        object.__setattr__(self, "source_evidence_refs", tuple(self.source_evidence_refs))
        object.__setattr__(self, "target_usage_complete", MappingProxyType(dict(self.target_usage_complete)))
        object.__setattr__(self, "scorer_usage_complete", MappingProxyType({
            name: MappingProxyType(dict(values)) for name, values in self.scorer_usage_complete.items()}))

    @property
    def usage_complete(self) -> bool:
        return bool(self.target_usage_complete) and all(self.target_usage_complete.values()) and all(
            complete for values in self.scorer_usage_complete.values() for complete in values.values())


@dataclass(frozen=True, slots=True)
class ComparisonReadCutoff:
    baseline: EvaluationReadCutoff
    candidate: EvaluationReadCutoff
    scoring: tuple[EvaluationReadCutoff, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scoring", tuple(self.scoring))


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    baseline: CandidateSlotRef
    candidate: CandidateSlotRef
    scores: tuple[ScoreComparisonSelection, ...]
    mode: Literal["strict", "exploratory"] = "strict"
    allowed_changes: tuple[str, ...] = ()
    gate_policy: GatePolicy | None = None
    cutoff: ComparisonReadCutoff | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("strict", "exploratory"):
            raise ValueError("unknown comparison mode")
        allowed = {"model_binding", "agent_instructions", "capabilities", "task_definition",
                   "graph_definition", "candidate_parameters"}
        if not set(self.allowed_changes) <= allowed:
            raise ValueError("unknown allowed candidate change")
        if not self.scores or len(set(self.scores)) != len(self.scores):
            raise ValueError("comparison requires unique score selections")
        if self.gate_policy is not None and any(bound.selection not in self.scores for bound in self.gate_policy.bounds):
            raise ValueError("gate bound must select a compared dimension")
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "allowed_changes", tuple(self.allowed_changes))


@dataclass(frozen=True, slots=True)
class ContractDifference:
    path: str
    baseline: JsonValue
    candidate: JsonValue
    allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline", _freeze_json(self.baseline))
        object.__setattr__(self, "candidate", _freeze_json(self.candidate))


class _FrozenList(list):
    def _readonly(self, *args: object, **kwargs: object) -> None:
        raise TypeError("report values are immutable")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _readonly
    append = extend = insert = pop = remove = clear = reverse = sort = _readonly


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return cast(JsonValue, MappingProxyType({key: _freeze_json(item) for key, item in value.items()}))
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FailureCount:
    code: str
    count: int


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate_slot_id: str
    planned: int
    started: int
    succeeded: int
    failed: int
    cancelled: int
    unavailable: int
    scoring_complete_trials: int


@dataclass(frozen=True, slots=True)
class ScorerSummary:
    scorer_slot_id: str
    dimension: str
    planned: int
    pending: int
    valid: int
    not_applicable: int
    error: int
    not_attempted: int
    mean: float | None
    weight_sum: float
    coverage: float
    failure_counts: tuple[FailureCount, ...]
    candidate_slot_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("mean", "weight_sum", "coverage"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "failure_counts", tuple(self.failure_counts))


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    report_id: str
    experiment_id: str
    kind: Literal["experiment", "score_only"]
    source_experiment_id: str | None
    dataset: DatasetRef
    candidates: tuple[CandidateSummary, ...]
    scores: tuple[ScorerSummary, ...]
    failures: tuple[FailureCount, ...]
    cutoff: EvaluationReadCutoff
    created_at: datetime
    trials: tuple[TrialView, ...] = ()
    score_attempts: tuple[ScoreAttemptView, ...] = ()
    completion: Literal["running", "cancelling", "complete", "cancelled", "needs_attention"] = "complete"

    def __post_init__(self) -> None:
        for name in ("candidates", "scores", "failures", "trials", "score_attempts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PairedScoreRow:
    selection: ScoreComparisonSelection
    case_ref: CaseRef
    repetition: int
    baseline_trial: TargetTrialRef | None
    candidate_trial: TargetTrialRef | None
    status: Literal["complete", "baseline_missing", "candidate_missing", "both_missing", "not_comparable"]
    baseline_value: float | None
    candidate_value: float | None
    baseline_reason: str | None
    candidate_reason: str | None
    weight: float

    def __post_init__(self) -> None:
        for name in ("baseline_value", "candidate_value", "weight"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class PairedDimensionSummary:
    selection: ScoreComparisonSelection
    planned_pairs: int
    complete_pairs: int
    baseline_missing: int
    candidate_missing: int
    both_missing: int
    not_comparable: int
    mean_difference: float | None
    complete_cases: int
    weight_sum: float
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    direction: Literal["higher", "lower"] = "higher"

    def __post_init__(self) -> None:
        for name in ("mean_difference", "weight_sum", "baseline_mean", "candidate_mean"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    report_id: str
    baseline: CandidateSlotRef
    candidate: CandidateSlotRef
    selections: tuple[ScoreComparisonSelection, ...]
    compatibility: Literal["compatible", "incompatible"]
    differences: tuple[ContractDifference, ...]
    dimensions: tuple[PairedDimensionSummary, ...]
    gate: Literal["pass", "fail", "inconclusive", "not_configured"]
    gate_reasons: tuple[str, ...]
    cutoff: ComparisonReadCutoff
    created_at: datetime
    pairs: tuple[PairedScoreRow, ...] = ()
    gate_policy: GatePolicy | None = None

    def __post_init__(self) -> None:
        for name in ("selections", "differences", "dimensions", "gate_reasons", "pairs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def _failures(counts: Mapping[str, int]) -> tuple[FailureCount, ...]:
    return tuple(FailureCount(key, value) for key, value in sorted(counts.items()) if value)


def _weighted(values: Mapping[CaseRef, Sequence[float]], weights: Mapping[CaseRef, float]) -> tuple[float | None, float]:
    weight_sum = math.fsum(weights[ref] for ref, items in values.items() if items)
    if not weight_sum:
        return None, 0.0
    total = math.fsum(weights[ref] * (math.fsum(items) / len(items)) for ref, items in values.items() if items)
    return total / weight_sum, weight_sum


def _score_value(score: ScoreAttemptView | None, dimension: str, decision_id: str | None = None) -> tuple[str, float | None, str | None]:
    if score is None:
        return "not_attempted", None, "score_missing"
    if decision_id is not None and score.decision_id != decision_id:
        return "not_attempted", None, "decision_missing"
    if score.status not in ("valid", "not_applicable"):
        return score.status, None, score.reason or score.status
    if score.status == "not_applicable" and score.score is None:
        return "not_applicable", None, score.reason or "not_applicable"
    if score.score is None or dimension not in score.score.dimensions:
        return "error", None, "dimension_missing"
    value = score.score.dimensions[dimension]
    if isinstance(value, ScoreNotApplicable):
        return "not_applicable", None, value.reason
    return "valid", float(value), None


def _trial_state(trial: TrialView | None) -> str:
    if trial is None:
        return "pending"
    if trial.execution_status is not None:
        status = trial.execution_status.value.lower()
        if status == "blocked":
            return "failed"
        if status in ("succeeded", "failed", "cancelled", "recovery_required"):
            return status
    if trial.disposition is not None:
        if trial.disposition.terminal:
            kind = trial.disposition.kind
            return {"cancelled": "cancelled", "permanent_invalid": "failed"}.get(kind, "unavailable")
        if trial.disposition.kind == "recovery_required":
            return "recovery_required"
    return "pending"


def build_evaluation_report(
    manifest: EvaluationManifest, cases: Sequence[CaseContract], trials: Sequence[TrialView],
    scores: Sequence[ScoreAttemptView], *, cutoff: EvaluationReadCutoff,
    report_id: str, created_at: datetime,
) -> EvaluationReport:
    """Summarize all planned slots, including absent observations and pre-start failures."""
    if cutoff.experiment_id != manifest.experiment_id or cutoff.manifest_digest != manifest.digest:
        raise ValueError("cutoff does not identify the supplied manifest")
    weights = {case.ref: case.weight for case in cases}
    trial_map = {trial.trial: trial for trial in trials}
    score_map = {(score.trial, score.scorer_slot_id): score for score in scores
                 if score.scoring_experiment_id == manifest.experiment_id}
    planned = [(TargetTrialRef(manifest.experiment_id, plan.trial_id), plan.case_ref, plan.candidate_slot_id)
               for plan in manifest.trials]
    if manifest.kind == "score_only":
        planned = [(ref, trial_map[ref].case_ref, trial_map[ref].candidate_slot_id) for ref in manifest.source_trials]
    selected_trials = tuple(trial_map.get(ref) or TrialView(ref, plan.case_ref, plan.candidate_slot_id, plan.repetition)
                            for ref, plan in ((TargetTrialRef(manifest.experiment_id, plan.trial_id), plan)
                                              for plan in manifest.trials)) if manifest.kind == "experiment" else tuple(
                            trial_map[ref] for ref in manifest.source_trials)
    selected_scores = tuple(score_map.get((ref, scorer.slot_id)) or ScoreAttemptView(
        manifest.experiment_id, None, ref, scorer.slot_id, scorer.task, "not_attempted", reason="score_missing")
        for ref, _, _ in planned for scorer in manifest.scorers)
    candidates = tuple(dict.fromkeys(slot for _, _, slot in planned))
    summaries: list[CandidateSummary] = []
    scorer_summaries: list[ScorerSummary] = []
    failures: Counter[str] = Counter()
    required = [(scorer.slot_id, dimension.name) for scorer in manifest.scorers if scorer.required
                for dimension in scorer.dimensions]
    for slot in candidates:
        selected = [(ref, case_ref) for ref, case_ref, candidate in planned if candidate == slot]
        states = Counter(_trial_state(trial_map.get(ref)) for ref, _ in selected)
        started = sum(trial_map.get(ref) is not None and trial_map[ref].subject is not None for ref, _ in selected)
        complete = sum(all(_score_value(score_map.get((ref, scorer)), dimension)[0] == "valid"
                           for scorer, dimension in required) for ref, _ in selected)
        summaries.append(CandidateSummary(slot, len(selected), started, states["succeeded"], states["failed"],
                                          states["cancelled"], states["unavailable"], complete))
        for ref, _ in selected:
            trial = trial_map.get(ref)
            reason = None if trial is None else trial.error_code or (
                trial.disposition.reason_code if trial.disposition is not None else None)
            if reason:
                failures[reason] += 1
        for scorer in manifest.scorers:
            for dimension in scorer.dimensions:
                counts: Counter[str] = Counter()
                reasons: Counter[str] = Counter()
                values: dict[CaseRef, list[float]] = defaultdict(list)
                for ref, case_ref in selected:
                    status, value, reason = _score_value(score_map.get((ref, scorer.slot_id)), dimension.name)
                    counts[status] += 1
                    if value is not None:
                        values[case_ref].append(value)
                    if reason:
                        reasons[reason] += 1
                mean, weight_sum = _weighted(values, weights)
                scorer_summaries.append(ScorerSummary(
                    scorer.slot_id, dimension.name, len(selected), counts["pending"], counts["valid"],
                    counts["not_applicable"], counts["error"], counts["not_attempted"], mean, weight_sum,
                    counts["valid"] / len(selected) if selected else 0.0, _failures(reasons), slot,
                ))
    target_states = {_trial_state(trial) for trial in selected_trials} if manifest.kind == "experiment" else set()
    completion = "needs_attention" if "recovery_required" in target_states else (
        "running" if "pending" in target_states or any(score.status == "pending" for score in selected_scores) else "complete")
    return EvaluationReport(report_id, manifest.experiment_id, manifest.kind, manifest.source_experiment_id,
                            manifest.dataset, tuple(summaries), tuple(scorer_summaries), _failures(failures),
                            cutoff, created_at, selected_trials, selected_scores, completion)


def _candidate_parts(candidate: CandidateContract) -> dict[str, JsonValue]:
    definitions = []
    models = []
    instructions = []
    capabilities = []
    parameters = []
    for declaration in candidate.definition_contracts:
        definition = dict(declaration)
        if definition.get("type") == "agent":
            config = dict(definition["config"])
            binding = dict(config.pop("binding_contract"))
            agent = dict(binding.pop("agent_spec"))
            models.append({"model": agent.pop("model"), "contract": binding.pop("model_contract")})
            instructions.append({key: agent.pop(key) for key in ("system_prompt", "instructions")})
            capabilities.append({
                "selected": binding.pop("selected"), "subagents": binding.pop("subagents"),
                "subagent_bindings": binding.pop("subagent_bindings", []),
                "selectors": {key: agent.pop(key, []) for key in (
                    "allow_tools", "allow_skills", "allow_subagents", "allow_capabilities", "preload_skills")},
            })
            parameters.append({key: agent.pop(key) for key in (
                "usage_limits", "planning", "thinking", "tool_retries", "output_retries")})
            agent.pop("description", None)
            agent.pop("metadata", None)
            binding["agent_spec"] = agent
            config["binding_contract"] = binding
            definition["config"] = config
        definitions.append(definition)
    graph = None
    if candidate.graph_template is not None:
        graph = candidate.graph_template.to_mapping()
        graph.pop("namespace")
        graph.pop("tenant_id")
    return {
        "task_definition": {"task": None if candidate.task is None else
                            {"id": candidate.task.id, "revision": candidate.task.revision},
                            "definitions": definitions},
        "graph_definition": graph,
        "model_binding": models, "agent_instructions": instructions, "capabilities": capabilities,
        "candidate_parameters": parameters,
    }


def build_comparison_report(
    spec: ComparisonSpec, baseline_manifest: EvaluationManifest, candidate_manifest: EvaluationManifest,
    baseline_cases: Sequence[CaseContract], candidate_cases: Sequence[CaseContract],
    trials: Sequence[TrialView], scores: Sequence[ScoreAttemptView], *,
    baseline_dataset: DatasetContract, candidate_dataset: DatasetContract,
    cutoff: ComparisonReadCutoff, report_id: str, created_at: datetime,
    scoring_manifests: Sequence[EvaluationManifest] = (),
    completions: Mapping[str, str] | None = None, usage_complete: bool | None = None,
) -> ComparisonReport:
    """Compare fixed source views. No source lookup or latest-score selection occurs here."""
    differences: list[ContractDifference] = []

    def compare(path: str, left: JsonValue, right: JsonValue, allowed: bool = False) -> None:
        if left != right:
            differences.append(ContractDifference(path, left, right, allowed))

    if (baseline_manifest.experiment_id != spec.baseline.experiment_id or
            candidate_manifest.experiment_id != spec.candidate.experiment_id):
        raise ValueError("comparison manifest does not match candidate reference")
    if baseline_manifest.kind != "experiment" or candidate_manifest.kind != "experiment":
        raise ValueError("comparison candidates must own target trials")
    if spec.cutoff is not None and cutoff != spec.cutoff:
        raise ValueError("comparison cutoff differs from the selected snapshot")
    for manifest, source_cutoff in ((baseline_manifest, cutoff.baseline), (candidate_manifest, cutoff.candidate)):
        if source_cutoff.experiment_id != manifest.experiment_id or source_cutoff.manifest_digest != manifest.digest:
            raise ValueError("cutoff does not identify the supplied manifest")
    left_candidate = next(item for item in baseline_manifest.candidates if item.slot_id == spec.baseline.slot_id)
    right_candidate = next(item for item in candidate_manifest.candidates if item.slot_id == spec.candidate.slot_id)
    compare("dataset", baseline_dataset.to_mapping(), candidate_dataset.to_mapping())
    compare("cases", [item.to_mapping() for item in baseline_cases], [item.to_mapping() for item in candidate_cases])
    compare("input_mode", baseline_manifest.input_mode, candidate_manifest.input_mode)
    compare("policy", baseline_manifest.policy.to_mapping(), candidate_manifest.policy.to_mapping())
    if left_candidate.graph_template is not None and right_candidate.graph_template is not None:
        left_ref, right_ref = left_candidate.graph_template, right_candidate.graph_template
        compare("graph_scope", {"namespace": left_ref.namespace, "tenant_id": left_ref.tenant_id},
                {"namespace": right_ref.namespace, "tenant_id": right_ref.tenant_id})
    left_parts, right_parts = _candidate_parts(left_candidate), _candidate_parts(right_candidate)
    for path, left in left_parts.items():
        compare(path, left, right_parts[path], path in spec.allowed_changes)
    left_definitions = {(item["id"], item["revision"]): item for item in left_candidate.definition_contracts}
    for declaration in right_candidate.definition_contracts:
        key = (declaration["id"], declaration["revision"])
        if key in left_definitions:
            compare("candidate.contract_drift", dict(left_definitions[key]), dict(declaration))
    left_plans = {(plan.case_ref, plan.repetition): plan for plan in baseline_manifest.trials
                  if plan.candidate_slot_id == spec.baseline.slot_id}
    right_plans = {(plan.case_ref, plan.repetition): plan for plan in candidate_manifest.trials
                   if plan.candidate_slot_id == spec.candidate.slot_id}
    compare("repetition_plan", [{"case": ref.to_mapping(), "repetition": repetition} for ref, repetition in left_plans],
            [{"case": ref.to_mapping(), "repetition": repetition} for ref, repetition in right_plans])
    keys = tuple(dict.fromkeys((*left_plans, *right_plans)))
    score_cutoffs = {item.experiment_id: item for item in cutoff.scoring}
    manifests = {manifest.experiment_id: manifest for manifest in
                 (baseline_manifest, candidate_manifest, *scoring_manifests)}
    score_map = {(score.scoring_experiment_id, score.trial, score.scorer_slot_id): score for score in scores}
    trial_map = {trial.trial: trial for trial in trials}
    weights = {case.ref: case.weight for case in (*candidate_cases, *baseline_cases)}
    pending = False
    relevant_experiments = {baseline_manifest.experiment_id, candidate_manifest.experiment_id}
    resolved: list[tuple[ScoreComparisonSelection, str, str, ScorerContract | None]] = []
    for index, selection in enumerate(spec.scores):
        bindings: list[ScorerContract | None] = []
        ids: list[str] = []
        for side, selected, target_manifest in (("baseline", selection.baseline, baseline_manifest),
                                                ("candidate", selection.candidate, candidate_manifest)):
            scoring_id = selected.scoring_experiment_id or target_manifest.experiment_id
            ids.append(scoring_id)
            relevant_experiments.add(scoring_id)
            source = manifests.get(scoring_id)
            binding = None if source is None else next((item for item in source.scorers
                                                        if item.slot_id == selected.scorer_slot_id), None)
            valid_source = source is not None and (source.experiment_id == target_manifest.experiment_id or
                           source.kind == "score_only" and source.source_experiment_id == target_manifest.experiment_id)
            if not valid_source or binding is None or not any(item.name == selected.dimension for item in binding.dimensions):
                differences.append(ContractDifference(f"scores.{index}.{side}.source", "selected target/scorer/dimension", scoring_id, False))
                binding = None
            if valid_source and source.kind == "score_only":
                source_cutoff = score_cutoffs.get(scoring_id)
                if source_cutoff is None or source_cutoff.manifest_digest != source.digest:
                    raise ValueError("cutoff does not identify the selected scoring manifest")
            if valid_source and binding is not None:
                planned_refs = set(source.source_trials) if source.kind == "score_only" else {
                    TargetTrialRef(source.experiment_id, plan.trial_id) for plan in source.trials}
                target_refs = {TargetTrialRef(target_manifest.experiment_id, plan.trial_id) for plan in target_manifest.trials}
                if not planned_refs <= target_refs:
                    differences.append(ContractDifference(f"scores.{index}.{side}.trials", "source target trials", scoring_id, False))
                for score in scores:
                    if score.scoring_experiment_id == scoring_id and score.scorer_slot_id == selected.scorer_slot_id:
                        if score.trial not in planned_refs or score.scorer_task != binding.task:
                            differences.append(ContractDifference(f"scores.{index}.{side}.binding", "selected source trial and scorer", score.trial.to_mapping(), False))
                side_slot = spec.baseline.slot_id if side == "baseline" else spec.candidate.slot_id
                selected_refs = {TargetTrialRef(target_manifest.experiment_id, plan.trial_id)
                                 for plan in target_manifest.trials if plan.candidate_slot_id == side_slot}
                if selected.decision_id is not None and not any(
                    score.scoring_experiment_id == scoring_id and score.scorer_slot_id == selected.scorer_slot_id
                    and score.trial in selected_refs and score.decision_id == selected.decision_id for score in scores
                ):
                    differences.append(ContractDifference(f"scores.{index}.{side}.decision", selected.decision_id, None, False))
            bindings.append(binding)
        left_binding, right_binding = bindings
        if left_binding is not None and right_binding is not None:
            left = left_binding.to_mapping()
            right = right_binding.to_mapping()
            left.pop("slot_id")
            right.pop("slot_id")
            compare(f"scores.{index}.contract", left, right)
            compare(f"scores.{index}.dimension", selection.baseline.dimension, selection.candidate.dimension)
        resolved.append((selection, ids[0], ids[1], right_binding))
    incompatible = any(not item.allowed for item in differences)
    rows: list[PairedScoreRow] = []
    dimensions: list[PairedDimensionSummary] = []
    for selection, left_id, right_id, binding in resolved:
        selected_rows: list[PairedScoreRow] = []
        left_values: dict[CaseRef, list[float]] = defaultdict(list)
        right_values: dict[CaseRef, list[float]] = defaultdict(list)
        deltas: dict[CaseRef, list[float]] = defaultdict(list)
        for case_ref, repetition in keys:
            left_plan, right_plan = left_plans.get((case_ref, repetition)), right_plans.get((case_ref, repetition))
            left_ref = None if left_plan is None else TargetTrialRef(baseline_manifest.experiment_id, left_plan.trial_id)
            right_ref = None if right_plan is None else TargetTrialRef(candidate_manifest.experiment_id, right_plan.trial_id)
            left_score = score_map.get((left_id, left_ref, selection.baseline.scorer_slot_id))
            right_score = score_map.get((right_id, right_ref, selection.candidate.scorer_slot_id))
            left_status, left_value, left_reason = _score_value(left_score, selection.baseline.dimension, selection.baseline.decision_id)
            right_status, right_value, right_reason = _score_value(right_score, selection.candidate.dimension, selection.candidate.decision_id)
            pending |= left_status == "pending" or right_status == "pending"
            if incompatible:
                status = "not_comparable"
            elif left_value is None:
                status = "both_missing" if right_value is None else "baseline_missing"
            elif right_value is None:
                status = "candidate_missing"
            else:
                status = "complete"
                left_values[case_ref].append(left_value)
                right_values[case_ref].append(right_value)
                deltas[case_ref].append(right_value - left_value)
            selected_rows.append(PairedScoreRow(selection, case_ref, repetition, left_ref, right_ref, status,
                                                left_value, right_value, left_reason, right_reason, weights[case_ref]))
        counts = Counter(row.status for row in selected_rows)
        mean, weight_sum = _weighted(deltas, weights)
        direction = "higher" if binding is None else next(
            item.direction for item in binding.dimensions if item.name == selection.candidate.dimension)
        dimensions.append(PairedDimensionSummary(
            selection, len(keys), counts["complete"], counts["baseline_missing"], counts["candidate_missing"],
            counts["both_missing"], counts["not_comparable"], mean, len(deltas), weight_sum,
            _weighted(left_values, weights)[0], _weighted(right_values, weights)[0], direction,
        ))
        rows.extend(selected_rows)
    target_refs = [TargetTrialRef(manifest.experiment_id, plan.trial_id)
                   for manifest, slot in ((baseline_manifest, spec.baseline.slot_id), (candidate_manifest, spec.candidate.slot_id))
                   for plan in manifest.trials if plan.candidate_slot_id == slot]
    states = [_trial_state(trial_map.get(ref)) for ref in target_refs]
    pending |= any(state in ("pending", "recovery_required") for state in states)
    completion_values = [value for key, value in (completions or {}).items() if key in relevant_experiments]
    pending |= any(value in ("running", "cancelling", "needs_attention") for value in completion_values)
    cancelled = "cancelled" in completion_values
    candidate_states = [_trial_state(trial_map.get(TargetTrialRef(candidate_manifest.experiment_id, plan.trial_id)))
                        for plan in right_plans.values()]
    gate, reasons = _gate(spec, dimensions, incompatible=incompatible, pending=pending, cancelled=cancelled,
                          usage_complete=usage_complete,
                          failure_rate=sum(state in ("failed", "unavailable", "cancelled") for state in candidate_states)
                          / len(candidate_states) if candidate_states else 1.0)
    return ComparisonReport(report_id, spec.baseline, spec.candidate, spec.scores,
                            "incompatible" if incompatible else "compatible", tuple(differences), tuple(dimensions),
                            gate, reasons, cutoff, created_at, tuple(rows), spec.gate_policy)


def _gate(
    spec: ComparisonSpec, dimensions: Sequence[PairedDimensionSummary], *, incompatible: bool,
    pending: bool, cancelled: bool, usage_complete: bool | None, failure_rate: float,
) -> tuple[Literal["pass", "fail", "inconclusive", "not_configured"], tuple[str, ...]]:
    policy = spec.gate_policy
    if policy is None:
        return "not_configured", ()
    reasons = tuple(reason for condition, reason in (
        (spec.mode == "exploratory", "exploratory_comparison"), (incompatible, "incompatible"),
        (pending, "pending_or_recovery_required"), (cancelled, "cancelled"),
        (policy.require_complete_usage and usage_complete is not True, "unknown_required_usage"),
    ) if condition)
    if reasons:
        return "inconclusive", reasons
    if any(item.complete_pairs == 0 or item.complete_cases < policy.minimum_cases or
           item.complete_pairs / item.planned_pairs < policy.minimum_coverage for item in dimensions):
        return "inconclusive", ("insufficient_coverage_or_cases",)
    failures: list[str] = []
    for bound in policy.bounds:
        summary = next(item for item in dimensions if item.selection == bound.selection)
        if bound.minimum is not None and summary.candidate_mean < bound.minimum:
            failures.append(f"{bound.selection.candidate.dimension}:below_minimum")
        if bound.maximum is not None and summary.candidate_mean > bound.maximum:
            failures.append(f"{bound.selection.candidate.dimension}:above_maximum")
        improvement = summary.mean_difference if summary.direction == "higher" else -summary.mean_difference
        if bound.max_regression is not None and improvement < -bound.max_regression:
            failures.append(f"{bound.selection.candidate.dimension}:regression")
    if policy.maximum_failure_rate is not None and failure_rate > policy.maximum_failure_rate:
        failures.append("maximum_failure_rate_exceeded")
    return ("fail", tuple(failures)) if failures else ("pass", ())


def _json(value: object) -> JsonValue:
    if isinstance(value, ScoreBundle):
        return value.to_mapping()
    if isinstance(value, ScoreNotApplicable):
        return {"kind": "not_applicable", "reason": value.reason}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return normalize_json_value(value)


def export_report(report: EvaluationReport | ComparisonReport, *, format: Literal["json", "csv", "markdown"] = "json") -> str:
    """Export a saved snapshot without re-reading or re-computing its sources."""
    if format == "json":
        return json.dumps(_json(report), ensure_ascii=False, allow_nan=False, sort_keys=True)
    if format == "csv":
        stream = io.StringIO(newline="")
        if isinstance(report, ComparisonReport):
            names = ("dataset_id", "case_id", "case_revision", "repetition", "baseline_trial", "candidate_trial",
                     "baseline_scoring_experiment", "candidate_scoring_experiment", "baseline_scorer", "candidate_scorer",
                     "dimension", "status", "baseline_value", "candidate_value", "baseline_reason", "candidate_reason", "weight")
            writer = csv.writer(stream)
            writer.writerow(names)
            for row in report.pairs:
                writer.writerow((row.case_ref.dataset_id, row.case_ref.case_id, row.case_ref.revision, row.repetition,
                    None if row.baseline_trial is None else row.baseline_trial.trial_id,
                    None if row.candidate_trial is None else row.candidate_trial.trial_id,
                    row.selection.baseline.scoring_experiment_id or report.baseline.experiment_id,
                    row.selection.candidate.scoring_experiment_id or report.candidate.experiment_id,
                    row.selection.baseline.scorer_slot_id, row.selection.candidate.scorer_slot_id,
                    row.selection.candidate.dimension, row.status,
                    "null" if row.baseline_value is None else row.baseline_value,
                    "null" if row.candidate_value is None else row.candidate_value,
                    row.baseline_reason, row.candidate_reason, row.weight))
        else:
            writer = csv.writer(stream)
            writer.writerow(("scoring_experiment_id", "target_experiment_id", "trial_id", "scorer_slot_id",
                             "score_attempt_id", "dimension", "status", "value", "reason"))
            for score in report.score_attempts:
                dimensions = () if score.score is None else tuple(score.score.dimensions)
                for dimension in dimensions or (None,):
                    status, value, reason = _score_value(score, dimension) if dimension is not None else (score.status, None, score.reason)
                    writer.writerow((score.scoring_experiment_id, score.trial.target_experiment_id, score.trial.trial_id,
                                     score.scorer_slot_id, score.score_attempt_id, dimension, status,
                                     "null" if value is None else value, reason))
        return stream.getvalue()
    if format != "markdown":
        raise ValueError("unknown report format")
    lines = [f"# Evaluation report {report.report_id}", ""]
    if isinstance(report, ComparisonReport):
        lines += [f"Compatibility: {report.compatibility}; gate: {report.gate}", "",
                  "| Dimension | Complete / planned | Baseline | Candidate | Difference |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {item.selection.candidate.dimension} | {item.complete_pairs} / {item.planned_pairs} | "
                  f"{item.baseline_mean} | {item.candidate_mean} | {item.mean_difference} |" for item in report.dimensions]
        lines += ["", *report.gate_reasons]
    else:
        lines += ["| Candidate | Scorer | Dimension | Valid / planned | Mean |", "|---|---|---|---:|---:|"]
        lines += [f"| {item.candidate_slot_id} | {item.scorer_slot_id} | {item.dimension} | "
                  f"{item.valid} / {item.planned} | {item.mean} |" for item in report.scores]
    return "\n".join(lines) + "\n"


__all__ = [
    "CandidateSlotRef", "ScoreSelection", "ScoreComparisonSelection", "DimensionBound", "GatePolicy",
    "EvaluationReadCutoff", "ComparisonReadCutoff", "ComparisonSpec", "ContractDifference", "FailureCount",
    "CandidateSummary", "ScorerSummary", "EvaluationReport", "PairedScoreRow", "PairedDimensionSummary",
    "ComparisonReport", "build_evaluation_report", "build_comparison_report", "export_report",
]
