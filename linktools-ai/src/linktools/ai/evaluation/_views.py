#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only projections of evaluation plans, executions, and scores."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..core import ExecutionStatus, TaskStatus
from ..task import TaskRef
from ._contracts import CaseRef, DatasetRef, TargetTrialRef
from ._evidence import EvidenceRef, ExecutionSubjectRef, GraphSubjectRef, ScoreBundle


@dataclass(frozen=True, slots=True)
class SlotDispositionView:
    kind: Literal["permanent_unavailable", "permanent_invalid", "cancelled", "recoverable_blocked"]
    reason_code: str
    terminal: bool
    retryable: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationIssue:
    code: str
    message: str
    trial: TargetTrialRef | None = None
    scorer_slot_id: str | None = None
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    planned_trials: int = 0
    terminal_trials: int = 0
    planned_scores: int = 0
    terminal_scores: int = 0
    valid_scores: int = 0
    blocked_slots: int = 0
    source_trial_count: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationView:
    experiment_id: str
    kind: Literal["experiment", "score_only"]
    source_experiment_id: str | None
    dataset: DatasetRef
    progress: EvaluationProgress
    completion: Literal["running", "cancelling", "complete", "cancelled", "needs_attention"]
    needs_attention: tuple[EvaluationIssue, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TrialView:
    trial: TargetTrialRef
    case_ref: CaseRef
    candidate_slot_id: str
    repetition: int
    graph_ref: GraphSubjectRef | None = None
    subject: ExecutionSubjectRef | GraphSubjectRef | None = None
    execution_status: ExecutionStatus | TaskStatus | None = None
    disposition: SlotDispositionView | None = None
    evidence_ref: EvidenceRef | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ScoreAttemptView:
    scoring_experiment_id: str
    score_attempt_id: str | None
    trial: TargetTrialRef
    scorer_slot_id: str
    scorer_task: TaskRef
    status: Literal["pending", "valid", "error", "not_applicable", "not_attempted"]
    score: ScoreBundle | None = None
    scorer_execution: ExecutionSubjectRef | None = None
    scorer_graph: GraphSubjectRef | None = None
    scorer_node_id: str | None = None
    evidence_ref: EvidenceRef | None = None
    decision_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TrialFilter:
    candidate_slot_ids: tuple[str, ...] = ()
    case_refs: tuple[CaseRef, ...] = ()
    terminal: bool | None = None


@dataclass(frozen=True, slots=True)
class ScoreFilter:
    scorer_slot_ids: tuple[str, ...] = ()
    trial_ids: tuple[str, ...] = ()
    statuses: tuple[Literal["pending", "valid", "error", "not_applicable", "not_attempted"], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationPurgeResult:
    evaluations: tuple[str, ...]
    blocked: tuple[str, ...]
    objects_deleted: int
    objects_missing: int
    objects_retained: int
    objects_blocked: int
    datasets: tuple[DatasetRef, ...] = ()


__all__ = ["EvaluationPurgeResult", "EvaluationIssue", "EvaluationProgress", "EvaluationView", "ScoreAttemptView", "ScoreFilter",
           "SlotDispositionView", "TrialFilter", "TrialView"]
