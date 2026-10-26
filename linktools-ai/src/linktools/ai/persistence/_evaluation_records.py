#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable evaluation control facts, separate from native execution state."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from linktools.ai.core import Principal, JsonValue
from linktools.ai.evaluation import (
    ScoreBundle, CaseContract, DatasetContract,
    EvaluationManifest,
    EvidenceRef,
    ScoreAttemptView,
    SlotDispositionView,
    TargetTrialRef,
)
from linktools.ai.task import TaskGraphSubmission, TaskInvocationInputRef
from linktools.ai.storage import ObjectRef
from linktools.ai.persistence._plan import PersistenceDomain


@dataclass(frozen=True, slots=True)
class EvaluationDatasetRecord:
    contract: DatasetContract
    owner_principal_id: str
    created_at: datetime | None = None
    content_expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EvaluationCaseRecord:
    contract: CaseContract
    owner_principal_id: str
    created_at: datetime | None = None
    content_expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EvaluationLaunchIntent:
    slot_id: str
    trial: TargetTrialRef
    scorer_slot_id: str | None
    submission: TaskGraphSubmission
    deadline_at: datetime | None
    confirmed: bool = False
    released: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationTrialEvidence:
    trial: TargetTrialRef
    evidence_ref: EvidenceRef


@dataclass(frozen=True, slots=True)
class EvaluationSlotDisposition:
    slot_id: str
    disposition: SlotDispositionView


@dataclass(frozen=True, slots=True)
class EvaluationHumanDecision:
    slot_id: str
    decision_id: str
    actor: Principal
    created_at: datetime
    request_digest: str
    idempotency_key_digest: str
    score: ScoreBundle


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    manifest: EvaluationManifest
    manifest_digest: str
    request_digest: str
    idempotency_key_digest: str
    gate: Literal["open", "closed_cancel", "closed_budget"]
    revision: int
    created_at: datetime
    updated_at: datetime
    intents: tuple[EvaluationLaunchIntent, ...] = ()
    evidence: tuple[EvaluationTrialEvidence, ...] = ()
    scores: tuple[ScoreAttemptView, ...] = ()
    dispositions: tuple[EvaluationSlotDisposition, ...] = ()
    human_decisions: tuple[EvaluationHumanDecision, ...] = ()
    content_expires_at: datetime | None = None
    metadata_expires_at: datetime | None = None
    content_deleted_at: datetime | None = None
    owned_input_captures: tuple[TaskInvocationInputRef, ...] = ()

    def __post_init__(self) -> None:
        if self.content_deleted_at is None and any(
            candidate.graph_template is not None and candidate.graph_template.template is None
            for candidate in self.manifest.candidates
        ):
            raise ValueError("live evaluation requires its graph template content")

    @property
    def experiment_id(self) -> str:
        return self.manifest.experiment_id


@dataclass(frozen=True, slots=True)
class EvaluationTombstone:
    evaluation_id: str
    owner_principal_id: str
    request_digest: str
    idempotency_key_digest: str
    manifest_digest: str
    deleted_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationContentTombstone:
    kind: str
    identity: JsonValue
    owner_principal_id: str
    digest: str
    deleted_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationCleanupRecord:
    evaluation_id: str
    owner_principal_id: str
    objects: tuple[tuple[PersistenceDomain, ObjectRef], ...]
    created_at: datetime


__all__ = [
    "EvaluationCleanupRecord", "EvaluationTombstone", "EvaluationContentTombstone",
    "EvaluationDatasetRecord", "EvaluationCaseRecord",
    "EvaluationLaunchIntent", "EvaluationTrialEvidence",
    "EvaluationSlotDisposition", "EvaluationRecord", "EvaluationHumanDecision",
]
