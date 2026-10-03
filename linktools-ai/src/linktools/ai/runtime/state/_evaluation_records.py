#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable evaluation control facts, separate from native execution state."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ...core import Principal, JsonValue
from ...evaluation import (
    ScoreBundle, CaseContract, DatasetContract,
    EvaluationManifest,
    EvidenceRef,
    ScoreAttemptView,
    SlotDispositionView,
    TargetTrialRef,
)
from ...task import TaskGraphSubmission
from ...storage import ObjectRef
from ._plan import RuntimeDomain


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

    @property
    def evaluation_id(self) -> str:
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
    objects: tuple[tuple[RuntimeDomain, ObjectRef], ...]
    created_at: datetime


__all__ = [
    "EvaluationCleanupRecord", "EvaluationTombstone", "EvaluationContentTombstone",
    "EvaluationDatasetRecord", "EvaluationCaseRecord",
    "EvaluationLaunchIntent", "EvaluationTrialEvidence",
    "EvaluationSlotDisposition", "EvaluationRecord", "EvaluationHumanDecision",
]
