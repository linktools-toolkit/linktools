#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable evaluation control facts, separate from native execution state."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from ..core import Principal, JsonValue, validate_idempotency_key, validate_resource_id
from ..errors import AIError
from ..evaluation import (
    ScoreBundle, CaseContract, DatasetContract, CaseRef, DatasetRef,
    EvaluationManifest,
    EvidenceRef,
    ScoreAttemptView,
    SlotDispositionView,
    TargetTrialRef,
)
from ..task import TaskGraphSubmission, TaskInvocationInputRef
from ..storage import ObjectRef
from ._plan import PersistenceDomain


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
    idempotency_key: str = field(repr=False)
    score: ScoreBundle

    def __post_init__(self) -> None:
        try:
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("evaluation idempotency key is invalid") from error


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    manifest: EvaluationManifest
    manifest_digest: str
    request_digest: str
    idempotency_key: str = field(repr=False)
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
        try:
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("evaluation idempotency key is invalid") from error
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
    idempotency_key: str = field(repr=False)
    manifest_digest: str
    deleted_at: datetime

    def __post_init__(self) -> None:
        try:
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("evaluation idempotency key is invalid") from error


@dataclass(frozen=True, slots=True)
class EvaluationContentTombstone:
    kind: str
    identity: JsonValue
    owner_principal_id: str
    digest: str
    deleted_at: datetime


    def __post_init__(self) -> None:
        self.record_identity

    @property
    def record_identity(self) -> tuple[str, ...]:
        """Project this content kind's semantic identity into record fields."""
        if self.kind == "evaluation_dataset":
            if not isinstance(self.identity, list) or len(self.identity) != 2:
                raise ValueError("dataset tombstone identity is invalid")
            identifier, revision = self.identity
            if not isinstance(identifier, str) or not isinstance(revision, int):
                raise ValueError("dataset tombstone identity is invalid")
            dataset = DatasetRef(identifier, revision)
            return (dataset.id, str(dataset.revision))
        if self.kind == "evaluation_case":
            if not isinstance(self.identity, list) or len(self.identity) != 3:
                raise ValueError("case tombstone identity is invalid")
            dataset_id, case_id, revision = self.identity
            if not isinstance(dataset_id, str) or not isinstance(case_id, str) or not isinstance(revision, int):
                raise ValueError("case tombstone identity is invalid")
            case = CaseRef(dataset_id, case_id, revision)
            return (case.dataset_id, case.case_id, str(case.revision))
        if self.kind in {"evaluation_evidence", "evaluation_report", "evaluation_comparison"}:
            if not isinstance(self.identity, str):
                raise ValueError("evaluation content tombstone identity is invalid")
            try:
                return (validate_resource_id(self.identity),)
            except AIError as error:
                raise ValueError("evaluation content tombstone identity is invalid") from error
        raise ValueError("evaluation content tombstone kind is invalid")


@dataclass(frozen=True, slots=True)
class EvaluationCleanupRecord:
    evaluation_id: str
    owner_principal_id: str
    objects: tuple[tuple[PersistenceDomain, ObjectRef], ...]
    created_at: datetime
    capture_keys: tuple[str, ...] = ()


__all__ = [
    "EvaluationCleanupRecord", "EvaluationTombstone", "EvaluationContentTombstone",
    "EvaluationDatasetRecord", "EvaluationCaseRecord",
    "EvaluationLaunchIntent", "EvaluationTrialEvidence",
    "EvaluationSlotDisposition", "EvaluationRecord", "EvaluationHumanDecision",
]
