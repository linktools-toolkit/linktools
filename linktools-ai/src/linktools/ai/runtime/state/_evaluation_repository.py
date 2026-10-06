#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation facts and admission control on the existing state transaction."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import TypeVar

from ...core import IdempotencyStatus, ResourceKind
from ...errors import AIError, ErrorCode
from ...evaluation import (
    CaseContract, CaseRef, ComparisonReport, DatasetContract, DatasetRef,
    EvaluationReport, EvaluationReadCutoff, EvidenceBundle, EvidenceRef, ScoreBundle,
    SlotDispositionView, ScoreNotApplicable,
)
from ._contracts import IdempotencyRecord
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._evaluation_records import (
    EvaluationLaunchIntent, EvaluationRecord, EvaluationTrialEvidence,
    EvaluationDatasetRecord, EvaluationCaseRecord, EvaluationSlotDisposition,
    EvaluationTombstone, EvaluationContentTombstone, EvaluationCleanupRecord,
)
from ._plan import RuntimeDomain
from ._repository_common import RepositoryBase, projected_record, replace_checked, record_cursor, record_state
from ...storage import ObjectRef
from ._codec import encode_domain, iter_runtime_object_refs
from ._store import StateStore, StateTransaction, RecordQuery

ValueT = TypeVar("ValueT")


class EvaluationRepositoryImpl(RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id,
                         domain=RuntimeDomain.EVALUATION)
        self._background_tasks: set[asyncio.Task[object]] = set()

    async def _commit(
        self, operation: Callable[[], Awaitable[ValueT]],
        readback: Callable[[], Awaitable[ValueT | None]],
    ) -> ValueT:
        async def observe() -> CommitObservation[ValueT]:
            value = await readback()
            return CommitObservation(DurableCommitState.NOT_COMMITTED if value is None
                                     else DurableCommitState.COMMITTED, value)
        result = await run_durable_commit(operation, observe, background_tasks=self._background_tasks)
        if result.cancelled:
            raise asyncio.CancelledError()
        if result.committed:
            return result.value
        if result.error is not None:
            raise result.error
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    async def _put(
        self, transaction: StateTransaction, kind: str, identity: object,
        value: ValueT,
    ) -> ValueT:
        tombstone_kind = "evaluation_tombstone" if kind == "evaluation" else "evaluation_content_tombstone"
        tombstone_id = identity if kind == "evaluation" else [kind, identity]
        if kind in {"evaluation", "evaluation_case", "evaluation_dataset", "evaluation_evidence", "evaluation_report", "evaluation_comparison"} and await transaction.get_record(self._key(tombstone_kind, tombstone_id)) is not None:
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
        current = await transaction.get_record(self._key(kind, identity))
        if current is not None:
            existing = await self._decode(current, type(value))
            if isinstance(existing, (EvaluationDatasetRecord, EvaluationCaseRecord)) and existing.owner_principal_id != value.owner_principal_id:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if isinstance(existing, (EvaluationDatasetRecord, EvaluationCaseRecord)):
                if existing.contract != value.contract or (isinstance(existing, EvaluationDatasetRecord) and existing.content_expires_at != value.content_expires_at):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return existing
            if existing != value:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return existing
        await transaction.insert_record(self._stored(kind, identity, value, state=record_state(value)))
        return value

    async def _get(self, kind: str, identity: object, target: type[ValueT]) -> ValueT | None:
        record = await self._record(self._key(kind, identity))
        if record is None:
            tombstone_kind = "evaluation_tombstone" if kind == "evaluation" else "evaluation_content_tombstone"
            tombstone_id = identity if kind == "evaluation" else [kind, identity]
            if kind in {"evaluation", "evaluation_case", "evaluation_dataset", "evaluation_evidence", "evaluation_report", "evaluation_comparison"} and await self._record(self._key(tombstone_kind, tombstone_id)) is not None:
                raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
            return None
        return await self._decode(record, target)

    async def publish_dataset(
        self, dataset: DatasetContract, cases: tuple[CaseContract, ...],
        *, idempotency: IdempotencyRecord, owner_principal_id: str,
        content_expires_at: datetime | None = None,
    ) -> DatasetRef:
        async def write(transaction: StateTransaction) -> DatasetRef:
            identity = [idempotency.scope, idempotency.idempotency_key_digest]
            existing = await transaction.get_record(self._key("idempotency", identity))
            if existing is not None:
                receipt = await self._decode(existing, IdempotencyRecord)
                if receipt.request_digest != idempotency.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            else:
                await self._put(transaction, "idempotency", identity, idempotency)
            for case in cases:
                await self._put(transaction, "evaluation_case",
                                [case.ref.dataset_id, case.ref.case_id, case.ref.revision],
                                EvaluationCaseRecord(case, owner_principal_id, idempotency.created_at, content_expires_at))
            await self._put(transaction, "evaluation_dataset",
                            [dataset.ref.id, dataset.ref.revision],
                            EvaluationDatasetRecord(dataset, owner_principal_id, idempotency.created_at, content_expires_at))
            return dataset.ref
        async def readback() -> DatasetRef | None:
            current = await self.get_dataset(dataset.ref)
            if current != dataset:
                return None
            receipt = await self._get("idempotency", [idempotency.scope, idempotency.idempotency_key_digest], IdempotencyRecord)
            return dataset.ref if receipt is not None and receipt.request_digest == idempotency.request_digest else None
        return await self._commit(lambda: self._store.mutate(write), readback)

    async def get_dataset(self, ref: DatasetRef) -> DatasetContract | None:
        record = await self._get("evaluation_dataset", [ref.id, ref.revision], EvaluationDatasetRecord)
        if record is not None and record.content_expires_at is not None and record.content_expires_at <= datetime.now(timezone.utc):
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
        return None if record is None else record.contract

    async def _content_owner(self, kind: str, identity: object, target: type[ValueT]) -> str | None:
        row = await self._record(self._key(kind, identity))
        if row is not None:
            return (await self._decode(row, target)).owner_principal_id
        tombstone = await self._get("evaluation_content_tombstone", [kind, identity], EvaluationContentTombstone)
        return None if tombstone is None else tombstone.owner_principal_id

    async def dataset_owner(self, ref: DatasetRef) -> str | None:
        return await self._content_owner("evaluation_dataset", [ref.id, ref.revision], EvaluationDatasetRecord)

    async def case_owner(self, ref: CaseRef) -> str | None:
        return await self._content_owner("evaluation_case", [ref.dataset_id, ref.case_id, ref.revision], EvaluationCaseRecord)

    async def dataset_expiry(self, ref: DatasetRef) -> datetime | None:
        async def read(transaction: StateTransaction) -> datetime | None:
            row = await transaction.get_record(self._key("evaluation_dataset", [ref.id, ref.revision]))
            if row is None:
                raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
            dataset = await self._decode(row, EvaluationDatasetRecord)
            deadlines = [] if dataset.content_expires_at is None else [dataset.content_expires_at]
            for reference in dataset.contract.ordered_case_refs:
                row = await transaction.get_record(self._key("evaluation_case", [reference.dataset_id, reference.case_id, reference.revision]))
                if row is None:
                    raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
                case = await self._decode(row, EvaluationCaseRecord)
                if case.content_expires_at is not None:
                    deadlines.append(case.content_expires_at)
            return min(deadlines) if deadlines else None
        return await self._store.read(read)

    async def get_case(self, ref: CaseRef) -> CaseContract | None:
        record = await self._get("evaluation_case", [ref.dataset_id, ref.case_id, ref.revision], EvaluationCaseRecord)
        if record is not None and record.content_expires_at is not None and record.content_expires_at <= datetime.now(timezone.utc):
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
        return None if record is None else record.contract

    async def reserve_experiment(self, record: EvaluationRecord) -> EvaluationRecord:
        async def write(transaction: StateTransaction) -> EvaluationRecord:
            scope = "evaluation.run"
            identity = [scope, record.idempotency_key_digest]
            existing = await transaction.get_record(self._key("idempotency", identity))
            if existing is not None:
                reservation = await self._decode(existing, IdempotencyRecord)
                if reservation.request_digest != record.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                stored = await transaction.get_record(self._key("evaluation", reservation.resource_id))
                if stored is None:
                    if await transaction.get_record(self._key("evaluation_tombstone", reservation.resource_id)) is not None:
                        raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_record = await self._decode(stored, EvaluationRecord)
                self._require_content(existing_record, datetime.now(timezone.utc))
                return existing_record
            reservation = IdempotencyRecord(
                scope, record.idempotency_key_digest, record.request_digest,
                ResourceKind.EVALUATION, record.experiment_id, IdempotencyStatus.COMPLETED,
                record.manifest_digest, None, record.created_at, record.created_at,
            )
            await self._put(transaction, "idempotency", identity, reservation)
            await self._put(transaction, "evaluation", record.experiment_id, record)
            return record
        async def readback() -> EvaluationRecord | None:
            receipt = await self._get("idempotency", ["evaluation.run", record.idempotency_key_digest], IdempotencyRecord)
            if receipt is None or receipt.request_digest != record.request_digest:
                return None
            current = await self.get(receipt.resource_id, tenant_id=self.tenant_id)
            if current is not None:
                self._require_content(current, datetime.now(timezone.utc))
            return current
        return await self._commit(lambda: self._store.mutate(write), readback)

    async def get(self, experiment_id: str, *, tenant_id: str) -> EvaluationRecord | None:
        if tenant_id != self.tenant_id:
            return None
        return await self._get("evaluation", experiment_id, EvaluationRecord)

    async def update(
        self, experiment_id: str,
        change: Callable[[EvaluationRecord], EvaluationRecord],
    ) -> EvaluationRecord:
        async def write(transaction: StateTransaction) -> EvaluationRecord:
            current = await transaction.get_record(self._key("evaluation", experiment_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, EvaluationRecord)
            changed = change(value)
            if changed == value:
                return value
            if (changed.manifest != value.manifest or
                    changed.request_digest != value.request_digest or
                    changed.manifest_digest != value.manifest_digest or
                    changed.owned_input_captures != value.owned_input_captures):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            changed = replace(changed, revision=value.revision + 1,
                              updated_at=datetime.now(timezone.utc))
            await replace_checked(transaction, projected_record(self, current, changed),
                                  current.storage_version)
            return changed
        async def readback() -> EvaluationRecord | None:
            current = await self.get(experiment_id, tenant_id=self.tenant_id)
            if current is None:
                return None
            proposed = change(current)
            return current if proposed == current else None
        return await self._commit(lambda: self._store.mutate(write), readback)

    async def register_launch_intent(
        self, experiment_id: str, intent: EvaluationLaunchIntent, *, capacity: int,
    ) -> EvaluationRecord:
        def register(record: EvaluationRecord) -> EvaluationRecord:
            self._require_content(record, datetime.now(timezone.utc))
            current = next((item for item in record.intents if item.slot_id == intent.slot_id), None)
            if current is not None:
                if (current.submission != intent.submission or
                        current.trial != intent.trial or
                        current.scorer_slot_id != intent.scorer_slot_id):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return record
            if any(item.slot_id == intent.slot_id and item.disposition.terminal for item in record.dispositions):
                return record
            active = sum(not item.released and
                         (item.scorer_slot_id is None) == (intent.scorer_slot_id is None)
                         for item in record.intents)
            if record.gate != "open" or active >= capacity:
                return record
            return replace(record, intents=(*record.intents, intent),
                           dispositions=tuple(item for item in record.dispositions if item.slot_id != intent.slot_id))
        return await self.update(experiment_id, register)

    async def append_slot_disposition(
        self, experiment_id: str, item: EvaluationSlotDisposition,
    ) -> EvaluationRecord:
        def append(record: EvaluationRecord) -> EvaluationRecord:
            intent = next((value for value in record.intents if value.slot_id == item.slot_id), None)
            if intent is not None and not intent.released:
                return record
            current = next((value for value in record.dispositions if value.slot_id == item.slot_id), None)
            if current is not None and (current.disposition.terminal or current == item):
                return record
            return replace(record, dispositions=(
                *tuple(value for value in record.dispositions if value.slot_id != item.slot_id), item))
        return await self.update(experiment_id, append)

    async def close_reservation_gate(
        self, experiment_id: str, *, budget: bool = False,
    ) -> EvaluationRecord:
        return await self.update(experiment_id, lambda value: replace(
            value, gate="closed_budget" if budget else "closed_cancel")
            if value.gate == "open" or not budget and value.gate == "closed_budget" else value)

    async def settle_intent(
        self, experiment_id: str, slot_id: str, *, confirmed: bool, released: bool,
    ) -> EvaluationRecord:
        return await self.update(experiment_id, lambda value: replace(
            value, intents=tuple(replace(item,
                confirmed=item.confirmed or confirmed, released=item.released or released)
                if item.slot_id == slot_id else item for item in value.intents)))

    async def publish_trial_evidence(
        self, experiment_id: str, evidence: EvaluationTrialEvidence,
    ) -> EvaluationRecord:
        def publish(record: EvaluationRecord) -> EvaluationRecord:
            self._require_content(record, datetime.now(timezone.utc))
            if any(item.trial == evidence.trial for item in record.evidence):
                return record
            return replace(record, evidence=(*record.evidence, evidence))
        return await self.update(experiment_id, publish)

    async def publish_evidence(self, evidence: EvidenceBundle) -> EvidenceBundle:
        return await self._store.mutate(lambda tx: self._put(
            tx, "evaluation_evidence", evidence.ref.evidence_id, evidence))

    async def read_evidence(self, ref: EvidenceRef) -> EvidenceBundle | None:
        return await self._get("evaluation_evidence", ref.evidence_id, EvidenceBundle)

    async def publish_report(
        self, report: EvaluationReport | ComparisonReport,
    ) -> EvaluationReport | ComparisonReport:
        kind = "evaluation_report" if isinstance(report, EvaluationReport) else "evaluation_comparison"
        return await self._store.mutate(lambda tx: self._put(tx, kind, report.report_id, report))

    def _require_content(self, record: EvaluationRecord, now: datetime) -> None:
        if (record.content_deleted_at is not None or
                record.content_expires_at is not None and record.content_expires_at <= now or
                record.metadata_expires_at is not None and record.metadata_expires_at <= now):
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)

    async def list_expired(
        self, *, now: datetime, limit: int, owner_principal_id: str | None = None,
    ) -> tuple[EvaluationRecord, ...]:
        found = []
        cursor = None
        while len(found) < limit:
            records = await self._records("evaluation", cursor=cursor, limit=1000)
            for stored in records:
                record = await self._decode(stored, EvaluationRecord)
                if owner_principal_id is not None and record.manifest.principal.principal_id != owner_principal_id:
                    continue
                source_expired = await self._store.read(lambda tx: self._source_expired(tx, record, now))
                if ((record.content_deleted_at is None and (source_expired or record.content_expires_at is not None and record.content_expires_at <= now))
                        or record.metadata_expires_at is not None and record.metadata_expires_at <= now):
                    found.append(record)
                    if len(found) == limit:
                        break
            if len(records) < 1000:
                break
            cursor = record_cursor(records[-1])
        return tuple(found)

    async def _source_expired(self, transaction: StateTransaction, record: EvaluationRecord, now: datetime) -> bool:
        source_id = record.manifest.source_experiment_id
        seen = {record.experiment_id}
        while source_id is not None:
            if source_id in seen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(source_id)
            row = await transaction.get_record(self._key("evaluation", source_id))
            if row is None:
                return True
            source = await self._decode(row, EvaluationRecord)
            if source.content_deleted_at is not None or any(
                deadline is not None and deadline <= now
                for deadline in (source.content_expires_at, source.metadata_expires_at)
            ):
                return True
            source_id = source.manifest.source_experiment_id
        ref = record.manifest.dataset
        row = await transaction.get_record(self._key("evaluation_dataset", [ref.id, ref.revision]))
        if row is None:
            return await transaction.get_record(self._key("evaluation_content_tombstone", ["evaluation_dataset", [ref.id, ref.revision]])) is not None
        dataset = await self._decode(row, EvaluationDatasetRecord)
        if dataset.content_expires_at is not None and dataset.content_expires_at <= now:
            return True
        for case_ref in dataset.contract.ordered_case_refs:
            identity = [case_ref.dataset_id, case_ref.case_id, case_ref.revision]
            row = await transaction.get_record(self._key("evaluation_case", identity))
            if row is None:
                if await transaction.get_record(self._key("evaluation_content_tombstone", ["evaluation_case", identity])) is not None:
                    return True
                continue
            case = await self._decode(row, EvaluationCaseRecord)
            if case.content_expires_at is not None and case.content_expires_at <= now:
                return True
        return False

    async def purge_expired_datasets(
        self, *, now: datetime, owner_principal_id: str, limit: int,
    ) -> tuple[DatasetRef, ...]:
        async def write(transaction: StateTransaction) -> tuple[DatasetRef, ...]:
            deleted = []
            count = 0
            for kind, target in (("evaluation_dataset", EvaluationDatasetRecord), ("evaluation_case", EvaluationCaseRecord)):
                for row in await transaction.list_records(RecordQuery(kind=kind)):
                    value = await self._decode(row, target)
                    if value.owner_principal_id != owner_principal_id or value.content_expires_at is None or value.content_expires_at > now:
                        continue
                    if count >= limit:
                        return tuple(deleted)
                    ref = value.contract.ref
                    identity = ([ref.id, ref.revision] if isinstance(ref, DatasetRef)
                                else [ref.dataset_id, ref.case_id, ref.revision])
                    await transaction.delete_record(row.key_digest, expected_storage_version=row.storage_version)
                    await self._put(transaction, "evaluation_content_tombstone", [kind, identity],
                        EvaluationContentTombstone(kind, identity, owner_principal_id, value.contract.digest, now))
                    count += 1
                    if isinstance(ref, DatasetRef):
                        deleted.append(ref)
            return tuple(deleted)
        return await self._store.mutate(write)

    async def purge(self, experiment_id: str, *, now: datetime,
                    objects: tuple[tuple[RuntimeDomain, ObjectRef], ...] = ()) -> tuple[tuple[RuntimeDomain, ObjectRef], ...]:
        additional_objects = objects
        async def write(transaction: StateTransaction) -> tuple[tuple[RuntimeDomain, ObjectRef], ...]:
            stored = await transaction.get_record(self._key("evaluation", experiment_id))
            cleanup_stored = await transaction.get_record(self._key("evaluation_cleanup", experiment_id))
            cleanup = None if cleanup_stored is None else await self._decode(cleanup_stored, EvaluationCleanupRecord)
            if stored is None:
                return () if cleanup is None else cleanup.objects
            record = await self._decode(stored, EvaluationRecord)
            metadata_due = record.metadata_expires_at is not None and record.metadata_expires_at <= now
            content_due = record.content_deleted_at is None and (await self._source_expired(transaction, record, now)
                or record.content_expires_at is not None and record.content_expires_at <= now)
            if not (metadata_due or content_due):
                return () if cleanup is None else cleanup.objects
            if record.gate == "open" or any(not item.released for item in record.intents):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            owner = record.manifest.principal.principal_id
            objects = set((*additional_objects, *(() if cleanup is None else cleanup.objects)))
            objects.update(iter_runtime_object_refs(encode_domain(record), default_domain=RuntimeDomain.EVALUATION))
            for kind, target in (("evaluation_evidence", EvidenceBundle), ("evaluation_report", EvaluationReport),
                                 ("evaluation_comparison", ComparisonReport)):
                rows = await transaction.list_records(RecordQuery(kind=kind))
                for row in rows:
                    value = await self._decode(row, target)
                    belongs = (value.trial.target_experiment_id == experiment_id if isinstance(value, EvidenceBundle)
                               else value.experiment_id == experiment_id if isinstance(value, EvaluationReport)
                               else experiment_id in {value.baseline.experiment_id, value.candidate.experiment_id,
                                                      *(item.experiment_id for item in value.cutoff.scoring)})
                    if not belongs:
                        continue
                    identity = value.ref.evidence_id if isinstance(value, EvidenceBundle) else value.report_id
                    digest = value.ref.digest if isinstance(value, EvidenceBundle) else record.manifest_digest
                    objects.update(iter_runtime_object_refs(encode_domain(value), default_domain=RuntimeDomain.EVALUATION))
                    await transaction.delete_record(row.key_digest, expected_storage_version=row.storage_version)
                    await self._put(transaction, "evaluation_content_tombstone", [kind, identity],
                                    EvaluationContentTombstone(kind, identity, owner, digest, now))
            if metadata_due:
                await transaction.delete_record(stored.key_digest, expected_storage_version=stored.storage_version)
                await self._put(transaction, "evaluation_tombstone", experiment_id,
                    EvaluationTombstone(experiment_id, owner, record.request_digest,
                                        record.idempotency_key_digest, record.manifest_digest, now))
            else:
                def numerical_score(score: ScoreBundle) -> ScoreBundle:
                    return ScoreBundle(dimensions={name: ScoreNotApplicable(reason="evidence_deleted")
                        if isinstance(value, ScoreNotApplicable) else value for name, value in score.dimensions.items()})
                scores = tuple(replace(item, score=None if item.score is None else numerical_score(item.score),
                    evidence_ref=None, reason=None if item.reason is None else "evidence_deleted") for item in record.scores)
                decisions = tuple(replace(item, score=numerical_score(item.score)) for item in record.human_decisions)
                target_trials = (tuple((record.experiment_id, trial.trial_id) for trial in record.manifest.trials)
                                 if record.manifest.kind == "experiment" else
                                 tuple((trial.target_experiment_id, trial.trial_id) for trial in record.manifest.source_trials))
                slots = (*("target:" + trial.trial_id for trial in record.manifest.trials),
                         *("score:" + trial_id + ":" + scorer.slot_id for _, trial_id in target_trials
                           for scorer in record.manifest.scorers))
                dispositions = tuple(EvaluationSlotDisposition(slot,
                    SlotDispositionView("permanent_unavailable", "evidence_deleted", True, False, now)) for slot in slots)
                candidates = tuple(replace(candidate, graph_template=replace(candidate.graph_template, template=None))
                    if candidate.graph_template is not None else candidate for candidate in record.manifest.candidates)
                changed = replace(record, manifest=replace(record.manifest, candidates=candidates),
                                  owned_input_captures=(), intents=(), evidence=(), scores=scores, human_decisions=decisions,
                                  dispositions=dispositions, content_deleted_at=now,
                                  revision=record.revision + 1, updated_at=now)
                await replace_checked(transaction, projected_record(self, stored, changed), stored.storage_version)
            candidates = tuple(sorted(objects, key=lambda item: (item[0].value, item[1].key, item[1].digest)))
            updated = EvaluationCleanupRecord(experiment_id, owner, candidates, now if cleanup is None else cleanup.created_at)
            if cleanup_stored is None:
                await self._put(transaction, "evaluation_cleanup", experiment_id, updated)
            else:
                await replace_checked(transaction, projected_record(self, cleanup_stored, updated), cleanup_stored.storage_version)
            return candidates
        return await self._store.mutate(write)

    async def pending_cleanup(self, *, owner_principal_id: str, limit: int | None) -> tuple[EvaluationCleanupRecord, ...]:
        found = []
        cursor = None
        while limit is None or len(found) < limit:
            rows = await self._records("evaluation_cleanup", cursor=cursor, limit=1000)
            for row in rows:
                value = await self._decode(row, EvaluationCleanupRecord)
                if value.owner_principal_id == owner_principal_id and value.objects:
                    found.append(value)
                    if limit is not None and len(found) == limit:
                        break
            if len(rows) < 1000:
                break
            cursor = record_cursor(rows[-1])
        return tuple(found)

    async def acknowledge_cleanup(
        self, experiment_id: str, objects: tuple[tuple[RuntimeDomain, ObjectRef], ...],
    ) -> None:
        async def write(transaction: StateTransaction) -> None:
            row = await transaction.get_record(self._key("evaluation_cleanup", experiment_id))
            if row is None:
                return
            value = await self._decode(row, EvaluationCleanupRecord)
            updated = replace(value, objects=tuple(item for item in value.objects if item not in objects))
            if not updated.objects:
                await transaction.delete_record(row.key_digest, expected_storage_version=row.storage_version)
            elif updated != value:
                await replace_checked(transaction, projected_record(self, row, updated), row.storage_version)
        await self._store.mutate(write)

    async def get_report_at(self, cutoff: EvaluationReadCutoff) -> EvaluationReport | None:
        cursor = None
        while True:
            records = await self._records("evaluation_report", cursor=cursor, limit=1000,
                scope=self._scope("evaluation_report", "experiment", cutoff.experiment_id))
            for record in records:
                report = await self._decode(record, EvaluationReport)
                if report.cutoff == cutoff:
                    return report
            if len(records) < 1000:
                return None
            cursor = record_cursor(records[-1])

    async def get_report(self, report_id: str) -> EvaluationReport | ComparisonReport | None:
        return (await self._get("evaluation_report", report_id, EvaluationReport)
                or await self._get("evaluation_comparison", report_id, ComparisonReport))


__all__ = ["EvaluationRepositoryImpl"]
