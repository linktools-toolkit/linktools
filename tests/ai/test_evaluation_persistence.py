#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation facts share backend transactions and preserve every planned slot."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.core import IdempotencyStatus, ResourceKind, TaskStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateContract, CaseContract, CaseRef, DatasetContract, DatasetRef,
    DimensionContract, EvaluationManifest, EvaluationPolicy, ScorerContract,
    ScoreAttemptView, SlotDispositionView, TargetTrialRef, TaskCaseInput, TrialPlan, TrialView,
    evaluation_completion,
)
from linktools.ai.runtime._evaluation import RuntimeEvaluations
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime.state._contracts import IdempotencyRecord
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain, _encode_persisted_domain, decode_domain, encode_domain,
)
from linktools.ai.runtime.state._evaluation_records import (
    EvaluationCleanupRecord, EvaluationLaunchIntent, EvaluationRecord, EvaluationSlotDisposition,
    EvaluationTombstone,
)
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskGraphState, TaskGraphSubmission, TaskNode, TaskNodeView, TaskRef


NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)
PRINCIPAL = service_principal("tenant", "owner")


def storage_for(kind: str, path: Path) -> RuntimeStorage:
    if kind == "memory":
        return RuntimeStorage.in_memory()
    if kind == "filesystem":
        return RuntimeStorage.filesystem(path)
    if kind == "sqlite":
        return RuntimeStorage.sqlite(path / "state.sqlite")
    return RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=StorageRoute.sqlite(path / "execution.sqlite"),
        task=StorageRoute.filesystem(path / "task"),
        evaluation=StorageRoute.sqlite(path / "evaluation.sqlite"),
    ))


def experiment() -> EvaluationRecord:
    manifest = EvaluationManifest(
        "experiment", "experiment", None, DatasetRef("dataset", 1),
        (CandidateContract("target", TaskRef("target", 1), None, ()),),
        (ScorerContract("score", TaskRef("score", 1), {},
                        (DimensionContract("quality", "number", "higher"),), {}),),
        (TrialPlan("trial", CaseRef("dataset", "case", 1), "target", 1),), (),
        EvaluationPolicy(), "fixed_input", PRINCIPAL,
    )
    return EvaluationRecord(manifest, manifest.digest, "a" * 64, "b" * 64, "open", 0, NOW, NOW)


def intent() -> EvaluationLaunchIntent:
    graph = TaskGraph("trial-graph", (TaskNode("target", task=TaskRef("target", 1)),))
    admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, "trial-start"))
    return EvaluationLaunchIntent("target:trial", TargetTrialRef("experiment", "trial"), None,
                                  TaskGraphSubmission("evaluation", admission, graph), None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (
    TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.SUCCEEDED,
    TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED,
))
async def test_score_collection_reads_fresh_record_only_for_terminal_node(
    status: TaskStatus, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(RuntimeEvaluations)
    record = experiment()
    graph = TaskGraph("score-graph", (TaskNode("score", task=TaskRef("score", 1)),))
    admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, "score-start"))
    launch = EvaluationLaunchIntent("score:trial:score", TargetTrialRef("experiment", "trial"),
        "score", TaskGraphSubmission("evaluation", admission, graph), None, confirmed=True)
    state = TaskGraphState(graph.graph_id, status, graph.nodes, (
        TaskNodeView(graph.graph_id, "score", (), status, None, 0, None, None, None, None),
    ))
    saved = ScoreAttemptView("experiment", "attempt", launch.trial, "score", TaskRef("score", 1), "error")
    fresh = replace(record, scores=(saved,), revision=1)
    reads: list[str] = []

    async def read(experiment_id, principal, *, allow_expired):
        assert principal is PRINCIPAL
        assert allow_expired
        reads.append(experiment_id)
        return fresh

    monkeypatch.setattr(service, "_record", read)
    await service._collect_score(record, launch, state)
    assert reads == ([] if status in {TaskStatus.PENDING, TaskStatus.RUNNING} else ["experiment"])
    assert record.scores == ()
    assert fresh.scores == (saved,)


def test_evaluation_identity_round_trips_through_record_and_retention_wire() -> None:
    record = experiment()
    restored = decode_domain(encode_domain(record), EvaluationRecord)
    assert restored == record
    assert restored.experiment_id == record.manifest.experiment_id

    tombstone = EvaluationTombstone(record.experiment_id, "owner", record.request_digest,
        record.idempotency_key_digest, record.manifest_digest, NOW)
    cleanup = EvaluationCleanupRecord(record.experiment_id, "owner", (), NOW)
    for value, wire_id, fields in (
        (tombstone, "evaluation_tombstone", {
            "evaluation_id": record.experiment_id,
            "owner_principal_id": "owner",
            "request_digest": record.request_digest,
            "idempotency_key_digest": record.idempotency_key_digest,
            "manifest_digest": record.manifest_digest,
            "deleted_at": {"$datetime": NOW.isoformat()},
        }),
        (cleanup, "evaluation_cleanup_record", {
            "evaluation_id": record.experiment_id,
            "owner_principal_id": "owner",
            "objects": {"$tuple": []},
            "created_at": {"$datetime": NOW.isoformat()},
        }),
    ):
        wire = {"$dataclass": wire_id, "fields": fields}
        assert encode_domain(value) == wire
        assert decode_domain(wire, type(value)) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_dataset_publication_and_slot_arbitration_are_backend_atomic(backend: str, tmp_path: Path) -> None:
    storage = storage_for(backend, tmp_path)
    storage = RuntimeState(storage)
    await storage.initialize(namespace="evaluation", tenant_id="tenant")
    try:
        repository = storage.evaluation.records
        case = CaseContract(CaseRef("dataset", "case", 1), TaskCaseInput(input={"value": 1}))
        dataset = DatasetContract(DatasetRef("dataset", 1), (case.ref,), "task_input")
        receipt = IdempotencyRecord("evaluation.dataset.publish", "c" * 64, "d" * 64,
            ResourceKind.EVALUATION, "dataset", IdempotencyStatus.COMPLETED, dataset.digest, None, NOW, NOW)
        await repository.publish_dataset(dataset, (case,), idempotency=receipt, owner_principal_id="owner")
        assert await repository.dataset_owner(dataset.ref) == "owner"
        with pytest.raises(AIError) as conflict:
            await repository.publish_dataset(dataset, (replace(case, input=TaskCaseInput(input={"value": 2})),),
                idempotency=replace(receipt, idempotency_key_digest="e" * 64), owner_principal_id="owner")
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await repository.get_case(case.ref) == case
        assert await storage.evaluation.idempotency.get(receipt.scope, "e" * 64, tenant_id="tenant") is None
        await repository.reserve_experiment(experiment())
        pending = intent()
        disposition = EvaluationSlotDisposition(pending.slot_id,
            SlotDispositionView("permanent_unavailable", "input_missing", True, False, NOW))
        await asyncio.gather(
            repository.register_launch_intent("experiment", pending, capacity=1),
            repository.append_slot_disposition("experiment", disposition),
        )
        record = await repository.get("experiment", tenant_id="tenant")
        assert bool(record.intents) != bool(record.dispositions)
        await repository.close_reservation_gate("experiment")
        rejected = replace(pending, slot_id="target:second")
        final = await repository.register_launch_intent("experiment", rejected, capacity=10)
        assert all(item.slot_id != rejected.slot_id for item in final.intents)
        assert final.manifest.trials == experiment().manifest.trials
        assert await repository.reserve_experiment(experiment()) == final
    finally:
        await storage.close()


@pytest.mark.parametrize("kind", (
    "permanent_unavailable", "permanent_invalid", "cancelled", "recoverable_blocked",
))
def test_disposition_kind_owns_flags_and_completion(kind: str) -> None:
    retryable = kind == "recoverable_blocked"
    disposition = SlotDispositionView(kind, "reason", not retryable, retryable, NOW)
    assert disposition == SlotDispositionView(kind=kind, reason_code="reason", created_at=NOW)
    assert disposition == replace(
        SlotDispositionView("recoverable_blocked", "reason", False, True, NOW), kind=kind,
    )
    assert disposition.terminal is not retryable
    assert disposition.retryable is retryable
    changed = replace(disposition, kind="cancelled" if retryable else "recoverable_blocked")
    assert changed.terminal is retryable
    assert changed.retryable is not retryable
    trial = TrialView(TargetTrialRef("experiment", "trial"), CaseRef("dataset", "case", 1),
                      "target", 1, disposition=disposition)
    assert trial.terminal is not retryable
    assert trial.needs_attention is retryable
    assert evaluation_completion((trial,), ()) == ("needs_attention" if retryable else "complete")
    with pytest.raises(ValueError):
        SlotDispositionView(kind, "reason", retryable, retryable, NOW)
    with pytest.raises(ValueError):
        SlotDispositionView(kind, "reason", not retryable, not retryable, NOW)


@pytest.mark.parametrize("kind,terminal,retryable", (
    ("unknown", True, False),
    ("cancelled", 1, False),
    ("recoverable_blocked", False, "true"),
))
def test_disposition_rejects_unknown_kind_and_non_boolean_flags(
    kind: str, terminal: object, retryable: object,
) -> None:
    with pytest.raises(ValueError):
        SlotDispositionView(kind, "reason", terminal, retryable, NOW)


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("kind", (
    "permanent_unavailable", "permanent_invalid", "cancelled", "recoverable_blocked",
))
def test_disposition_wire_preserves_validated_derived_flags(kind: str, persisted: bool) -> None:
    retryable = kind == "recoverable_blocked"
    disposition = SlotDispositionView(kind, "reason", not retryable, retryable, NOW)
    expected = {"$dataclass": "evaluation_slot_disposition_view", "fields": {
        "kind": kind, "reason_code": "reason", "terminal": not retryable,
        "retryable": retryable, "created_at": {"$datetime": NOW.isoformat()},
    }}
    if persisted:
        expected["schema"] = 1
        encoded = _encode_persisted_domain(disposition)

        def decode(payload: dict[str, object]) -> SlotDispositionView:
            return _decode_enveloped_domain({"v": 1, "value": {
                "type": "evaluation_slot_disposition_view", "payload": payload,
            }}, SlotDispositionView)
    else:
        encoded = encode_domain(disposition)

        def decode(payload: dict[str, object]) -> SlotDispositionView:
            return decode_domain(payload, SlotDispositionView)

    assert encoded == expected
    assert decode(encoded) == disposition
    encoded["fields"]["additional_projection"] = {"ordinary": "field"}
    assert decode(encoded) == disposition
    for name in ("terminal", "retryable"):
        for corrupt in (not getattr(disposition, name), 1, None):
            invalid = {**encoded, "fields": {**encoded["fields"], name: corrupt}}
            with pytest.raises(AIError) as raised:
                decode(invalid)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        invalid = {**encoded, "fields": dict(encoded["fields"])}
        del invalid["fields"][name]
        with pytest.raises(AIError) as raised:
            decode(invalid)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    invalid = {**encoded, "fields": {**encoded["fields"], "kind": "unknown"}}
    with pytest.raises(AIError) as raised:
        decode(invalid)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
@pytest.mark.parametrize("kind", (
    "permanent_unavailable", "permanent_invalid", "cancelled", "recoverable_blocked",
))
async def test_disposition_recovery_and_terminal_arbitration_persist(
    backend: str, kind: str, tmp_path: Path,
) -> None:
    storage = RuntimeState(storage_for(backend, tmp_path))
    await storage.initialize(namespace="evaluation", tenant_id="tenant")
    try:
        repository = storage.evaluation.records
        await repository.reserve_experiment(experiment())
        pending = intent()
        blocked = EvaluationSlotDisposition(pending.slot_id,
            SlotDispositionView("recoverable_blocked", "retry", False, True, NOW))
        await repository.append_slot_disposition("experiment", blocked)
        restored = await repository.get("experiment", tenant_id="tenant")
        assert restored.dispositions == (blocked,)
        disposition = EvaluationSlotDisposition(pending.slot_id,
            SlotDispositionView(kind, "reason", kind != "recoverable_blocked",
                                kind == "recoverable_blocked", NOW))
        await repository.append_slot_disposition("experiment", disposition)
        restored = await repository.get("experiment", tenant_id="tenant")
        assert restored.dispositions == (disposition,)
        await repository.append_slot_disposition("experiment", blocked)
        registered = await repository.register_launch_intent("experiment", pending, capacity=1)
        if kind == "recoverable_blocked":
            assert registered.intents == (pending,)
            assert registered.dispositions == ()
        else:
            assert registered.intents == ()
            assert registered.dispositions == (disposition,)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_content_purge_persists_terminal_disposition_projections(backend: str, tmp_path: Path) -> None:
    storage = RuntimeState(storage_for(backend, tmp_path))
    await storage.initialize(namespace="evaluation", tenant_id="tenant")
    try:
        repository = storage.evaluation.records
        await repository.reserve_experiment(replace(experiment(), gate="closed_cancel",
            content_expires_at=NOW, metadata_expires_at=NOW + timedelta(days=1)))
        await repository.purge("experiment", now=NOW)
        record = await repository.get("experiment", tenant_id="tenant")
        assert record.content_deleted_at == NOW
        assert {item.slot_id for item in record.dispositions} == {"target:trial", "score:trial:score"}
        assert all(item.disposition == SlotDispositionView(
            "permanent_unavailable", "evidence_deleted", True, False, NOW,
        ) for item in record.dispositions)
    finally:
        await storage.close()
