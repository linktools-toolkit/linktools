#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation facts share backend transactions and preserve every planned slot."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import IdempotencyStatus, ResourceKind, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateContract, CaseContract, CaseRef, DatasetContract, DatasetRef,
    DimensionContract, EvaluationManifest, EvaluationPolicy, ScorerContract,
    SlotDispositionView, TargetTrialRef, TaskCaseInput, TrialPlan,
)
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime.state._contracts import IdempotencyRecord
from linktools.ai.runtime.state._evaluation_records import (
    EvaluationLaunchIntent, EvaluationRecord, EvaluationSlotDisposition,
)
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskGraphSubmission, TaskNode, TaskRef


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
        execution=RuntimeStorageRoute.sqlite(path / "execution.sqlite"),
        task=RuntimeStorageRoute.filesystem(path / "task"),
        evaluation=RuntimeStorageRoute.sqlite(path / "evaluation.sqlite"),
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "mixed"))
async def test_dataset_publication_and_slot_arbitration_are_backend_atomic(backend: str, tmp_path: Path) -> None:
    storage = storage_for(backend, tmp_path)
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
