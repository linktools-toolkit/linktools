#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph budget owners survive portable snapshots without execution records."""

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import BudgetUsage, RunBudget, TaskStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain, decode_record, encode_domain, encode_envelope, encode_record,
)
from linktools.ai.runtime.state._store import RecordQuery, StateTransaction, StoredRecord
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object
from linktools.ai.task import TaskGraph, TaskGraphSubmission

from ._runtime_test_helpers import RuntimeUsageModels


_NAMESPACE = "graph-budget-snapshot"
_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)


async def _create_source(
    root: Path, phase: str,
) -> tuple[TaskGraphSubmission, BudgetUsage | None]:
    storage = RuntimeStorage.from_root(root)
    async with Runtime.open(
        _NAMESPACE, models=RuntimeUsageModels(), storage=storage,
    ) as runtime:
        engine = runtime.tasks.bind()
        graph = TaskGraph("empty", ())
        budget = RunBudget(model_requests=3)
        submission = await engine.describe_submission(
            graph, idempotency_key="snapshot", budget=budget,
        )
        with pytest.raises(AIError) as missing:
            await storage.execution.budgets.read("graph:empty")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        if phase == "described":
            return submission, None
        submission = await engine.prepare_submission(
            graph, idempotency_key="snapshot", budget=budget,
        )
        if phase == "admitted":
            assert (await engine.start_prepared(submission)).admitted
            assert (await (await engine.get("empty")).wait()).result.status is TaskStatus.SUCCEEDED
        records = await storage.execution.budgets.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="execution")),
        )
        assert not records
        return submission, await storage.execution.budgets.read("graph:empty")


async def _export_source(root: Path, objects: InMemoryObjectStore) -> ObjectRef:
    storage = RuntimeStorage.from_root(root)
    await storage.initialize(namespace=_NAMESPACE, tenant_id="default", read_only=True)
    try:
        return await storage.export_snapshot(object_store=objects, limits=_LIMITS)
    finally:
        await storage.close()


def _changed_limits(record: StoredRecord) -> StoredRecord:
    usage = _decode_enveloped_domain(record.data, BudgetUsage)
    value = encode_domain(replace(usage, limits=RunBudget(model_requests=4)))
    assert isinstance(value, dict)
    return replace(record, data=encode_envelope(value))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("described", "prepared", "admitted"))
async def test_empty_graph_budget_snapshot_preserves_submission_phase(
    phase: str, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    submission, expected_usage = await _create_source(source, phase)
    objects = InMemoryObjectStore("snapshot")
    reference = await _export_source(source, objects)
    restored = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        reference, object_store=objects, root=restored, limits=_LIMITS,
    )
    storage = RuntimeStorage.from_root(restored)
    async with Runtime.open(
        _NAMESPACE, models=RuntimeUsageModels(), storage=storage,
    ) as runtime:
        if phase == "described":
            with pytest.raises(AIError) as missing:
                await storage.execution.budgets.read("graph:empty")
            assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert await storage.task.admissions.submission_status(submission.ref) is None
        else:
            assert await storage.execution.budgets.read("graph:empty") == expected_usage
            assert await storage.task.admissions.submission_status(submission.ref) == phase
            engine = runtime.tasks.bind()
            assert (await engine.start_prepared(submission)).admitted
            run = await engine.get("empty")
            assert (await run.wait()).result.status is TaskStatus.SUCCEEDED
            assert await run.budget_usage() == expected_usage


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted"))
@pytest.mark.parametrize("corruption", ("missing", "limits"))
async def test_graph_budget_export_rejects_missing_or_changed_scope(
    phase: str, corruption: str, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    await _create_source(source, phase)
    storage = RuntimeStorage.from_root(source)
    await storage.initialize(namespace=_NAMESPACE, tenant_id="default")
    try:
        async def corrupt(transaction: StateTransaction) -> None:
            records = await transaction.list_records(RecordQuery(kind="budget_scope"))
            assert len(records) == 1
            record = records[0]
            assert await transaction.delete_record(record.key_digest)
            if corruption == "limits":
                await transaction.insert_record(_changed_limits(record))

        await storage.execution.budgets.state_store.mutate(corrupt)
    finally:
        await storage.close()
    with pytest.raises(AIError) as invalid:
        await _export_source(source, InMemoryObjectStore("snapshot"))
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted"))
@pytest.mark.parametrize("corruption", ("missing", "limits", "execution_domain"))
async def test_graph_budget_restore_rejects_missing_or_changed_scope_before_writing(
    phase: str, corruption: str, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    await _create_source(source, phase)
    objects = InMemoryObjectStore("snapshot")
    reference = await _export_source(source, objects)
    manifest = json.loads(await read_object(
        objects, reference.key, expected_digest=reference.digest, expected_size=reference.size,
    ))
    records = manifest["domains"]["execution"]["records"]
    assert len(records) == 1 and records[0]["kind"] == "budget_scope"
    if corruption == "missing":
        records.clear()
    elif corruption == "limits":
        records[0] = encode_record(_changed_limits(decode_record(records[0])))
    else:
        del manifest["domains"]["execution"]
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    key = "tampered/" + digest

    async def chunks() -> AsyncIterator[bytes]:
        yield payload

    await objects.put(key, chunks(), expected_size=len(payload), expected_digest=digest)
    restored = tmp_path / "restored"
    with pytest.raises(AIError) as invalid:
        await RuntimeStorage.restore_snapshot(
            ObjectRef(objects.store_id, key, digest, len(payload)),
            object_store=objects, root=restored, limits=_LIMITS,
        )
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not restored.exists()
