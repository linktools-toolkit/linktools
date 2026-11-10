#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete native owners survive portable restore without replaying writers."""

import json
import hashlib
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import NativeCheckpoint, Persistence, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits, snapshot_manifest_identity
from linktools.ai.snapshot._snapshot import _storage_snapshot_usage
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object

from .test_execution_recovery_commands import _execution


LIMITS = SnapshotLimits(10000, 20_000_000)


async def _manifest(objects, reference):
    return json.loads(await read_object(objects, reference.key,
        expected_digest=reference.digest, expected_size=reference.size))


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ("empty", "inline", "object"))
async def test_native_checkpoint_roundtrip_keeps_retry_occurrences_and_tombstones(tmp_path: Path, content: str) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    objects = InMemoryObjectStore("portable")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with Persistence.open("native-backup", storage=storage, tenant_id="tenant") as state:
        native = await state.create_step_store(max_snapshots_per_run=1)
        try:
            await state.execution.executions.create_execution(replace(
                _execution(now), native_step_source=native.backup_source,
            ))
            await native.register_run(RunRecord(run_id="run"))
            event = StepEvent(run_id="run", kind="model_request_started", step_index=1, timestamp=now)
            await native.append_event(event)
            await native.append_event(event)
            if content != "empty":
                messages = [ModelRequest(parts=[UserPromptPart("payload" * (10000 if content == "object" else 1))])]
                for index in (1, 2):
                    snapshot = ContinuableSnapshot(run_id="run", step_index=index, messages=messages,
                                                   timestamp=now, idempotency_key=str(index))
                    await native.save_snapshot(NativeCheckpoint.encode(snapshot, output_retries_used=3))
        finally:
            await native.close()
    async with Persistence.open("native-backup", storage=storage, tenant_id="tenant", read_only=True) as reader:
        reference = await reader.export_snapshot(object_store=objects, limits=LIMITS)
    original = await _manifest(objects, reference)
    backup = original["native_checkpoints"][0]
    assert len(backup["events"]) == 2
    assert backup["events"][0]["key_digest"] != backup["events"][1]["key_digest"]
    if content != "empty":
        assert [row["pruned"] for row in backup["snapshots"]] == [True, False]
    entry_count, _ = _storage_snapshot_usage(original)
    assert entry_count >= len(backup["runs"]) + len(backup["events"]) + len(backup["snapshots"])
    if content != "empty":
        corrupted = deepcopy(original)
        row = corrupted["native_checkpoints"][0]["snapshots"][-1]
        descriptor = json.loads(row["messages"])
        descriptor["checkpoint"]["message_sha256"] = "0" * 64
        row["messages"] = canonical_json_bytes(descriptor).decode()
        payload = canonical_json_bytes(corrupted)
        digest = hashlib.sha256(payload).hexdigest()
        async def chunks():
            yield payload
        await objects.put("corrupt", chunks(), expected_size=len(payload), expected_digest=digest)
        rejected_root = tmp_path / "rejected"
        with pytest.raises(AIError) as rejected:
            await Persistence.restore_snapshot(ObjectRef(objects.store_id, "corrupt", digest, len(payload)),
                object_store=objects, root=rejected_root, limits=LIMITS)
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert not rejected_root.exists()
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=objects, root=root, limits=LIMITS)
    async with Persistence.open("native-backup", storage=PersistenceStorage.from_root(root),
                                tenant_id="tenant", read_only=True) as reader:
        execution = await reader.execution.executions.get("execution", tenant_id="tenant")
        assert execution.native_step_source.group_id != original["native_checkpoints"][0]["group_id"]
        native = await reader.checkpoint_backup_source(execution.native_step_source)
        if content != "empty":
            snapshots = await native.list_snapshots(run_id="run")
            assert len(snapshots) == 1
            assert NativeCheckpoint.decode(snapshots[0]).output_retries_used == 3
        repeated = await reader.export_snapshot(object_store=objects, limits=LIMITS)
    restored = await _manifest(objects, repeated)
    assert restored["native_checkpoints"][0] == {**backup,
        "group_id": restored["native_checkpoints"][0]["group_id"]}
    assert snapshot_manifest_identity(original) == snapshot_manifest_identity(restored)


@pytest.mark.asyncio
async def test_native_tables_without_admitted_source_do_not_prove_backup(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "unproven.db")
    async with Persistence.open("unproven", storage=storage, tenant_id="tenant") as state:
        native = await state.create_step_store()
        await native.close()
        await state.execution.executions.create_execution(_execution(datetime.now(timezone.utc)))
    async with Persistence.open("unproven", storage=storage, tenant_id="tenant", read_only=True) as reader:
        with pytest.raises(AIError) as failure:
            await reader.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        assert failure.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_runtime_default_checkpoint_owner_remains_native_after_restore(tmp_path: Path) -> None:
    from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability, TraceCapability
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from pydantic_ai.messages import ModelResponse, TextPart
    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("durable answer")])

    features = RuntimeFeatures("portable-runtime")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability())
    features.capability(StepPersistenceCapability())
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("portable-runtime", storage=storage) as state:
        async with Runtime.attach(persistence=state,
            features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question", idempotency_key="question")
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
    objects = InMemoryObjectStore("portable")
    async with Persistence.open("portable-runtime", storage=storage, read_only=True) as reader:
        reference = await reader.export_snapshot(object_store=objects, limits=LIMITS)
    manifest = await _manifest(objects, reference)
    assert manifest["native_checkpoints"][0]["snapshots"]
    root = tmp_path / "runtime-restored"
    await Persistence.restore_snapshot(reference, object_store=objects, root=root, limits=LIMITS)
    async with Persistence.open("portable-runtime", storage=PersistenceStorage.from_root(root), read_only=True) as reader:
        repeated = await reader.export_snapshot(object_store=objects, limits=LIMITS)
    restored = await _manifest(objects, repeated)
    assert snapshot_manifest_identity(manifest) == snapshot_manifest_identity(restored)
    assert restored["native_projections"] == [
        {**entry, "group_id": restored["native_checkpoints"][0]["group_id"]}
        for entry in manifest["native_projections"]
    ]
