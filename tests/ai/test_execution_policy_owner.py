#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime policy owners remain distinct from Step payloads during recovery."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.execution import ExecutionRecord, ResultRecord
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive_closure import ArchiveClosure
from linktools.ai.persistence._execution_content import expand_snapshot_execution_content
from linktools.ai.persistence.codec import EXECUTION_POLICY_FIELDS, _decode_enveloped_domain, decode_envelope
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_archive_closure import Offline
from .test_history_projection_conformance import _record


@pytest.mark.asyncio
async def test_real_policy_owner_survives_offline_expansion_gc_and_snapshot(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "policy.db")
    objects = InMemoryObjectStore("policy-snapshot")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    value = replace(_record(ExecutionStatus.STARTED, 0), mode="plan", planning=True,
                    durable_trace=False, conversation=False)
    async with Persistence.open("policy-owner", storage=storage, tenant_id="tenant") as state:
        repository = state.execution.executions
        await repository.create_execution(value)
        raw = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        fields = decode_envelope(raw.data).value["payload"]["fields"]
        assert not set(EXECUTION_POLICY_FIELDS).intersection(fields)
        assert set(fields["binding"]) == {"$execution_binding", "archive_scope", "content_scope"}
        assert "runtime_overrides" not in fields
        scope, archive = repository.state_store.storage_group.opened_archive
        contents = ArchiveClosure(scope, await archive.read(lambda tx: tx.scan_records())).binding_contents()
        policy = fields["runtime_policy"]
        assert contents[(policy["archive_scope"], policy["content_scope"], policy["$runtime_policy"])].codec == "runtime-policy"
        semantic = expand_snapshot_execution_content(raw, contents, namespace="policy-owner", tenant_id="tenant")
        assert _decode_enveloped_domain(semantic.data, ExecutionRecord) == value
        await state.purge_unreferenced_objects((), exclusive=Offline())
        assert await repository.get_fresh(value.execution_id, tenant_id="tenant") == value
    async with Persistence.open("policy-owner", storage=storage, tenant_id="tenant", read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("policy-owner", storage=PersistenceStorage.from_root(tmp_path / "restored"),
                                tenant_id="tenant", read_only=True) as state:
        assert await state.execution.executions.get_fresh(value.execution_id, tenant_id="tenant") == value


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", (False, True))
async def test_result_projection_reads_its_fixed_conversation_policy(tmp_path: Path, fresh: bool) -> None:
    value = replace(_record(ExecutionStatus.SUCCEEDED, 0), durable_trace=False, conversation=False)
    value = replace(value, result=ResultRecord(StoredPayload.inline_json("result"), StopReason.END_TURN,
                                               UsageMetrics(), value.updated_at), completed_at=value.updated_at)
    async with Persistence.open("policy-result", storage=PersistenceStorage.sqlite(tmp_path / "result.db"), tenant_id="tenant") as state:
        await state.execution.executions.create_execution(value)
        result = await state.execution.executions.get_result_record(value.execution_id, tenant_id="tenant", fresh=fresh)
        assert result is not None and result.conversation is False
        assert result.result == value.result and result.completed_at == value.completed_at
