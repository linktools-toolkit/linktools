#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot maintenance preserves complete routed state and access guards."""

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.persistence.snapshots import (
    RuntimeSnapshotOperations,
    SnapshotResources,
    restore_snapshot,
)
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object


LIMITS = SnapshotLimits(max_entries=100, max_bytes=100_000)


def resources(*, read_only: bool = True, checkpoint: bool = False) -> SnapshotResources:
    objects = InMemoryObjectStore()
    domains = frozenset({PersistenceDomain.EXECUTION, PersistenceDomain.TRACE}
                        | ({PersistenceDomain.CHECKPOINT} if checkpoint else set()))
    return SnapshotResources(
        namespace="snapshot-operations", tenant_id="tenant",
        durable_domains=domains,
        stores={domain: InMemoryStateStore() for domain in domains},
        object_stores={domain: objects for domain in PersistenceDomain},
        read_only=read_only,
    )


def ready() -> None:
    pass


class Offline:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("write_failure", (False, True))
@pytest.mark.parametrize("checkpoint", (False, True))
async def test_snapshot_preserves_every_durable_domain_and_closes_restore_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_failure: bool, checkpoint: bool,
) -> None:
    source = resources(checkpoint=checkpoint)
    for store in source.stores.values():
        await store.initialize()
    archive = InMemoryObjectStore()
    ref = await RuntimeSnapshotOperations(source, require_ready=ready).export_snapshot(
        object_store=archive, limits=LIMITS,
    )
    manifest = json.loads(await read_object(archive, ref.key, expected_digest=ref.digest, expected_size=ref.size))
    assert set(manifest["domains"]) == {domain.value for domain in source.durable_domains}
    assert manifest["format_version"] == 1
    assert not {"native_messages", "native_checkpoints", "native_projections"}.intersection(manifest)
    restored = resources(read_only=False, checkpoint=checkpoint)
    for store in restored.stores.values():
        await store.initialize()
    closed = False
    if write_failure:
        async def fail_write(_mutation: object) -> None:
            raise OSError("target write failed")

        monkeypatch.setattr(restored.stores[PersistenceDomain.EXECUTION], "mutate", fail_write)

    @asynccontextmanager
    async def open_target(root: Path, namespace: str, tenant_id: str) -> AsyncIterator[SnapshotResources]:
        nonlocal closed
        assert root == tmp_path / "restored"
        assert (namespace, tenant_id) == (source.namespace, source.tenant_id)
        try:
            yield restored
        finally:
            closed = True

    if write_failure:
        with pytest.raises(OSError, match="target write failed"):
            await restore_snapshot(ref, object_store=archive, root=tmp_path / "restored",
                                   limits=LIMITS, open_target=open_target)
        assert closed
        return
    await restore_snapshot(ref, object_store=archive, root=tmp_path / "restored",
                           limits=LIMITS, open_target=open_target)
    assert closed
    roundtrip = await RuntimeSnapshotOperations(replace(restored, read_only=True), require_ready=ready).export_snapshot(
        object_store=archive, limits=LIMITS,
    )
    assert roundtrip == ref


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("export", "cleanup"))
async def test_snapshot_maintenance_rejects_closed_resources(operation: str) -> None:
    def closed() -> None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    maintenance = RuntimeSnapshotOperations(resources(), require_ready=closed)
    with pytest.raises(AIError) as raised:
        if operation == "export":
            await maintenance.export_snapshot(object_store=InMemoryObjectStore(), limits=LIMITS)
        else:
            await maintenance.purge_unreferenced_objects((), exclusive=Offline())
    assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_snapshot_maintenance_requires_correct_access_mode() -> None:
    with pytest.raises(AIError) as exported:
        await RuntimeSnapshotOperations(resources(read_only=False), require_ready=ready).export_snapshot(
            object_store=InMemoryObjectStore(), limits=LIMITS,
        )
    assert exported.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    with pytest.raises(AIError) as cleaned:
        await RuntimeSnapshotOperations(resources(), require_ready=ready).purge_unreferenced_objects(
            (), exclusive=Offline(),
        )
    assert cleaned.value.code is ErrorCode.STORAGE_READ_ONLY


@pytest.mark.asyncio
async def test_object_cleanup_requires_exclusivity_before_deletion() -> None:
    source = resources(read_only=False)
    objects = source.object_stores[PersistenceDomain.EXECUTION]

    async def chunks() -> AsyncIterator[bytes]:
        yield b"orphan"

    stat = await objects.put("orphan", chunks(), expected_size=6,
                             expected_digest=hashlib.sha256(b"orphan").hexdigest())
    ref = ObjectRef(objects.store_id, stat.key, stat.digest, stat.size)

    class Unavailable:
        @asynccontextmanager
        async def offline_exclusivity(self) -> AsyncIterator[None]:
            raise RuntimeError("writers still active")
            yield

    with pytest.raises(RuntimeError, match="writers still active"):
        await RuntimeSnapshotOperations(source, require_ready=ready).purge_unreferenced_objects(
            ((PersistenceDomain.EXECUTION, ref),), exclusive=Unavailable(),
        )
    assert await objects.stat("orphan") is not None


@pytest.mark.asyncio
async def test_snapshot_keeps_checkpoint_controls_without_a_capability(tmp_path: Path) -> None:
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.persistence import Persistence
    from linktools.ai.persistence.contracts import ExecutionRecoveryDetail
    from .test_execution_recovery_commands import _execution
    from dataclasses import replace
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    control = replace(_execution(now), recovery=ExecutionRecoveryDetail(), restart_checkpoints=False)
    configuration = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    writer = Persistence(configuration)
    await writer.initialize(namespace="checkpoint-snapshot", tenant_id="tenant")
    try:
        await writer.execution.executions.create_execution(control)
    finally:
        await writer.close()

    reader = Persistence(configuration)
    await reader.initialize(namespace="checkpoint-snapshot", tenant_id="tenant", read_only=True)
    archive = InMemoryObjectStore()
    try:
        reference = await reader.export_snapshot(object_store=archive, limits=LIMITS)
    finally:
        await reader.close()
    manifest = json.loads(await read_object(archive, reference.key,
        expected_digest=reference.digest, expected_size=reference.size))
    assert "execution" in {value["kind"] for value in manifest["domains"]["execution"]["records"]}

    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=restored_root, limits=LIMITS)
    restored = Persistence(PersistenceStorage.from_root(restored_root))
    await restored.initialize(namespace="checkpoint-snapshot", tenant_id="tenant", read_only=True)
    try:
        assert await restored.execution.executions.get("execution", tenant_id="tenant") == control
        roundtrip = await restored.export_snapshot(object_store=archive, limits=LIMITS)
    finally:
        await restored.close()
    from linktools.ai.persistence.snapshots import snapshot_manifest_identity
    roundtrip_manifest = json.loads(await read_object(archive, roundtrip.key,
        expected_digest=roundtrip.digest, expected_size=roundtrip.size))
    # Restore relocates archive scopes; compare the complete portable closure.
    assert snapshot_manifest_identity(roundtrip_manifest) == snapshot_manifest_identity(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("missing-export", "foreign-export", "missing-restore"))
async def test_pending_snapshot_requires_its_declared_checkpoint_owner(tmp_path: Path, route: str) -> None:
    from linktools.ai.core import ExecutionStatus, ToolOperationStatus
    from linktools.ai.execution import ExecutionRecoveryDetail, PendingDeferredCall, PendingToolContinuation
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.persistence.codec import decode_envelope, pending_operation_reference
    from linktools.ai.persistence.contracts import ToolOperationRecord
    from linktools.ai.storage import StoredPayload
    from .test_step_projection import _execution

    limits = SnapshotLimits(10000, 20_000_000)
    storage = PersistenceStorage.sqlite(tmp_path / "pending.db")
    arguments = StoredPayload.inline_json({"argument": True})
    current = _execution()
    pending = PendingToolContinuation("run", approvals=(
        PendingDeferredCall("call", "tool", arguments, operation_id="operation"),))
    current = replace(current, status=ExecutionStatus.WAITING_DEFERRED,
        recovery=ExecutionRecoveryDetail(agent_run_id="run", pending_tools=pending))
    operation = ToolOperationRecord("operation", "execution", "run", "call", "tool", arguments.digest,
        "binding", True, ToolOperationStatus.PENDING, None, 0, None,
        current.created_at, current.updated_at, arguments_payload=arguments)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        await state.checkpoint.tools.reserve(operation)
        await state.execution.executions.create_execution(current)
        repository = state.execution.executions
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert record is not None
        fields = decode_envelope(record.data).value["payload"]["fields"]
        argument = fields["recovery"]["fields"]["pending_tools"]["fields"]["arguments"]["0"]
        assert pending_operation_reference(argument) == "operation"

    archive = InMemoryObjectStore()
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        if route != "missing-restore":
            stores = {PersistenceDomain.EXECUTION: state.execution.executions.state_store}
            foreign = InMemoryStateStore() if route == "foreign-export" else None
            if foreign is not None:
                await foreign.initialize()
                stores[PersistenceDomain.CHECKPOINT] = foreign
            source = SnapshotResources("native-result", "tenant", frozenset(stores), stores,
                {domain: state.object_store(domain) for domain in stores}, read_only=True)
            try:
                with pytest.raises(AIError) as missing:
                    await RuntimeSnapshotOperations(source, require_ready=ready).export_snapshot(
                        object_store=archive, limits=limits)
                assert missing.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
            finally:
                if foreign is not None:
                    await foreign.close()
            return
        snapshot = await state.export_snapshot(object_store=archive, limits=limits)

    closed = False

    @asynccontextmanager
    async def open_target(root: Path, namespace: str, tenant_id: str) -> AsyncIterator[SnapshotResources]:
        nonlocal closed
        target = replace(resources(read_only=False), namespace=namespace, tenant_id=tenant_id)
        for store in target.stores.values():
            await store.initialize()
        try:
            yield target
        finally:
            closed = True
            for store in target.stores.values():
                await store.close()

    with pytest.raises(AIError) as missing:
        await restore_snapshot(snapshot, object_store=archive, root=tmp_path / "restored",
                               limits=limits, open_target=open_target)
    assert missing.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert closed
