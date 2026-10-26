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


def resources(*, read_only: bool = True) -> SnapshotResources:
    objects = InMemoryObjectStore()
    domains = frozenset({PersistenceDomain.EXECUTION, PersistenceDomain.TRACE})
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
async def test_snapshot_preserves_every_durable_domain_and_closes_restore_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_failure: bool,
) -> None:
    source = resources()
    for store in source.stores.values():
        await store.initialize()
    archive = InMemoryObjectStore()
    ref = await RuntimeSnapshotOperations(source, require_ready=ready).export_snapshot(
        object_store=archive, limits=LIMITS,
    )
    manifest = json.loads(await read_object(archive, ref.key, expected_digest=ref.digest, expected_size=ref.size))
    assert set(manifest["domains"]) == {"execution", "trace"}
    assert manifest["format_version"] == 1
    restored = resources(read_only=False)
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
async def test_snapshot_keeps_existing_checkpoint_without_a_module_declaration(tmp_path: Path) -> None:
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.persistence import Persistence
    from linktools.ai.persistence._step_contracts import AgentRunRecord

    configuration = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    writer = Persistence(configuration)
    await writer.initialize(namespace="checkpoint-snapshot", tenant_id="tenant")
    try:
        await writer.run_store.read_store(PersistenceDomain.CHECKPOINT).register_agent_run(AgentRunRecord("retained-run"))
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
    assert manifest["domains"]["checkpoint"]["records"]

    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=restored_root, limits=LIMITS)
    restored = Persistence(PersistenceStorage.from_root(restored_root))
    await restored.initialize(namespace="checkpoint-snapshot", tenant_id="tenant", read_only=True)
    try:
        roundtrip = await restored.export_snapshot(object_store=archive, limits=LIMITS)
    finally:
        await restored.close()
    assert roundtrip == reference
