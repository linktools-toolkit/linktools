#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opaque capability state cannot prove that a Runtime object is unreferenced."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from linktools.ai.core import CommitPolicy, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.capabilities import CapabilityWriter, RuntimeCapabilityWrites
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, ObjectStore, StorageRoute

from .test_archive_closure import _prepare
from .test_evaluation_retention import Offline

pytestmark = pytest.mark.asyncio


def _storage(root: Path, objects: ObjectStore, backend: str, *, volatile: bool = False) -> PersistenceStorage:
    storage = (PersistenceStorage.filesystem(root / 'state', object_store=objects) if backend == 'filesystem'
               else PersistenceStorage.sqlite(root / 'state.db', object_store=objects))
    if volatile:
        storage = PersistenceStorage(replace(storage.plan, capability_routes={'opaque': StorageRoute.memory()}),
                                     object_store=objects)
    return storage


def _record(payload: Mapping[str, JsonValue]) -> StoredRecord:
    return StoredRecord(hashlib.sha256(b'opaque-record').digest(), None, None, 'opaque', 'opaque', 'retained',
                        0, None, 0, None, payload)


@dataclass
class _Owner:
    commit: CommitPolicy = CommitPolicy.sync()
    writer: CapabilityWriter | None = None

    def bind_persistence(self, writer: CapabilityWriter) -> '_Owner':
        return replace(self, writer=writer)

    async def persist_in_transaction(self, transaction: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
        await transaction.insert_record(_record(payload))


@pytest.mark.parametrize('backend', ['filesystem', 'sqlite'])
@pytest.mark.parametrize('volatile', [False, True])
async def test_cleanup_blocks_live_custom_partition_without_changing_core_cleanup(
    tmp_path: Path, backend: str, volatile: bool,
) -> None:
    objects = FilesystemObjectStore(tmp_path / 'objects') if backend == 'filesystem' else InMemoryObjectStore()
    guard = objects if backend == 'filesystem' else Offline()
    state = Persistence(_storage(tmp_path, objects, backend, volatile=volatile))
    await state.initialize(namespace='opaque-cleanup', tenant_id='tenant')
    try:
        maintenance = state.snapshot_operations()
        orphan = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                          PersistenceDomain.EXECUTION, 'tenant', b'ordinary orphan')
        candidate = ((PersistenceDomain.EXECUTION, orphan),)
        if not volatile:
            assert (await maintenance.purge_unreferenced_objects(candidate, exclusive=guard)).deleted == candidate
        reference = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                             PersistenceDomain.EXECUTION, 'tenant', b'opaque retained content')
        partition = await state.capability_store('opaque')
        await partition.mutate(lambda tx: tx.insert_record(_record({'artifact_handle': reference.key})))
        candidate = ((PersistenceDomain.EXECUTION, reference),)
        result = await maintenance.purge_unreferenced_objects(candidate, exclusive=guard)
        assert result.blocked == candidate
        assert not result.deleted and not result.missing and not result.retained
        assert await objects.stat(reference.key) is not None
        assert (await partition.read(lambda tx: tx.scan_records()))[0].data['artifact_handle'] == reference.key
        assert not (await maintenance.purge_unreferenced_objects((), exclusive=guard)).blocked
        with pytest.raises(AIError) as rejected:
            await maintenance.purge_unreferenced_objects(candidate, exclusive=guard, limit=0)
        assert rejected.value.code is ErrorCode.PAGE_LIMIT_INVALID
    finally:
        await state.close()


@pytest.mark.parametrize('backend', ['filesystem', 'sqlite'])
@pytest.mark.parametrize('separate_resource', [False, True])
async def test_cleanup_reopened_capability_marker_blocks_object_and_archive_deletion(
    tmp_path: Path, backend: str, separate_resource: bool,
) -> None:
    objects = InMemoryObjectStore()
    storage = _storage(tmp_path, objects, backend)
    plain_storage = storage
    if separate_resource:
        storage = PersistenceStorage(replace(storage.plan, capability_routes={
            'opaque': StorageRoute.sqlite(tmp_path / 'custom.db'),
        }), object_store=objects)
    state = Persistence(storage)
    await state.initialize(namespace='opaque-cleanup', tenant_id='tenant')
    manager = None
    try:
        reference = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                             PersistenceDomain.EXECUTION, 'tenant', b'opaque retained content')
        group = state.execution.executions.state_store.storage_group
        scope, archive_store = await group.open_archive(create=True)
        archive = ContentArchive(archive_store, scope, namespace='opaque-cleanup', tenant_id='tenant', object_store=objects)
        archive_root, _ = await _prepare(archive, (b'opaque retained archive',))
        archive_before = await archive_store.read(lambda tx: tx.scan_records())
        partition = await state.capability_store('opaque')
        manager = RuntimeCapabilityWrites({'opaque': partition}, namespace='opaque-cleanup', tenant_id='tenant',
            durable_capability_ids=frozenset({'opaque'}), require_active_run=lambda run, execution: None,
            has_active_execution=lambda execution: False, marker_store=state.capability_marker_store())
        bound = manager.bind('opaque', 'run', 'execution', _Owner())
        assert bound.writer is not None
        await bound.writer.submit(key='write', payload={'artifact_handle': reference.key,
                                                       'archive_handle': json.dumps(archive_root.to_json())})
    finally:
        if manager is not None:
            await manager.close()
        await state.close()
    reopened = Persistence(plain_storage)
    await reopened.initialize(namespace='opaque-cleanup', tenant_id='tenant')
    try:
        assert reopened.capability_stores == {}
        assert not reopened.durable_capability_ids
        candidate = ((PersistenceDomain.EXECUTION, reference),)
        result = await reopened.purge_unreferenced_objects(candidate, exclusive=Offline())
        assert result.blocked == candidate
        assert not result.deleted and not result.missing and not result.retained
        assert await objects.stat(reference.key) is not None
        _, archived = await reopened.execution.executions.state_store.storage_group.open_archive()
        assert await archived.read(lambda tx: tx.scan_records()) == archive_before
    finally:
        await reopened.close()


async def test_cleanup_shared_sql_core_tenants_remain_collectable(tmp_path: Path) -> None:
    from linktools.ai.persistence.contracts import ArtifactRecord
    from .test_evaluation_retention import NOW

    objects = InMemoryObjectStore()
    storage = _storage(tmp_path, objects, 'sqlite')
    first, foreign = Persistence(storage), Persistence(storage)
    await first.initialize(namespace='opaque-cleanup', tenant_id='tenant')
    await foreign.initialize(namespace='other-namespace', tenant_id='other-tenant')
    try:
        retained = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                            PersistenceDomain.EXECUTION, 'tenant', b'foreign core reference')
        orphan = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                          PersistenceDomain.EXECUTION, 'tenant', b'ordinary orphan')
        await foreign.artifact.records.put_metadata(ArtifactRecord(
            'artifact', 'execution', 'producer', 'application/octet-stream', retained, NOW))
        assert not (await first.purge_unreferenced_objects((), exclusive=Offline())).blocked
        candidates = tuple((PersistenceDomain.EXECUTION, ref) for ref in (retained, orphan))
        result = await first.purge_unreferenced_objects(candidates, exclusive=Offline())
        assert result.retained == candidates[:1]
        assert result.deleted == candidates[1:]
        assert not result.blocked
        assert await objects.stat(retained.key) is not None
        assert await objects.stat(orphan.key) is None
    finally:
        await foreign.close()
        await first.close()


@pytest.mark.parametrize('fault', ['malformed', 'unavailable'])
async def test_cleanup_marker_failure_precedes_archive_and_object_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from linktools.ai.persistence import _object_cleanup
    from linktools.ai.persistence._capability_marker import _KEY, ensure_capability_partition_marker

    objects = InMemoryObjectStore()
    state = Persistence(_storage(tmp_path, objects, 'sqlite'))
    await state.initialize(namespace='opaque-cleanup', tenant_id='tenant')
    try:
        reference = await put_runtime_object(objects, ObjectKeyFactory('opaque-cleanup'),
                                             PersistenceDomain.EXECUTION, 'tenant', b'opaque retained content')
        group = state.execution.executions.state_store.storage_group
        scope, archived = await group.open_archive(create=True)
        archive = ContentArchive(archived, scope, namespace='opaque-cleanup', tenant_id='tenant', object_store=objects)
        await _prepare(archive, (b'opaque retained archive',))
        archive_before = await archived.read(lambda tx: tx.scan_records())
        marker = state.capability_marker_store()
        assert marker is not None
        await marker.mutate(ensure_capability_partition_marker)
        failure = RuntimeError('marker source unavailable')
        if fault == 'malformed':
            async def corrupt(transaction: StateTransaction) -> None:
                current = await transaction.get_operation(_KEY)
                assert current is not None
                assert await transaction.replace_operation(replace(current, data={'invalid': True}),
                                                           expected_state=current.state)
            await marker.mutate(corrupt)
        else:
            async def failed(transaction: StateTransaction) -> bool:
                raise failure
            monkeypatch.setattr(_object_cleanup, 'has_capability_partition_marker', failed)
        with pytest.raises(AIError if fault == 'malformed' else RuntimeError) as rejected:
            await state.purge_unreferenced_objects(((PersistenceDomain.EXECUTION, reference),), exclusive=Offline())
        if fault == 'malformed':
            assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            assert rejected.value is failure
        assert await objects.stat(reference.key) is not None
        assert await archived.read(lambda tx: tx.scan_records()) == archive_before
    finally:
        await state.close()
