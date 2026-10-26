#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable snapshots cannot silently omit optional capability partitions."""

from pathlib import Path
from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, StorageRoute


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
@pytest.mark.parametrize("custom", (False, True))
async def test_portable_snapshot_rejects_custom_partitions_only(tmp_path: Path, backend: str, custom: bool) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else PersistenceStorage.filesystem(tmp_path / "state")
    route = StorageRoute.sqlite(tmp_path / "state.db") if backend == "sqlite" else StorageRoute.filesystem(tmp_path / "state")
    routes = {"app.audit": route} if custom else {}
    storage = PersistenceStorage(replace(storage.plan, capability_routes=routes))
    source = Persistence(storage)
    await source.initialize(namespace="snapshot-capabilities", tenant_id="tenant")
    await source.close()
    offline = Persistence(storage)
    await offline.initialize(namespace="snapshot-capabilities", tenant_id="tenant", read_only=True)
    try:
        archive = InMemoryObjectStore()
        limits = SnapshotLimits(max_entries=100, max_bytes=100_000)
        if custom:
            with pytest.raises(AIError) as rejected:
                await offline.export_snapshot(object_store=archive, limits=limits)
            assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
            assert "Capability persistence partitions" in str(rejected.value)
        else:
            assert await offline.export_snapshot(object_store=archive, limits=limits)
    finally:
        await offline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
@pytest.mark.parametrize("with_features", (False, True))
@pytest.mark.parametrize("separate_resource", (False, True))
async def test_public_snapshot_detects_committed_custom_partitions_without_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_features: bool, separate_resource: bool, backend: str,
) -> None:
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import CommitPolicy
    from linktools.ai.snapshot import Snapshot
    from linktools.ai.history import History
    from .test_capability_atomic_writes import _opened
    from .test_runtime_snapshot_operations import Offline

    route = StorageRoute.sqlite(tmp_path / "custom.db") if separate_resource else None
    async with _opened(tmp_path, monkeypatch, backend=backend, policies=(CommitPolicy.sync(), CommitPolicy.sync()),
                       routes=(route, route)) as opened:
        audit, _ = await opened.start()
        await audit.submit("snapshot-marker")
        await opened.finish()
        features = RuntimeFeatures("snapshot-capabilities")
        for template in opened.templates:
            features.capability(template)
    async with Persistence.open('atomic-capabilities', storage=PersistenceStorage.sqlite(tmp_path / 'state.db') if backend == 'sqlite' else PersistenceStorage.filesystem(tmp_path / 'state'), tenant_id='default', read_only=True) as history_persistence_65, History.open(persistence=history_persistence_65, authorization=None) as history:
        assert history is not None
    storage = PersistenceStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else PersistenceStorage.filesystem(tmp_path / "state")
    if with_features:
        storage = PersistenceStorage(replace(storage.plan, capability_routes={
            identity: route for identity in ("audit", "index") if route is not None
        }))
    async with Persistence.open("atomic-capabilities", storage=storage, read_only=True) as persistence:
        with pytest.raises(AIError) as rejected:
            await Snapshot.create(persistence=persistence, object_store=InMemoryObjectStore(),
                                  exclusive=Offline(), limits=SnapshotLimits(max_entries=1000, max_bytes=1_000_000))
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert "Capability persistence partitions" in str(rejected.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
@pytest.mark.parametrize("with_features", (False, True))
@pytest.mark.parametrize("mixed", (False, True))
async def test_volatile_custom_writes_do_not_block_durable_core_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, with_features: bool, mixed: bool,
) -> None:
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import CommitPolicy
    from linktools.ai.snapshot import Snapshot
    from .test_capability_atomic_writes import _opened
    from .test_runtime_snapshot_operations import Offline

    routes = (StorageRoute.memory(), None if mixed else StorageRoute.transient())
    async with _opened(tmp_path, monkeypatch, backend=backend,
                       policies=(CommitPolicy.sync(), CommitPolicy.sync()), routes=routes) as opened:
        audit, index = await opened.start()
        await audit.submit("volatile-value")
        await index.submit("second-value")
        assert len(await opened.probe.facts("audit")) == 1
        assert len(await opened.probe.facts("index")) == 1
        assert opened.runtime._state.durable_capability_ids == (frozenset({"index"}) if mixed else frozenset())
        await opened.finish()
        features = RuntimeFeatures("snapshot-capabilities")
        for template in opened.templates:
            features.capability(template)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else PersistenceStorage.filesystem(tmp_path / "state")
    if with_features:
        storage = PersistenceStorage(replace(storage.plan, capability_routes={
            identity: route for identity, route in zip(("audit", "index"), routes) if route is not None
        }))
    async with Persistence.open("atomic-capabilities", storage=storage, read_only=True) as persistence:
        options = dict(persistence=persistence, object_store=InMemoryObjectStore(),
                       exclusive=Offline(), limits=SnapshotLimits(max_entries=1000, max_bytes=1_000_000))
        if mixed:
            with pytest.raises(AIError) as rejected:
                await Snapshot.create(**options)
            assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        else:
            assert await Snapshot.create(**options)
