#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable snapshots cannot silently omit optional capability partitions."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, StorageRoute


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
@pytest.mark.parametrize("custom", (False, True))
async def test_portable_snapshot_rejects_custom_partitions_only(tmp_path: Path, backend: str, custom: bool) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else RuntimeStorage.filesystem(tmp_path / "state")
    route = StorageRoute.sqlite(tmp_path / "state.db") if backend == "sqlite" else StorageRoute.filesystem(tmp_path / "state")
    routes = {"app.audit": route} if custom else {}
    source = RuntimeState(storage)
    await source.initialize(namespace="snapshot-capabilities", tenant_id="tenant", capability_routes=routes)
    await source.close()
    offline = RuntimeState(storage)
    await offline.initialize(namespace="snapshot-capabilities", tenant_id="tenant", capability_routes=routes, read_only=True)
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
    from linktools.ai.capability import RuntimeFeatures
    from linktools.ai.core import CommitPolicy
    from linktools.ai.runtime import RuntimeSnapshot, RuntimeHistory
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
    async with RuntimeHistory.open(
        "atomic-capabilities", tenant_id="default", features=features,
        storage=(RuntimeStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else RuntimeStorage.filesystem(tmp_path / "state")),
    ) as history:
        assert history is not None
    with pytest.raises(AIError) as rejected:
        await RuntimeSnapshot.create(
            "atomic-capabilities", tenant_id="default",
            storage=(RuntimeStorage.sqlite(tmp_path / "state.db") if backend == "sqlite" else RuntimeStorage.filesystem(tmp_path / "state")),
            object_store=InMemoryObjectStore(), features=features if with_features else None,
            exclusive=Offline(), limits=SnapshotLimits(max_entries=1000, max_bytes=1_000_000),
        )
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert "Capability persistence partitions" in str(rejected.value)
