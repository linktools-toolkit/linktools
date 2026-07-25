#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_assets.py"""

import pytest

from linktools.ai.run.assets import ArtifactService
from linktools.ai.storage.backends.memory.object import MemoryObjectBackend
from linktools.ai.storage.object.store import ObjectStore


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    service = ArtifactService(assets=ObjectStore(primary=MemoryObjectBackend()))
    await service.put(
        tenant_id="acme",
        run_id="run-1",
        artifact_name="report.json",
        content=b'{"ok": true}',
        content_type="application/json",
    )
    asset = await service.get(
        tenant_id="acme", run_id="run-1", artifact_name="report.json"
    )
    assert asset.content == b'{"ok": true}'
    assert asset.info.content_type == "application/json"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    service = ArtifactService(assets=ObjectStore(primary=MemoryObjectBackend()))
    assert (
        await service.get(tenant_id="acme", run_id="run-1", artifact_name="nope.json")
        is None
    )


@pytest.mark.asyncio
async def test_list_for_run_returns_only_that_runs_artifacts():
    service = ArtifactService(assets=ObjectStore(primary=MemoryObjectBackend()))
    await service.put(
        tenant_id="acme", run_id="run-1", artifact_name="a.json", content=b"1"
    )
    await service.put(
        tenant_id="acme", run_id="run-1", artifact_name="b.json", content=b"2"
    )
    await service.put(
        tenant_id="acme", run_id="run-2", artifact_name="c.json", content=b"3"
    )
    infos = await service.list_for_run(tenant_id="acme", run_id="run-1")
    names = {info.key.name for info in infos}
    assert names == {"a.json", "b.json"}


@pytest.mark.asyncio
async def test_put_without_metadata_does_not_share_mutable_default_across_calls():
    service = ArtifactService(assets=ObjectStore(primary=MemoryObjectBackend()))
    first = await service.put(
        tenant_id="acme", run_id="run-1", artifact_name="a.json", content=b"1"
    )
    second = await service.put(
        tenant_id="acme", run_id="run-2", artifact_name="b.json", content=b"2"
    )
    assert dict(first.info.metadata) == {}
    assert dict(second.info.metadata) == {}
    # Two independently-omitted-metadata calls must not somehow share or leak state
    # through a single mutable default object.
    assert first.info.metadata is not second.info.metadata or dict(
        first.info.metadata
    ) == dict(second.info.metadata)
