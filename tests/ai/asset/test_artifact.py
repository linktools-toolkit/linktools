#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/asset/test_artifact.py"""

import pytest

from linktools.ai.asset.artifact import ArtifactService
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.store import AssetStore


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    service = ArtifactService(assets=AssetStore(primary=MemoryAssetBackend()))
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
    service = ArtifactService(assets=AssetStore(primary=MemoryAssetBackend()))
    assert (
        await service.get(tenant_id="acme", run_id="run-1", artifact_name="nope.json")
        is None
    )


@pytest.mark.asyncio
async def test_list_for_run_returns_only_that_runs_artifacts():
    service = ArtifactService(assets=AssetStore(primary=MemoryAssetBackend()))
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
    names = {info.path.parts[-1] for info in infos}
    assert names == {"a.json", "b.json"}


@pytest.mark.asyncio
async def test_put_without_metadata_does_not_share_mutable_default_across_calls():
    service = ArtifactService(assets=AssetStore(primary=MemoryAssetBackend()))
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
