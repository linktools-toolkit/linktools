#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_artifact.py"""
import pytest

from linktools.ai.storage.resource.artifact import ArtifactService
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.store import ResourceStore


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    service = ArtifactService(resources=ResourceStore(primary=MemoryResourceBackend()))
    await service.put(tenant_id="acme", run_id="run-1", artifact_name="report.json", content=b'{"ok": true}', content_type="application/json")
    resource = await service.get(tenant_id="acme", run_id="run-1", artifact_name="report.json")
    assert resource.content == b'{"ok": true}'
    assert resource.info.content_type == "application/json"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    service = ArtifactService(resources=ResourceStore(primary=MemoryResourceBackend()))
    assert await service.get(tenant_id="acme", run_id="run-1", artifact_name="nope.json") is None


@pytest.mark.asyncio
async def test_list_for_run_returns_only_that_runs_artifacts():
    service = ArtifactService(resources=ResourceStore(primary=MemoryResourceBackend()))
    await service.put(tenant_id="acme", run_id="run-1", artifact_name="a.json", content=b"1")
    await service.put(tenant_id="acme", run_id="run-1", artifact_name="b.json", content=b"2")
    await service.put(tenant_id="acme", run_id="run-2", artifact_name="c.json", content=b"3")
    infos = await service.list_for_run(tenant_id="acme", run_id="run-1")
    names = {info.path.parts[-1] for info in infos}
    assert names == {"a.json", "b.json"}
