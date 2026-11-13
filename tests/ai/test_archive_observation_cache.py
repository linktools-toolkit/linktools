#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared content caches never inherit another route's observation policy."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._resource_scope import StorageResourceRequest, _shared_archive_cache
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.storage import InMemoryCacheBackend, StorageRoute, _write_cache

from .test_runtime_persistence_evidence import _execution


@pytest.mark.parametrize("configuration", ("same", "different", "uncached", "unrelated"))
def test_archive_cache_requires_coherent_participating_routes(configuration: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=100_000)
    other = InMemoryCacheBackend(max_bytes=100_000)
    requests = (
        StorageResourceRequest("execution", StorageRoute.memory(cache=cache), uses_archive=True),
        StorageResourceRequest("task", StorageRoute.memory(cache=(cache if configuration == "same" else
            other if configuration == "different" else None)), uses_archive=configuration != "unrelated"),
    )
    assert _shared_archive_cache(requests) is (cache if configuration in {"same", "unrelated"} else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", (False, True))
async def test_fresh_reads_reject_missing_source_behind_cached_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: bool,
) -> None:
    now = _write_cache.time()
    monkeypatch.setattr(_write_cache, "time", lambda: now)
    path = tmp_path / "state.db"
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with Persistence.open("cached-source", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(path, cache=cache)) as reader:
        expected = _execution()
        await reader.execution.executions.create_execution(expected)
        repository = reader.execution.executions
        assert await repository.get(expected.execution_id, tenant_id="tenant") == expected
        async with Persistence.open("cached-source", tenant_id="tenant",
                storage=PersistenceStorage.sqlite(path)) as writer:
            await writer.execution.executions.get(expected.execution_id, tenant_id="tenant")
            _, archive = writer.execution.executions.state_store.storage_group.opened_archive
            records = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
            policy, = (row for row in records if row.data.get("codec") == "runtime-policy")
            assert await archive.mutate(lambda tx: tx.delete_record(policy.key))
        assert await repository.get(expected.execution_id, tenant_id="tenant") == expected
        with pytest.raises(AIError) as failure:
            if batch:
                await repository.get_many_fresh((expected.execution_id,), tenant_id="tenant")
            else:
                await repository.get_fresh(expected.execution_id, tenant_id="tenant")
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
