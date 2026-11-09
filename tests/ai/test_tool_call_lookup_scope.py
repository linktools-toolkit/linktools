#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call lookups retain direct and recovered receipts in one read snapshot."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._checkpoint_repositories import ToolRepositoryImpl
from linktools.ai.storage import StoredPayload

from .test_tool_execution_boundary import _request


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return (PersistenceStorage.in_memory() if backend == "memory"
            else PersistenceStorage.sqlite(path / "tools.db"))


class _ReadTransaction:
    def __init__(self, transaction, transform=None):
        self.transaction = transaction
        self.transform = transform

    async def list_records(self, query):
        return await self.transaction.list_records(query)

    async def resolve_aliases(self, aliases):
        return await self.transaction.resolve_aliases(aliases)

    async def get_records(self, keys):
        records = await self.transaction.get_records(keys)
        return records if self.transform is None else self.transform(records)


class _ReadStore:
    def __init__(self, store, *, after_read=None, transform=None):
        self.store = store
        self.after_read = after_read
        self.transform = transform

    async def read(self, callback):
        result = await self.store.read(lambda tx: callback(_ReadTransaction(tx, self.transform)))
        if self.after_read is not None:
            after_read, self.after_read = self.after_read, None
            await after_read()
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_call_lookup_does_not_mix_direct_and_recovered_snapshots(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("lookup", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        direct = await tools.admit(replace(_request(), tool_call_id="direct"))
        request = replace(_request(), tool_operation_id="source", agent_run_id="source")
        source = await tools.admit(request)
        assert await tools.admit(replace(request, agent_run_id="run", recovery_agent_run_id="source")) == source

        async def complete_after_snapshot():
            for record in (direct, source):
                await tools.complete_payload(record.tool_operation_id, tenant_id="tenant", owner=record.owner,
                                             fence=record.fence, result_payload=StoredPayload.inline_text("result"))

        wrapped = ToolRepositoryImpl(
            _ReadStore(tools.state_store, after_read=complete_after_snapshot), namespace="lookup", tenant_id="tenant",
        )
        assert await wrapped.get_by_call_ids("run", ("call", "direct", "call", "missing"), tenant_id="tenant") == (source, direct)
        for record in (direct, source):
            assert (await tools.get_operation(record.tool_operation_id, tenant_id="tenant")).status is ToolOperationStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_call_lookup_preserves_direct_precedence_and_run_identity(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("lookup", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        source_request = replace(_request(), agent_run_id="source")
        source = await tools.admit(source_request)
        assert await tools.admit(replace(source_request, agent_run_id="run", recovery_agent_run_id="source")) == source
        assert await tools.get_by_call("run", "call", tenant_id="tenant") == source
        direct = await tools.admit(replace(_request(), tool_operation_id="direct"))
        other = await tools.admit(replace(_request(), tool_operation_id="other", agent_run_id="other", execution_id="other-execution"))
        assert await tools.get_by_call_ids("run", ("missing", "call", "call"), tenant_id="tenant") == (direct,)
        assert await tools.get_by_call("source", "call", tenant_id="tenant") == source
        assert await tools.get_by_call("other", "call", tenant_id="tenant") == other
        assert await tools.get_by_call("run", "call", tenant_id="other-tenant") is None
        assert await tools.get_by_call_ids("run", (), tenant_id="tenant") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("scope", ("read", "mutate"))
async def test_call_lookup_reuses_enclosing_state_scope(tmp_path: Path, backend: str, scope: str) -> None:
    async with Persistence.open("lookup", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        request = replace(_request(), agent_run_id="source")
        source = await tools.admit(request)
        await tools.admit(replace(request, agent_run_id="run", recovery_agent_run_id="source"))
        operation = tools.state_store.read if scope == "read" else tools.state_store.mutate
        values = await operation(lambda tx: tools.get_by_call_ids("run", ("call", "missing"), tenant_id="tenant"))
        assert values == (source,)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("damage", ("missing", "kind", "call", "payload"))
async def test_call_lookup_rejects_invalid_recovery_sources(tmp_path: Path, backend: str, damage: str) -> None:
    async with Persistence.open("lookup", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        request = replace(_request(), agent_run_id="source")
        source = await tools.admit(request)
        await tools.admit(replace(request, agent_run_id="run", recovery_agent_run_id="source"))
        other = await tools.admit(replace(_request(), tool_operation_id="other", agent_run_id="other", tool_call_id="other"))
        records = await tools.state_store.read(lambda tx: tx.scan_records())
        wrong_call = next(record for record in records if record.key_digest == tools._tool_key(other.tool_operation_id))

        def transform(records):
            if damage == "missing":
                return {}
            key, record = next(iter(records.items()))
            if damage == "kind":
                return {key: replace(record, kind="external_call")}
            if damage == "call":
                return {key: wrong_call}
            return {key: replace(record, data={})}

        wrapped = ToolRepositoryImpl(_ReadStore(tools.state_store, transform=transform), namespace="lookup", tenant_id="tenant")
        with pytest.raises(AIError) as rejected:
            await wrapped.get_by_call_ids("run", (source.tool_call_id,), tenant_id="tenant")
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_cancelled_call_lookup_releases_read_scope(tmp_path: Path, monkeypatch, backend: str) -> None:
    async with Persistence.open("lookup", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        source = await tools.admit(_request())
        started = asyncio.Event()
        original = tools._decode

        async def interrupted(record, target):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(tools, "_decode", interrupted)
        task = asyncio.create_task(tools.get_by_call("run", "call", tenant_id="tenant"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        monkeypatch.setattr(tools, "_decode", original)
        assert await asyncio.wait_for(tools.get_by_call("run", "call", tenant_id="tenant"), timeout=5) == source
