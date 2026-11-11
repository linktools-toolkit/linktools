#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact Tool subjects preserve direct precedence and recoverable record identity."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, _sql
from linktools.ai.persistence._snapshot_validation import canonical_snapshot_indexes, validate_snapshot_domain
from linktools.ai.persistence.codec import decode_record, encode_record
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import RecordQuery, StateTransaction, StoredAlias, alias_key, record_key
from linktools.ai.persistence.repositories import ToolRepositoryImpl
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore

from .test_runtime_record_cache import _record, _store
from .test_state_fact_reader import _custom_transactions
from .test_tool_execution_boundary import _request


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(path)


class _WithoutSubjectReads:
    def __init__(self, transaction: StateTransaction) -> None:
        self._transaction = transaction

    def __getattr__(self, name: str) -> object:
        if name == "read_record_subjects":
            raise AttributeError(name)
        return getattr(self._transaction, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("custom", (False, True))
async def test_tool_call_lookup_retains_direct_order_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, custom: bool,
) -> None:
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=_storage(backend, tmp_path / "calls.db")) as state:
        tools = state.checkpoint.tools
        request = replace(_request(), tool_operation_id="arbitrary-source", agent_run_id="original")
        source = await tools.admit(request)
        assert await tools.admit(replace(request, agent_run_id="resumed", recovery_agent_run_id="original")) == source
        for operation_id in ("first-direct", "last-direct"):
            await tools.admit(replace(request, agent_run_id="resumed", tool_operation_id=operation_id))
        direct = await tools.list_by_agent_run("resumed", tenant_id="tenant")
        source_only = await tools.admit(replace(request, tool_operation_id="source-only", tool_call_id="recovered"))
        assert await tools.admit(replace(request, tool_operation_id="source-only", tool_call_id="recovered",
                                        agent_run_id="resumed", recovery_agent_run_id="original")) == source_only
        values = {}
        for status in (ToolOperationStatus.PENDING, ToolOperationStatus.CANCELLED, ToolOperationStatus.EFFECT_UNKNOWN):
            values[status.value] = await tools.reserve(replace(source, tool_operation_id=f"reserved-{status.value}",
                                                              tool_call_id=status.value, agent_run_id="resumed", status=status))
        if custom:
            _custom_transactions(monkeypatch, backend, _WithoutSubjectReads)
        requested = ("CANCELLED", "call", "recovered", "EFFECT_UNKNOWN", "PENDING", "call", "missing")
        assert await tools.get_by_call_ids("resumed", requested, tenant_id="tenant") == (
            values["CANCELLED"], direct[-1], source_only, values["EFFECT_UNKNOWN"], values["PENDING"],
        )
        assert await tools.get_by_call_ids("resumed", requested, tenant_id="other") == ()
        assert await tools.get_by_call_ids("other-run", requested, tenant_id="tenant") == ()
        assert await tools.get_by_call_ids("resumed", (), tenant_id="tenant") == ()


@pytest.mark.asyncio
async def test_exact_tool_lookup_bounds_materialized_payloads(tmp_path: Path, monkeypatch, record_property) -> None:
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=_storage("sql", tmp_path / "calls.db")) as state:
        tools = state.checkpoint.tools
        for index in range(128):
            await tools.admit(replace(_request(), tool_operation_id=f"opaque-{index:03d}", tool_call_id=f"call-{index:03d}"))
        counters = Counter()
        original_row = _sql._record_from_row
        original_read = tools.state_store.read

        def materialize(row):
            counters["rows"] += 1
            counters["payload_bytes"] += len(row["payload_json"].encode("utf-8"))
            return original_row(row)

        async def read(callback):
            counters["read_transactions"] += 1
            return await original_read(callback)

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            counters[statement.split()[0].upper()] += 1

        monkeypatch.setattr(_sql, "_record_from_row", materialize)
        monkeypatch.setattr(tools.state_store, "read", read)
        engine = tools.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            baseline_records = await tools.list_by_agent_run("run", tenant_id="tenant")
            by_call = {value.tool_call_id: value for value in baseline_records}
            expected = (by_call["call-127"], by_call["call-000"])
            baseline = dict(counters)
            counters.clear()
            assert await tools.get_by_call_ids("run", ("call-127", "call-000", "call-127"), tenant_id="tenant") == expected
            candidate = dict(counters)
            counters.clear()
            assert await tools.get_by_call_ids("run", (), tenant_id="tenant") == ()
            assert counters == {}
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        record_property("tool_call_read_cost", json.dumps({"full_run": baseline, "exact_calls": candidate}))
        assert baseline["rows"] == 128
        assert candidate["rows"] == 2
        assert candidate["payload_bytes"] < baseline["payload_bytes"]
        assert candidate["read_transactions"] == baseline["read_transactions"] == 1
        assert candidate["SELECT"] == baseline["SELECT"] == 1
        assert not set(candidate) & {"INSERT", "UPDATE", "DELETE"}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_tool_subject_projection_roundtrips_and_rejects_corruption(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=_storage(backend, tmp_path / "calls.db")) as state:
        source = await state.checkpoint.tools.admit(_request())
        stored, = await state.checkpoint.tools.state_store.read(lambda tx: tx.list_records(RecordQuery(kind="tool_operation")))
        assert stored.subject_id == source.tool_call_id
        assert decode_record(encode_record(stored)) == stored
        for bad_subject in (None, "other-call"):
            with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
                await state.checkpoint.tools._decode(replace(stored, subject_id=bad_subject), type(source))
            corrupted = replace(stored, subject_id=bad_subject)
            indexes = dict(namespace="tool-locator", tenant_id="tenant", domain=PersistenceDomain.CHECKPOINT,
                           records=(corrupted,), facts=(), operations=())
            aliases, sequences = canonical_snapshot_indexes(**indexes)
            with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
                validate_snapshot_domain(**indexes, aliases=aliases, sequences=sequences)
        wire = encode_record(stored)
        del wire["subject"]
        with pytest.raises(AIError) as missing:
            decode_record(wire)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_tool_locator_reopen_and_same_version_snapshot(tmp_path: Path) -> None:
    storage = _storage("sql", tmp_path / "source.db")
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=storage) as state:
        request = replace(_request(), tool_operation_id="arbitrary-source", agent_run_id="original")
        source = await state.checkpoint.tools.admit(request)
        assert await state.checkpoint.tools.admit(replace(request, agent_run_id="resumed", recovery_agent_run_id="original")) == source
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=storage) as state:
        assert await state.checkpoint.tools.get_by_call("original", "call", tenant_id="tenant") == source
        assert await state.checkpoint.tools.get_by_call("resumed", "call", tenant_id="tenant") == source
    reader = Persistence(storage)
    await reader.initialize(namespace="tool-locator", tenant_id="tenant", read_only=True)
    archive = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    try:
        snapshot = await reader.export_snapshot(object_store=archive, limits=limits)
    finally:
        await reader.close()
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=archive, root=root, limits=limits)
    async with Persistence.open("tool-locator", tenant_id="tenant", storage=PersistenceStorage.from_root(root)) as state:
        assert await state.checkpoint.tools.get_by_call("original", "call", tenant_id="tenant") == source
        assert await state.checkpoint.tools.get_by_call("resumed", "call", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_record_subject_survives_shared_cache_codec(tmp_path: Path, monkeypatch, backend: str) -> None:
    async with _store(backend, tmp_path / backend, InMemoryCacheBackend(max_bytes=100000)) as store:
        record = replace(_record(1), subject_id="call")
        await store.mutate(lambda tx: tx.insert_record(record))
        original = store.read
        reads = 0

        async def read(callback):
            nonlocal reads
            reads += 1
            return await original(callback)

        monkeypatch.setattr(store, "read", read)
        assert await store.read_record(record.key) == record
        assert await store.read_record(record.key) == record
        assert reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_tool_aliases_preserve_exact_unicode_case_and_component_boundaries(tmp_path: Path, backend: str) -> None:
    pairs = (("a:b", "c"), ("a", "b:c"), ("Run", "call"), ("run", "call"),
             ("é", "call"), ("e\u0301", "call"), ("run", "Call"),
             ("界" * 42 + "ab", "é" * 64))
    async with Persistence.open("tool-alias", tenant_id="tenant", storage=_storage(backend, tmp_path / "aliases.db")) as state:
        tools = state.checkpoint.tools
        expected = []
        for index, (run_id, call_id) in enumerate(pairs):
            request = replace(_request(), tool_operation_id=f"operation-{index}", agent_run_id=f"source-{index}",
                              tool_call_id=call_id)
            source = await tools.admit(request)
            assert await tools.admit(replace(request, agent_run_id=run_id, recovery_agent_run_id=request.agent_run_id)) == source
            expected.append(source)
        for (run_id, call_id), source in zip(pairs, expected, strict=True):
            assert await tools.get_by_call(run_id, call_id, tenant_id="tenant") == source
        aliases = await tools.state_store.read(lambda tx: tx.scan_aliases())
        assert {value.alias_key for value in aliases} == {alias_key("tool-alias", "tenant", "checkpoint", "tool_call", pair) for pair in pairs}
        assert len(aliases) == len(pairs)
        for field in ("agent_run_id", "tool_call_id", "recovery_agent_run_id"):
            with pytest.raises(ValueError):
                replace(_request(), **{field: "é" * 65})
        for field in ("agent_run_id", "tool_call_id"):
            with pytest.raises(ValueError):
                replace(expected[0], **{field: "é" * 65})
        for run_id, call_id in (("é" * 65, "call"), ("run", "é" * 65)):
            with pytest.raises(AIError) as rejected:
                await tools.get_by_call(run_id, call_id, tenant_id="tenant")
            assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_tool_recovery_alias_cannot_claim_two_targets(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-alias", tenant_id="tenant", storage=_storage(backend, tmp_path / "aliases.db")) as state:
        tools = state.checkpoint.tools
        first_request = replace(_request(), agent_run_id="source-one")
        first = await tools.admit(first_request)
        second_request = replace(_request(), agent_run_id="source-two", tool_operation_id="second")
        second = await tools.admit(second_request)
        await tools.admit(replace(first_request, agent_run_id="recovered", recovery_agent_run_id="source-one"))
        with pytest.raises(AIError) as rejected:
            await tools.admit(replace(second_request, agent_run_id="recovered", recovery_agent_run_id="source-two"))
        assert rejected.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await tools.get_by_call("recovered", "call", tenant_id="tenant") == first
        assert await tools.get_operation("second", tenant_id="tenant") == second


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("custom", (False, True))
async def test_public_tool_repositories_isolate_recovery_aliases_on_one_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, custom: bool,
) -> None:
    async with Persistence.open("host", tenant_id="host-tenant", storage=_storage(backend, tmp_path / "shared.db")) as state:
        store = state.checkpoint.tools.state_store
        if custom:
            _custom_transactions(monkeypatch, backend, _WithoutSubjectReads)
        scopes = (("namespace", "tenant"), ("other", "tenant"), ("namespace", "other"))
        repositories = [ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant) for namespace, tenant in scopes]
        sources = []
        for index, (repository, (_, tenant)) in enumerate(zip(repositories, scopes, strict=True)):
            assert await repository.get_operation("same-operation", tenant_id=tenant) is None
            assert await repository.get_by_call("recovered", "call", tenant_id=tenant) is None
            request = replace(_request(), tool_operation_id="same-operation", agent_run_id="source", execution_id=f"execution-{index}")
            source = await repository.admit(request)
            assert await repository.admit(replace(request, agent_run_id="recovered", recovery_agent_run_id="source")) == source
            sources.append(source)
        for repository, (_, tenant), source in zip(repositories, scopes, sources, strict=True):
            assert await repository.get_operation("same-operation", tenant_id=tenant) == source
            assert await repository.get_by_call("recovered", "call", tenant_id=tenant) == source
        for namespace, tenant in scopes[1:]:
            corrupt = StoredAlias(alias_key(namespace, tenant, "checkpoint", "tool_call", ("foreign", "call")),
                                  record_key("namespace", "tenant", "checkpoint", "tool_operation", ("same-operation",)))
            async def insert_corrupt(transaction: StateTransaction) -> None:
                target = await transaction.get_record(corrupt.record_key)
                assert target is not None
                assert await transaction.guard_record(
                    target.key, expected_storage_revision=target.storage_revision,
                )
                await transaction.insert_alias(corrupt)

            await store.mutate(insert_corrupt)
            repository = ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant)
            with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
                await repository.get_by_call("foreign", "call", tenant_id=tenant)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("custom", (False, True))
async def test_public_tool_relations_isolate_shared_run_and_execution_on_one_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, custom: bool,
) -> None:
    async with Persistence.open("host", tenant_id="host", storage=_storage(backend, tmp_path / "relations.db")) as state:
        store = state.checkpoint.tools.state_store
        if custom:
            _custom_transactions(monkeypatch, backend, _WithoutSubjectReads)
        scopes = (("namespace", "tenant"), ("other", "tenant"), ("namespace", "other"))
        repositories = [ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant) for namespace, tenant in scopes]
        values = [await repository.admit(replace(_request(), tool_operation_id=f"operation-{index}",
                                                agent_run_id="shared-run", execution_id="shared-execution"))
                  for index, repository in enumerate(repositories)]
        for repository, (_, tenant), value in zip(repositories, scopes, values, strict=True):
            assert await repository.list_by_agent_run("shared-run", tenant_id=tenant) == (value,)
            assert await repository.list_by_execution("shared-execution", tenant_id=tenant) == (value,)
            assert await repository.get_by_call_ids("shared-run", (value.tool_call_id,), tenant_id=tenant) == (value,)
