#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional exact fact reads preserve scope, transaction views and legacy readers."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence import _memory, _sql
from linktools.ai.persistence.state import FactReader, SqlStateFactReader, StateTransaction, StateTransactionNestingError, StoredFact

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from linktools.ai.storage import SqlStorageContext, SqlValue

from .test_incremental_history_read_cost import _seed_trace
from .test_state_batch_delete_backends import _digest, _fact, _open_store, _record


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_fact_reads_preserve_exact_scope_and_transaction_view(tmp_path: Path, backend: str) -> None:
    owner = _record(1)
    facts = tuple(_fact(owner, sequence) for sequence in (0, 2, 5))
    stream = facts[0].stream_key
    other_stream = _digest("other-stream")
    async with _open_store(backend, tmp_path / backend) as store:
        other = await store.storage_group.open_partition(_digest("other-store"), create=True)
        assert other is not None

        async def seed(transaction):
            assert isinstance(transaction, FactReader)
            await transaction.insert_record(owner)
            await transaction.insert_facts((*facts, replace(facts[1], stream_key=other_stream)))
            assert await transaction.get_facts(stream, (5, 1, 0, 2, 5)) == facts
            assert await transaction.get_facts(stream, ()) == ()

        await store.mutate(seed)
        await other.mutate(seed)
        await other.mutate(lambda tx: tx.delete_fact_streams(owner.key))

        async def read(transaction):
            assert isinstance(transaction, FactReader)
            assert await transaction.get_facts(stream, (5, 1, 0, 2, 5)) == facts
            assert await transaction.get_facts(other_stream, (0, 2, 5)) == (replace(facts[1], stream_key=other_stream),)
            assert await transaction.get_facts(_digest("absent"), (0, 2, 5)) == ()
            assert await transaction.get_facts(stream, ()) == ()
            with pytest.raises(StateTransactionNestingError):
                await transaction.delete_fact_streams(owner.key)

        await store.read(read)
        assert await other.read(lambda tx: tx.get_facts(stream, (0, 2, 5))) == ()

        async def abort(transaction):
            assert isinstance(transaction, FactReader)
            await transaction.delete_fact_streams(owner.key)
            assert await transaction.get_facts(stream, (0, 2, 5)) == ()
            assert await transaction.guard_record(owner.key, expected_storage_revision=0)
            await transaction.insert_fact(facts[1])
            assert await transaction.get_facts(stream, (0, 2, 5)) == (facts[1],)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError, match="STORAGE_CONFLICT"):
            await store.mutate(abort)
        await store.read(read)


@pytest.mark.asyncio
async def test_sql_fact_reads_bound_parameters_and_skip_empty_selections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owner = _record(1)
    facts = tuple(_fact(owner, sequence) for sequence in (0, 2, 5, 1100))
    async with _open_store("sql", tmp_path / "sql") as store:
        async def seed(transaction):
            await transaction.insert_record(owner)
            await transaction.insert_facts(facts)
        await store.mutate(seed)
        parameter_counts, materialized = [], []
        original = _sql._fact_from_row

        def materialize(row):
            value = original(row)
            materialized.append(value.sequence)
            return value

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "ai_state_facts" in statement:
                parameter_counts.append(len(parameters))
                assert len(parameters) <= 999

        monkeypatch.setattr(_sql, "_fact_from_row", materialize)
        engine = store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await store.read(lambda tx: tx.get_facts(facts[0].stream_key, ())) == ()
            assert parameter_counts == []
            selected = (1100, 0, 0, *range(10, 1100))
            assert await store.read(lambda tx: tx.get_facts(facts[0].stream_key, selected)) == (facts[0], facts[3])
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert materialized == [0, 1100]
        assert len(parameter_counts) == 2


class _LegacyTransaction:
    def __init__(self, transaction: StateTransaction) -> None:
        self._transaction = transaction

    def __getattr__(self, name: str) -> object:
        if name == "get_facts":
            raise AttributeError(name)
        return getattr(self._transaction, name)


class _CustomFactReader(_LegacyTransaction):
    async def get_facts(self, stream_key: bytes, sequences: Sequence[int]) -> tuple[StoredFact, ...]:
        return await self._transaction.get_facts(stream_key, sequences)


class _CustomSqlFactReader(_CustomFactReader):
    async def read_joined_facts(
        self, context: "SqlStorageContext", statement: "Select",
    ) -> "tuple[tuple[StoredFact, Mapping[str, SqlValue]], ...]":
        assert isinstance(self._transaction, SqlStateFactReader)
        return await self._transaction.read_joined_facts(context, statement)


def _custom_transactions(monkeypatch, backend, wrapper):
    store_type = _memory.InMemoryStateStore if backend == "memory" else _sql.SqlStateStore
    original = store_type.read

    async def read(store, callback):
        # Decorate the consumer view, not the physical transaction's active identity.
        return await original(store, lambda transaction: callback(wrapper(transaction)))

    monkeypatch.setattr(store_type, "read", read)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("capable", (False, True))
async def test_custom_fact_capability_and_legacy_fallback_keep_history_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, capable: bool,
) -> None:
    storage = Persistence(PersistenceStorage.in_memory() if backend == "memory"
                          else PersistenceStorage.sqlite(tmp_path / "custom.db"))
    try:
        archive, run_id = await _seed_trace(storage, 9)
        arguments = dict(agent_run_id=run_id, message_seqs=(), tool_call_ids=("call-4", "call-0", "call-4", "call-8"),
                         event_high_water=5)
        expected = await archive.read_history_associations(**arguments)
        wrapper = (_CustomSqlFactReader if backend == "sql" else _CustomFactReader) if capable else _LegacyTransaction
        _custom_transactions(monkeypatch, backend, wrapper)

        async def inspect(transaction):
            assert isinstance(transaction, FactReader) is capable
            if capable and backend == "sql":
                assert isinstance(transaction, SqlStateFactReader)
            with pytest.raises(StateTransactionNestingError):
                await transaction.delete_fact_streams(archive._agent_run_key(run_id))

        await archive.state_store.read(inspect)
        assert await archive.read_history_associations(**arguments) == expected
        assert [value.tool_call_id for value in expected] == ["call-0", "call-4"]

        async def remove(transaction):
            await transaction.delete_fact_streams(archive._agent_run_key(run_id))
        await archive.state_store.mutate(remove)
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(**arguments)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ("missing", "duplicate", "reversed", "unselected"))
async def test_history_rejects_inexact_custom_fact_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str) -> None:
    storage = Persistence(PersistenceStorage.in_memory())
    try:
        archive, run_id = await _seed_trace(storage, 5)

        class Reader(_CustomFactReader):
            async def get_facts(self, stream_key: bytes, sequences: Sequence[int]) -> tuple[StoredFact, ...]:
                facts = await super().get_facts(stream_key, sequences)
                if malformed == "missing":
                    return facts[:-1]
                if malformed == "duplicate":
                    return (*facts, facts[-1])
                if malformed == "reversed":
                    return tuple(reversed(facts))
                return (facts[0], replace(facts[-1], sequence=2))

        _custom_transactions(monkeypatch, "memory", Reader)
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(), tool_call_ids=("call-0", "call-4"), event_high_water=5,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()
