#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current SQL reads preserve complete mutations and staged sequence deletion."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import (
    FactQuery, OperationQuery, RecordQuery, StateTransaction,
    StoredFact, StoredOperation, StoredRecord,
)

pytestmark = pytest.mark.asyncio

def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str) -> StoredRecord:
    return StoredRecord(
        _key(value), None, None, "probe", (value).encode("ascii"), "pending", 0,
        None, 0, None, {"value": 0},
    )


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_sequence_cas_reads_own_writes_and_rolls_back(
    tmp_path: Path, backend: str,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.db"))
    key = _key("sequence")
    async with Persistence.open("sequence", storage=storage) as persistence:
        store = persistence.execution.executions.state_store
        assert await store.mutate(lambda tx: tx.next_sequence(key)) == 1

        async def advance(tx: StateTransaction) -> None:
            assert await tx.get_sequence(key) == 1
            assert await tx.advance_sequence(key, 1) == 2
            assert await tx.get_sequence(key) == 2
            assert await tx.get_sequences((key,)) == {key: 2}
            assert await tx.scan_sequences() == {key: 2}
            assert await tx.advance_sequence(key, await tx.get_sequence(key)) == 3
            with pytest.raises(AIError) as caught:
                await tx.advance_sequence(key, 2)
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            assert await tx.get_sequence(key) == 3

        await store.mutate(advance)

        async def abort(tx: StateTransaction) -> None:
            assert await tx.get_sequence(key) == 3
            assert await tx.advance_sequence(key, 3) == 4
            assert await tx.get_sequence(key) == 4
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as caught:
            await store.mutate(abort)
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda tx: tx.get_sequence(key)) == 3


async def test_live_reader_invalidates_mutable_and_negative_caches(
    tmp_path: Path,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    writer_owner, reader_owner = Persistence(storage), Persistence(storage)
    record = _record("record")
    stream, sequence, operation_key = map(_key, ("stream", "seq", "op"))
    operation = StoredOperation(operation_key, stream, 1, "pending", True, {})
    fact = StoredFact(stream, 1, record.key, "event", None, None, {})

    async def observe(transaction: StateTransaction) -> tuple[object, ...]:
        return (
            await transaction.get_record(record.key),
            await transaction.get_sequence(sequence),
            await transaction.list_facts(FactQuery(stream)),
            await transaction.get_operation(operation_key),
            await transaction.list_records(RecordQuery(kind="probe", limit=10)),
        )

    await writer_owner.initialize(namespace="live", tenant_id="tenant")
    await reader_owner.initialize(namespace="live", tenant_id="tenant", read_only=True)
    writer = writer_owner.execution.executions.state_store
    reader = reader_owner.execution.executions.state_store
    try:
        assert await reader.read(observe) == (None, 0, (), None, ())

        async def create(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
            await transaction.next_sequence(sequence)
            await transaction.insert_fact(fact)
            await transaction.insert_operation(operation)

        await writer.mutate(create)
        assert await reader.read(observe) == (
            record, 1, (fact,), operation, (record,),
        )
        updated = replace(record, storage_revision=1, data={"value": 1})
        completed = replace(operation, state="done")
        next_fact = replace(fact, sequence=2)

        async def update(transaction: StateTransaction) -> None:
            assert await transaction.replace_record(updated, expected_storage_revision=0)
            await transaction.next_sequence(sequence)
            await transaction.insert_fact(next_fact)
            assert await transaction.replace_operation(completed, expected_state="pending")

        await writer.mutate(update)
        assert await reader.read(observe) == (
            updated, 2, (fact, next_fact), completed, (updated,),
        )

        async def delete(transaction: StateTransaction) -> None:
            assert await transaction.delete_record(record.key)
            await transaction.delete_sequence(sequence)
            await transaction.delete_operations(OperationQuery(stream_key=stream))

        await writer.mutate(delete)
        assert await reader.read(observe) == (None, 0, (), None, ())
        await writer.mutate(create)
        assert await reader.read(observe) == (
            record, 1, (fact,), operation, (record,),
        )
    finally:
        await reader_owner.close()
        await writer_owner.close()


async def test_deleting_absent_and_unpublished_sequences_is_idempotent(
    tmp_path: Path,
) -> None:
    owner = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db"))
    await owner.initialize(namespace="live", tenant_id="tenant")
    writer = owner.execution.executions.state_store
    missing, transient, retained = map(_key, ("missing", "transient", "retained"))

    async def mutate(transaction: StateTransaction) -> None:
        await transaction.delete_sequences((missing, missing))
        await transaction.next_sequence(transient)
        await transaction.delete_sequences((transient, missing))
        assert await transaction.get_sequence(transient) == 0
        await transaction.next_sequence(retained)

    try:
        await writer.mutate(mutate)
        assert await writer.read(lambda transaction: transaction.scan_sequences()) == {retained: 1}
        await writer.mutate(lambda transaction: transaction.delete_sequences((missing, retained)))
        assert await writer.read(lambda transaction: transaction.scan_sequences()) == {}
        await writer.validate_integrity()
    finally:
        await owner.close()
