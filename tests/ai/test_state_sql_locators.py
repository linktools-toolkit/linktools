#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL lookup projections never replace the complete State identity."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import _sql
from linktools.ai.persistence.state import (
    FactQuery, OperationQuery, StateTransaction, StoredAlias, StoredFact,
    StoredOperation, StoredRecord,
)

from .test_state_batch_delete_backends import _open_store

pytestmark = pytest.mark.asyncio

_RECORD = StoredRecord(b"first", b"scope", b"parent", "probe", b"sort", None,
                       0, None, 0, None, {"value": "original"})
_ALIAS = StoredAlias(b"first", _RECORD.key)
_FACT = StoredFact(b"first", 1, _RECORD.key, "probe", None, None, {"value": "original"})
_OPERATION = StoredOperation(b"first", b"first", 1, "pending", False, {"value": "original"})


async def _seed(transaction: StateTransaction) -> None:
    await transaction.insert_record(_RECORD)
    await transaction.insert_alias(_ALIAS)
    await transaction.insert_fact(_FACT)
    await transaction.insert_operation(_OPERATION)
    await transaction.next_sequence(b"first")


@pytest.mark.parametrize("kind", ("record", "immutable-record", "alias", "fact", "operation-key", "operation-stream"))
async def test_colliding_state_identity_cannot_read_or_replace_the_existing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    monkeypatch.setattr(_sql, "_key_locator",
                        lambda value: None if value is None else "b" * 64 if value == b"independent" else "a" * 64)
    async with _open_store("sql", tmp_path) as store:
        await store.mutate(_seed)
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await store.read_record(b"other")
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await store.read(lambda tx: tx.resolve_alias(b"other"))
        assert await store.read(lambda tx: tx.list_facts(FactQuery(b"other"))) == ()
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await store.read(lambda tx: tx.get_operation(b"other"))
        assert await store.read(lambda tx: tx.list_operations(OperationQuery(stream_key=b"other"))) == ()

        async def collide(transaction: StateTransaction) -> None:
            if kind == "record":
                await transaction.insert_record(replace(_RECORD, key=b"other"))
            elif kind == "immutable-record":
                await transaction.insert_immutable_records((replace(_RECORD, key=b"other"),))
            elif kind in {"alias", "fact"}:
                assert await transaction.guard_record(_RECORD.key, expected_storage_revision=0) is not None
                if kind == "alias":
                    await transaction.insert_alias(StoredAlias(b"other", _RECORD.key))
                else:
                    await transaction.insert_fact(replace(_FACT, stream_key=b"other"))
            elif kind == "operation-key":
                await transaction.insert_operation(replace(_OPERATION, key=b"other", sequence=2))
            else:
                await transaction.insert_operation(replace(_OPERATION, key=b"independent", stream_key=b"other"))

        with pytest.raises(AIError) as collision:
            await store.mutate(collide)
        assert collision.value.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_INTEGRITY_ERROR}
        assert await store.read_record(_RECORD.key) == _RECORD
        assert await store.read(lambda tx: tx.scan_aliases()) == (_ALIAS,)
        assert await store.read(lambda tx: tx.scan_facts()) == (_FACT,)
        assert await store.read(lambda tx: tx.scan_operations()) == (_OPERATION,)
        assert await store.read(lambda tx: tx.get_sequence(b"first")) == 1
        await store.validate_integrity()


@pytest.mark.parametrize("batch", (False, True))
async def test_colliding_sequence_reservation_rolls_back_without_incrementing_another_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: bool,
) -> None:
    monkeypatch.setattr(_sql, "_key_locator", lambda value: None if value is None else "a" * 64)
    async with _open_store("sql", tmp_path) as store:
        await store.mutate(lambda tx: tx.next_sequence(b"first"))
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await store.read(lambda tx: tx.get_sequence(b"other"))
        requests = {b"first": 3, b"other": 5} if batch else {b"other": 5}
        with pytest.raises(AIError) as collision:
            await store.mutate(lambda tx: tx.reserve_sequences(requests))
        assert collision.value.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_INTEGRITY_ERROR}
        assert await store.read(lambda tx: tx.scan_sequences()) == {b"first": 1}
        with pytest.raises(AIError) as collision:
            await store.mutate(lambda tx: tx.advance_sequence(b"other", 1))
        assert collision.value.code is ErrorCode.STORAGE_CONFLICT
        await store.mutate(lambda tx: tx.delete_sequence(b"other"))
        assert await store.read(lambda tx: tx.get_sequence(b"first")) == 1


@pytest.mark.parametrize("table,column", (
    ("ai_state_records", "key_locator"),
    ("ai_state_records", "scope_locator"),
    ("ai_state_records", "parent_locator"),
    ("ai_state_records", "sort_prefix"),
    ("ai_tool_operation_aliases", "alias_locator"),
    ("ai_tool_operation_aliases", "record_locator"),
    ("ai_state_facts", "stream_locator"),
    ("ai_state_facts", "owner_locator"),
    ("ai_state_operations", "operation_locator"),
    ("ai_state_operations", "stream_locator"),
    ("ai_state_counters", "stream_locator"),
))
async def test_integrity_scan_rejects_tampered_state_lookup_projections(
    tmp_path: Path, table: str, column: str,
) -> None:
    async with _open_store("sql", tmp_path) as store:
        await store.mutate(_seed)
        await store.validate_integrity()
        async with store.context.engine.begin() as connection:
            await connection.execute(
                text(f"UPDATE {table} SET {column} = :corrupt WHERE store_key = :store"),
                {"corrupt": b"wrong" if column == "sort_prefix" else "0" * 64,
                 "store": store.store_key},
            )
        with pytest.raises(AIError) as corrupt:
            await store.validate_integrity()
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
