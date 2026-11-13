#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Payload retirement batches SQL while preserving owner revision fences."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
from linktools.ai.persistence._step_payload_owner import (
    payload_record_from_row, payload_store_key, payload_store_locator,
)
from linktools.ai.persistence.state import STATE_GROUP_ARCHIVE_KEY, StoredFact


def _payload(index: int, tenant: str = "tenant"):
    body = str(index).encode()
    store = payload_store_key("payload-delete", tenant)
    return payload_record_from_row({
        "store_key": store, "store_locator": payload_store_locator(store),
        "sha256": hashlib.sha256(body).hexdigest(), "message": body,
        "compression": "raw", "raw_size": len(body), "payload_json": None, "gc_revision": 0,
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("commit", "rollback", "conflict"))
async def test_payload_batch_delete_keeps_scope_fences_and_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'payloads.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine, store_key=STATE_GROUP_ARCHIVE_KEY)
    await store.initialize()
    records = tuple(_payload(index) for index in range(3))
    other = _payload(0, "other")
    try:
        facts = tuple(StoredFact(f"stream-{index}".encode(), 1, row.key, "probe", None, None, {})
                      for index, row in enumerate((*records, other)))

        async def seed(tx):
            await tx.insert_records((*records, other))
            await tx.insert_facts(facts)

        await store.mutate(seed)
        counts = Counter()

        def count(_connection, _cursor, statement, _parameters, _context, _many):
            counts[statement.split()[0]] += 1

        if outcome == "conflict":
            original = _SqlTransaction.get_records

            async def change_revision(transaction, keys):
                found = await original(transaction, keys)
                owner = transaction._payload_owner()
                await transaction._execute(update(owner.table).where(owner.condition(records[-1].key)).values(gc_revision=1))
                return found

            monkeypatch.setattr(_SqlTransaction, "get_records", change_revision)

        async def retire(tx):
            await tx.delete_records(tuple(row.key for row in records) + (records[0].key, _payload(99).key))
            if outcome == "rollback":
                raise RuntimeError("rollback")

        event.listen(engine.sync_engine, "before_cursor_execute", count)
        try:
            if outcome == "commit":
                await store.mutate(retire)
                assert counts["SELECT"] == 1
                assert counts["UPDATE"] == 1
                assert counts["DELETE"] == 2
            elif outcome == "rollback":
                with pytest.raises(AIError) as raised:
                    await store.mutate(retire)
                assert raised.value.code is ErrorCode.INTERNAL_ERROR
                assert isinstance(raised.value.__cause__, RuntimeError)
            else:
                with pytest.raises(AIError) as raised:
                    await store.mutate(retire)
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", count)
            monkeypatch.undo()
        found = await store.read(lambda tx: tx.get_records(tuple(row.key for row in (*records, other))))
        expected = (other,) if outcome == "commit" else (*records, other)
        assert found == {row.key: row for row in expected}
        remaining_facts = await store.read(lambda tx: tx.scan_facts())
        assert {fact.stream_key for fact in remaining_facts} == {fact.stream_key for fact in ((facts[-1],) if outcome == "commit" else facts)}
    finally:
        await store.close()
        await engine.dispose()
