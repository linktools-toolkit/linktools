#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution lifecycle and exact request receipts share a physical owner."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, select

from linktools.ai.persistence._record_owner import record_owner_table
from linktools.ai.persistence.state import RecordQuery, RecordReplacement, record_key

from .test_state_group_record_batch import _database, _record


def _address(domain: str, kind: str, *identity: str) -> bytes:
    return record_key("owner-split", "tenant", domain, kind, identity)


@pytest.mark.asyncio
async def test_execution_owner_batches_exclude_custom_and_other_domain_records(tmp_path: Path) -> None:
    execution = replace(_record("execution", key=_address("execution", "execution", "run")),
                        kind="execution", state="STARTED", subject_id="execution")
    receipt = replace(_record("execution-receipt", key=_address("execution", "idempotency", "scope", "request")),
                      kind="idempotency", scope_key=b"scope", subject_id="request")
    evaluation = replace(receipt, key=_address("evaluation", "idempotency", "scope", "request"),
                         sort_key=b"evaluation")
    opaque = replace(receipt, key=b"opaque", sort_key=b"opaque")
    assert record_owner_table(execution.key) == record_owner_table(receipt.key) == "ai_execution_records"
    assert record_owner_table(evaluation.key) == record_owner_table(opaque.key) == "ai_state_records"
    async with _database(tmp_path / "owner.db") as (engine, store, _):
        await store.mutate(lambda tx: tx.insert_records((evaluation, opaque)))
        writes = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many) -> None:
            if statement.startswith(("INSERT", "UPDATE")):
                writes.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await store.mutate(lambda tx: tx.insert_records((execution, receipt)))
            updated = (replace(execution, state="DONE", storage_revision=1),
                       replace(receipt, state="SUCCEEDED", storage_revision=1))
            await store.mutate(lambda tx: tx.replace_records(tuple(
                RecordReplacement(record, 0) for record in updated)))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)

        assert len(writes) == 2
        assert all("ai_execution_records" in statement for statement in writes)
        keys = (execution.key, receipt.key, evaluation.key, opaque.key)
        assert await store.read(lambda tx: tx.get_records(keys)) == {
            record.key: record for record in (*updated, evaluation, opaque)
        }
        query = RecordQuery(kind="idempotency", scope_key=b"scope")
        expected = (evaluation, updated[1], opaque)
        assert await store.read(lambda tx: tx.list_records(query)) == expected
        assert await store.read(lambda tx: tx.read_record_subjects(
            scope_key=b"scope", kind="idempotency", subject_ids=("request",))) == expected

        tables = store.storage_group.metadata.tables
        assert not any(name.endswith("_sha256") for name in tables["ai_state_records"].c.keys())
        async with engine.connect() as connection:
            for owner, records in (("ai_state_records", (evaluation, opaque)),
                                   ("ai_execution_records", updated)):
                table = tables[owner]
                physical = (await connection.execute(select(table.c.key).where(table.c.store_key == store.store_key))).scalars()
                assert set(physical) == {record.key for record in records}
