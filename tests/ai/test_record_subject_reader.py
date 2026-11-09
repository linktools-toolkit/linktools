#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact record subjects preserve isolation, ordering, and immutable identity."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import RecordReplacement, RecordSubjectReader
from .test_state_batch_delete_backends import _digest, _open_store, _record

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_record_subject_reads_preserve_scope_kind_and_transaction_view(
    tmp_path: Path, backend: str,
) -> None:
    subject, scope = _digest("subject"), _digest("scope")
    records = tuple(replace(_record(index), subject_digest=subject, sort_key="equal")
                    for index in (1, 2))
    expected = tuple(sorted(records, key=lambda record: record.key_digest))
    excluded = (replace(_record(3), subject_digest=subject, scope_digest=_digest("other")),
                replace(_record(4), subject_digest=subject, kind="other"), _record(5))
    async with _open_store(backend, tmp_path / backend) as store:
        other = await store.storage_group.open_partition(_digest("other-store"), create=True)
        assert other is not None

        async def seed(tx):
            assert isinstance(tx, RecordSubjectReader)
            await tx.insert_records((*records, *excluded))
            assert await tx.read_record_subjects(
                scope_digest=scope, kind="probe", subject_digests=(subject, subject),
            ) == expected
        await store.mutate(seed)
        await other.mutate(lambda tx: tx.insert_record(replace(_record(6), subject_digest=subject)))
        assert await store.read(lambda tx: tx.read_record_subjects(
            scope_digest=scope, kind="probe", subject_digests=(subject, _digest("absent")),
        )) == expected
        assert await store.read(lambda tx: tx.read_record_subjects(
            scope_digest=scope, kind="probe", subject_digests=(),
        )) == ()


@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("initial,changed", ((None, "new"), ("old", None), ("old", "new")))
async def test_record_subject_identity_cannot_change(
    tmp_path: Path, backend: str, initial: str | None, changed: str | None,
) -> None:
    record = replace(_record(1), subject_digest=None if initial is None else _digest(initial))
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        candidate = replace(record, storage_revision=1,
                            subject_digest=None if changed is None else _digest(changed))
        with pytest.raises((ValueError, AIError)) as raised:
            await store.mutate(lambda tx: tx.replace_record(candidate, expected_storage_revision=0))
        if backend == "sql":
            assert raised.value.code is ErrorCode.INTERNAL_ERROR
            assert isinstance(raised.value.__cause__, ValueError)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == record
        valid = replace(record, storage_revision=1, data={"new": True})
        assert await store.mutate(lambda tx: tx.replace_record(valid, expected_storage_revision=0))
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == valid


async def test_sql_subject_reads_bound_parameters_and_globally_order_chunks(tmp_path: Path) -> None:
    subjects = tuple(_digest(f"subject-{index}") for index in range(1100))
    first = replace(_record(1), subject_digest=subjects[0], sort_key="z")
    last = replace(_record(2), subject_digest=subjects[-1], sort_key="a")
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_records((first, last)))
        counts = []

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "ai_state_records" in statement:
                counts.append(len(parameters))
                assert len(parameters) <= 999
                assert "subject_digest IN" in statement
                assert " OR " not in statement

        engine = store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await store.read(lambda tx: tx.read_record_subjects(
                scope_digest=first.scope_digest, kind=first.kind, subject_digests=(),
            )) == ()
            assert counts == []
            assert await store.read(lambda tx: tx.read_record_subjects(
                scope_digest=first.scope_digest, kind=first.kind, subject_digests=(*subjects, *subjects),
            )) == (last, first)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert len(counts) == 2


async def test_sql_subject_cas_batches_stay_within_parameter_budget(tmp_path: Path) -> None:
    records = tuple(replace(_record(index), subject_digest=_digest(f"subject-{index}"),
                            parent_digest=_digest("parent")) for index in range(65))
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_records(records))
        counts = []

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("UPDATE") and "ai_state_records" in statement:
                counts.append(len(parameters))
                assert len(parameters) <= 999
                assert "subject_digest" not in statement.split(" SET ", 1)[1].split(" WHERE ", 1)[0]

        engine = store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            await store.mutate(lambda tx: tx.replace_records(tuple(
                RecordReplacement(replace(record, storage_revision=1), 0) for record in records
            )))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        print(f"subject CAS bind counts: {counts!r}")
        assert counts


async def test_sql_subject_lookup_query_plan(tmp_path: Path) -> None:
    records = tuple(replace(_record(index), subject_digest=_digest(f"subject-{index}"))
                    for index in range(1100))
    async with _open_store("sql", tmp_path / "sql") as store:
        async def seed(tx):
            for offset in range(0, len(records), 40):
                await tx.insert_records(records[offset:offset + 40])
        await store.mutate(seed)
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "subject_digest IN" in statement:
                statements.append((statement, parameters))

        engine = store.storage_group.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            selected = (records[0], records[1099])
            assert await store.read(lambda tx: tx.read_record_subjects(
                scope_digest=records[0].scope_digest, kind="probe",
                subject_digests=tuple(record.subject_digest for record in selected),
            )) == selected
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        statement, parameters = statements[0]
        async with engine.connect() as connection:
            fresh = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)).all()
            await connection.exec_driver_sql("ANALYZE ai_state_records")
            analyzed = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)).all()
        for plan in (fresh, analyzed):
            details = " ".join(row[3] for row in plan)
            assert "scope_digest_subject_digest" in details
            assert "store_digest_kind_sort_key" not in details
        print(f"subject lookup fresh EXPLAIN: {fresh!r}")
        print(f"subject lookup analyzed EXPLAIN: {analyzed!r}")
        assert fresh and analyzed
