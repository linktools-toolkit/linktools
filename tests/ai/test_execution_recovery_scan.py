#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution scan buckets preserve logical filtering and raw-key pagination."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence._record_owner import EXECUTION_SCAN_STATES, record_scan_kind, record_scan_kinds
from linktools.ai.persistence._record_query import ExecutionRecoveryCandidateQuery
from linktools.ai.persistence.state import RecordQuery

from .test_state_batch_delete_backends import _digest, _open_store, _record


def test_execution_scan_bucket_selection_keeps_exact_persisted_states() -> None:
    assert EXECUTION_SCAN_STATES == frozenset({
        "PENDING_START", "STARTED", "RECOVERY_REQUIRED", "CANCELLING",
        "FINALIZING", "WAITING_DEFERRED",
    })
    assert record_scan_kinds("execution", EXECUTION_SCAN_STATES) == ("e:1",)
    assert record_scan_kinds("execution", frozenset({"DONE"})) == ("e:1", "e:0")
    assert record_scan_kinds("execution", frozenset({"STARTED "})) == ("e:1", "e:0")
    assert record_scan_kinds("execution", frozenset({"unknown"})) == ("e:1", "e:0")
    assert record_scan_kinds("execution ", frozenset({"STARTED"})) == ("e:1",)
    assert record_scan_kinds("execution", frozenset({"STARTED", "DONE"})) == ("e:1", "e:0")
    assert record_scan_kinds("execution", None) == ("e:1", "e:0")
    assert record_scan_kinds("execution", frozenset()) == ()
    assert record_scan_kinds("e:1", None) == ("k:e:1",)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("kind", ("execution", "execution "))
@pytest.mark.parametrize("states", (
    None, frozenset(), EXECUTION_SCAN_STATES, frozenset({"STARTED"}),
    frozenset({"DONE"}), frozenset({"STARTED", "DONE"}), frozenset({"unknown"}),
    frozenset({"STARTED "}),
))
async def test_execution_scan_preserves_states_filters_and_bounded_merge(
    tmp_path: Path, backend: str, kind: str, states: frozenset[str] | None,
) -> None:
    parent = _digest("parent")
    values = (None, "DONE", *sorted(EXECUTION_SCAN_STATES), "unknown", "DONE", "STARTED", "STARTED ")
    records = tuple(
        replace(_record(index), kind=kind, state=state, parent_key=parent)
        for index, state in enumerate(values)
    )
    excluded = (
        replace(_record(40), kind="other", state="STARTED", parent_key=parent),
        replace(_record(41), kind="execution", state="STARTED", parent_key=_digest("other-parent")),
        replace(_record(42), kind="execution", state="STARTED", parent_key=parent,
                scope_key=_digest("other-scope")),
    )
    expected = tuple(record for record in records if states is None or record.state in states)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_records((*records, *excluded)))
        other = await store.storage_group.open_partition(_digest("other-store"), create=True)
        assert other is not None
        await other.mutate(lambda tx: tx.insert_record(records[0]))
        query = RecordQuery(kind=kind, states=states, scope_key=records[0].scope_key,
                            parent_key=parent)
        assert await store.read(lambda tx: tx.list_records(query)) == expected
        assert await store.read(lambda tx: tx.list_records(replace(query, limit=3))) == expected[:3]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("states", (None, EXECUTION_SCAN_STATES, frozenset({"STARTED", "DONE"})))
async def test_execution_scan_pages_use_full_sort_key_and_raw_identity(
    tmp_path: Path, backend: str, states: frozenset[str] | None,
) -> None:
    prefix = b"x" * 128
    records = tuple(
        replace(_record(index), key=key, kind="execution", state=state, sort_key=prefix + suffix)
        for index, (key, state, suffix) in enumerate((
            (b"z", "STARTED", b"a"), (b"a", "DONE", b"a"),
            (b"b", "WAITING_DEFERRED", b"a"), (b"c", "STARTED", b"b"),
            (b"d", "DONE", b"b"), (b"e", None, b"c"),
        ))
    )
    expected = tuple(sorted(
        (record for record in records if states is None or record.state in states),
        key=lambda record: (record.sort_key, record.key),
    ))
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_records(records))
        query = RecordQuery(kind="execution", states=states, sort_key_prefix=prefix, limit=2)
        seen = []
        for _ in range(len(expected) + 1):
            page = await store.read(lambda tx: tx.list_records(query))
            if not page:
                break
            seen.extend(page)
            query = replace(query, after_sort_key=page[-1].sort_key, after_key=page[-1].key)
        else:
            pytest.fail("raw cursor did not advance")
        assert tuple(seen) == expected
        # A sort-only cursor excludes every equal full-sort row, not just one bucket.
        query = RecordQuery(kind="execution", states=states, after_sort_key=prefix + b"a")
        assert await store.read(lambda tx: tx.list_records(query)) == tuple(
            record for record in expected if record.sort_key > prefix + b"a"
        )
        # A cursor need not identify an existing row.
        query = RecordQuery(kind="execution", states=states,
                            after_sort_key=prefix + b"a", after_key=b"m")
        assert await store.read(lambda tx: tx.list_records(query)) == tuple(
            record for record in expected
            if (record.sort_key, record.key) > (prefix + b"a", b"m")
        )


@pytest.mark.asyncio
async def test_execution_recovery_sqlite_plan_uses_store_and_scan_bucket(tmp_path: Path) -> None:
    from linktools.ai.persistence.state import record_key

    records = tuple(
        replace(_record(index), kind="execution",
                key=(record_key("test", "tenant", "execution", "execution", (str(index),))
                     if index % 128 == 64 else _record(index).key),
                state="STARTED" if index % 64 == 0 else "DONE")
        for index in range(256)
    )
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_records(records))
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "ai_state_records" in statement:
                statements.append((statement, parameters))

        engine = store.storage_group.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            query = ExecutionRecoveryCandidateQuery(kind="execution", limit=2)
            assert await store.read(lambda tx: tx.list_records(query)) == (records[0], records[64])
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(statements) == 1
        statement, parameters = statements[0]
        assert "scan_kind" in statement
        assert statement.count("UNION ALL") == 1
        async with engine.connect() as connection:
            fresh = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)).all()
            await connection.exec_driver_sql("ANALYZE ai_state_records")
            await connection.exec_driver_sql("ANALYZE ai_execution_records")
            analyzed = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)).all()
        for plan in (fresh, analyzed):
            details = " ".join(row[3] for row in plan)
            assert "store_key_scan_kind_sort_prefix" in details
            assert "store_key=? AND scan_kind=?" in details
            assert "SCAN ai_state_records" not in details
            assert "SCAN ai_execution_records" not in details
            assert "SEARCH ai_execution_records" in details
            # Full-sort and raw-key ties may require a right-part sort; the
            # candidate range must still be bounded by store and recovery bucket.



def test_mysql_bounded_recovery_merge_keeps_order_limits_and_raw_key() -> None:
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence._plan import PersistenceDomain
    from linktools.ai.persistence._schema import build_runtime_sql_metadata
    from linktools.ai.persistence._sql import _record_query_statement

    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    query = ExecutionRecoveryCandidateQuery(
        kind="execution", after_sort_key=b"sort", after_key=b"key", limit=4,
    )
    mysql57 = mysql.dialect(paramstyle="format")
    mysql57.server_version_info = (5, 7, 44)

    def sql(dialect, selected=query) -> str:
        statement = _record_query_statement(metadata, b"store", selected, dialect=dialect.name)
        return " ".join(str(statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).split())

    direct = sql(mysql57)
    assert direct.startswith("(SELECT")
    assert "LIMIT 4) UNION ALL (SELECT" in direct
    assert direct.endswith("LIMIT 4) ORDER BY sort_prefix, sort_key, `key` LIMIT 4")
    assert "FROM ai_state_records" in direct and "FROM ai_execution_records" in direct
    for predicate in ("scan_kind = 'e:1'", "store_key = 'store'", "sort_prefix >= 'sort'", ".sort_key > 'sort'", ".`key` > 'key'"):
        assert direct.count(predicate) == 2
    for dialect in (postgresql.dialect(), sqlite.dialect()):
        assert sql(dialect).startswith("SELECT")
    assert sql(mysql57, replace(query, limit=None)).startswith("SELECT")
    assert sql(mysql57, RecordQuery(kind="execution", limit=4)).startswith("SELECT")


@pytest.mark.parametrize("options", ({"kind": "other"}, {"kind": "execution", "states": frozenset({"STARTED"})}))
def test_internal_recovery_query_does_not_overload_status_filters(options):
    with pytest.raises(ValueError):
        ExecutionRecoveryCandidateQuery(**options)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_recovery_candidates_do_not_decode_unrelated_cold_payloads(tmp_path, backend):
    # StateStore permits opaque custom records. Recovery candidate selection
    # must not interpret every cold execution-kind record as a valid envelope.
    cold = replace(_record(0), kind="execution", state="SUCCEEDED", data={"opaque": "cold"})
    marker = {"value": {"payload": {"fields": {"recovery": {"fields": {
        "handoff_phase": {"value": "prepared"}}}}}}}
    claimed = replace(_record(1), kind="execution", state="FAILED", data=marker)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_records((cold, claimed)))
        query = ExecutionRecoveryCandidateQuery(kind="execution")
        assert await store.read(lambda tx: tx.list_records(query)) == (claimed,)
    # Selecting the marker is not authority to consume a malformed record.
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.codec import execution_prepared_handoff
    with pytest.raises(AIError) as caught:
        execution_prepared_handoff(marker)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR



def test_scan_projection_is_not_domain_state_or_a_typed_owner_column():
    from linktools.ai.persistence._record_owner import record_owner_values
    from linktools.ai.persistence._sql import _record_values

    record = replace(_record(0), kind="execution", state="FAILED")
    values = _record_values(record)
    assert values["state"] == "FAILED" and values["scan_kind"] == "e:0"
    assert record_scan_kind("execution", "FAILED", prepared=True) == "e:1"
    assert record_scan_kind("execution ", "STARTED ") == "e:1"
    for owner in ("ai_step_tool_operations", "ai_step_run_admissions"):
        assert "scan_kind" not in record_owner_values(owner, {"scan_kind": "k:probe"})


@pytest.mark.asyncio
async def test_sql_scan_projection_follows_mutations_and_survives_lease_only_writes(tmp_path):
    from sqlalchemy import select
    from linktools.ai.persistence.state import RecordReplacement

    marker = {"value": {"payload": {"fields": {"recovery": {"fields": {
        "handoff_phase": {"value": "prepared"}}}}}}}
    hot = replace(_record(0), kind="execution", state="STARTED")
    cold = replace(_record(1), kind="execution", state="FAILED")
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_records((hot, cold)))
        table = store.storage_group.metadata.tables["ai_state_records"]

        async def buckets():
            async with store.storage_group.context.engine.connect() as connection:
                return dict((await connection.execute(select(table.c.key, table.c.scan_kind).where(
                    table.c.store_key == store.store_key))).all())

        assert await buckets() == {hot.key: "e:1", cold.key: "e:0"}
        stopped = replace(hot, state="SUCCEEDED", storage_revision=1)
        pending = replace(cold, data=marker, storage_revision=1)
        await store.mutate(lambda tx: tx.replace_records((
            RecordReplacement(stopped, 0), RecordReplacement(pending, 0))))
        assert await buckets() == {hot.key: "e:0", cold.key: "e:1"}
        assert await store.mutate(lambda tx: tx.update_record_lease(
            pending.key, expected_storage_revision=1, lease_owner=None, lease_fence=1,
            lease_expires_at=None))
        assert await buckets() == {hot.key: "e:0", cold.key: "e:1"}
        after_lease = await store.read_record(pending.key)
        completed = replace(after_lease, data=cold.data, storage_revision=3)
        assert await store.mutate(lambda tx: tx.replace_record(completed, expected_storage_revision=2))
        assert await buckets() == {hot.key: "e:0", cold.key: "e:0"}
        immutable = replace(_record(2), kind="archive_probe", state=None)
        await store.mutate(lambda tx: tx.insert_immutable_records((immutable,)))
        assert (await buckets())[immutable.key] == "k:archive_probe"
