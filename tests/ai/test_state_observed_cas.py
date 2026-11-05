#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical revision receipts permit read-free conditional mutations."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import RecordReplacement, StateTransaction, StateTransactionNestingError
from linktools.ai.storage import SQLiteDialect
from tests.ai.test_state_batch_delete_backends import _digest, _open_store, _record
from tests.ai.test_state_group_record_batch import _database

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_revision_guard_reuses_current_revision_and_source_payload(
    backend: str, tmp_path: Path,
) -> None:
    original = replace(_record(1), data={"revision": 71, "result": "original"})
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(original))

        async def mutate(tx: StateTransaction) -> None:
            assert await tx.guard_record_revision(original.key_digest, expected_storage_revision=0) == 1
            guarded = await tx.guard_record(original.key_digest, expected_storage_revision=0)
            assert guarded == replace(original, storage_revision=1)
            assert await tx.guard_record(original.key_digest, expected_storage_revision=0) == guarded
            assert await tx.guard_record(original.key_digest, expected_storage_revision=1) == guarded
            candidate = replace(guarded, storage_revision=2, state="updated")
            await tx.replace_records((RecordReplacement(candidate, 1),))
            assert await tx.get_record(original.key_digest) == candidate
            assert await tx.guard_record(original.key_digest, expected_storage_revision=0) == candidate
            assert await tx.guard_record_revision(original.key_digest, expected_storage_revision=0) == 2

        await store.mutate(mutate)
        actual = await store.read(lambda tx: tx.get_record(original.key_digest))
        assert actual == replace(original, storage_revision=2, state="updated")
        assert await store.mutate(lambda tx: tx.guard_record(
            original.key_digest, expected_storage_revision=0,
        )) is None
        with pytest.raises(StateTransactionNestingError, match="read-only"):
            await store.read(lambda tx: tx.guard_record(
                original.key_digest, expected_storage_revision=0,
            ))


@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("field,value", (("scope_digest", None), ("parent_digest", _digest("parent")),
                                        ("kind", "other"), ("kind", "PROBE"), ("kind", "probe "),
                                        ("sort_key", "other"), ("sort_key", "0001 ")))
async def test_conditional_mutation_preserves_every_immutable_identity(
    backend: str, field: str, value: object, tmp_path: Path,
) -> None:
    original = _record(1)
    candidate = replace(original, **{field: value})
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(original))
        async def reject(tx: StateTransaction) -> None:
            with pytest.raises(ValueError, match="physical identity"):
                await tx.replace_records((RecordReplacement(replace(candidate, storage_revision=1), 0),))
        await store.mutate(reject)
        assert await store.read(lambda tx: tx.get_record(original.key_digest)) == original


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_revision_guard_validates_token_and_does_not_survive_rollback(
    backend: str, tmp_path: Path,
) -> None:
    original = _record(1)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(original))
        assert await store.mutate(lambda tx: tx.guard_record_revision(_digest("missing"), expected_storage_revision=0)) is None
        assert await store.mutate(lambda tx: tx.guard_record_revision(original.key_digest, expected_storage_revision=1)) is None
        for revision in (-1, True):
            async def reject(tx: StateTransaction) -> None:
                with pytest.raises(ValueError, match="expected_storage_revision"):
                    await tx.guard_record_revision(original.key_digest, expected_storage_revision=revision)
            await store.mutate(reject)
        with pytest.raises(StateTransactionNestingError, match="read-only"):
            await store.read(lambda tx: tx.guard_record_revision(original.key_digest, expected_storage_revision=0))

        async def abort(tx: StateTransaction) -> None:
            assert await tx.guard_record_revision(original.key_digest, expected_storage_revision=0) == 1
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError, match="STORAGE_CONFLICT"):
            await store.mutate(abort)
        assert await store.read(lambda tx: tx.get_record(original.key_digest)) == original


@pytest.mark.parametrize("backend", ("memory", "sql"))
@pytest.mark.parametrize("changed", ({"data": {"result": "ghost"}}, {"state": "ghost"},
                                     {"lease_owner": "ghost"}, {"lease_fence": 91}))
async def test_revision_receipt_cannot_publish_caller_supplied_mutable_fields(
    backend: str, changed: dict[str, object], tmp_path: Path,
) -> None:
    original = _record(1)
    fabricated = replace(original, **changed)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(original))

        async def mutate(tx: StateTransaction) -> None:
            receipt = await tx.guard_record_revision(
                fabricated.key_digest, expected_storage_revision=fabricated.storage_revision,
            )
            assert receipt == 1
            assert isinstance(receipt, int)
            source = await tx.guard_record(original.key_digest, expected_storage_revision=0)
            assert source == replace(original, storage_revision=1)
            assert source != replace(fabricated, storage_revision=1)
        await store.mutate(mutate)
        assert await store.read(lambda tx: tx.get_record(original.key_digest)) == replace(original, storage_revision=1)


@pytest.mark.parametrize("executemany", (False, True))
async def test_known_record_writes_execute_no_select_and_compile_for_mysql(
    executemany: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SQLiteDialect, "prefer_update_executemany", property(lambda self: executemany))
    async with _database(tmp_path / "observed.db") as (engine, store, _):
        records = tuple(replace(_record(index), parent_digest=None if index % 2 else _digest("parent"))
                        for index in range(70))
        await store.mutate(lambda tx: tx.insert_records(records))
        statements: list[str] = []
        mysql_updates: list[str] = []

        def capture(conn, cursor, sql, parameters, context, many) -> None:
            statements.append(sql)
            if sql.startswith("UPDATE ai_state_records"):
                mysql_updates.append(str(context.compiled.statement.compile(dialect=mysql.dialect())))

        async def mutate(tx: StateTransaction) -> None:
            assert await tx.guard_record_revision(records[0].key_digest, expected_storage_revision=0) == 1
            assert await tx.guard_record_revision(records[0].key_digest, expected_storage_revision=0) == 1
            await tx.replace_records(tuple(RecordReplacement(
                replace(record, storage_revision=2 if index == 0 else 1, data={"revision": 99}),
                1 if index == 0 else 0,
            ) for index, record in enumerate(records)))
            actual = await tx.get_records(tuple(record.key_digest for record in records))
            assert len(actual) == len(records)
            assert all(record.data["revision"] == 99 for record in actual.values())

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await store.mutate(mutate)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert not any(sql.lstrip().upper().startswith("SELECT") for sql in statements), statements
        assert len(mysql_updates) >= 2
        assert all("scope_digest" in sql and "parent_digest" in sql and "sort_key" in sql
                   and "storage_revision" in sql and "RETURNING" not in sql for sql in mysql_updates[1:])


@pytest.mark.parametrize("bound", (False, True), ids=("literal", "executemany-bind"))
async def test_mysql_identity_conditions_compare_bytes_and_bind_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool,
) -> None:
    from sqlalchemy import bindparam
    from sqlalchemy.dialects.mysql.asyncmy import dialect

    compiler_dialect = dialect(dbapi=dialect.import_dbapi())
    values = {"scope_digest": None, "parent_digest": None, "kind": "probe 測🙂 ", "sort_key": "0001 測🙂 "}
    async with _database(tmp_path / "compile.db") as (_, store, _):
        async def inspect(tx: StateTransaction) -> None:
            with monkeypatch.context() as patch:
                patch.setattr(SQLiteDialect, "name", property(lambda self: "mysql"))
                conditions = tx._record_identity_conditions(tx._table("ai_state_records"), {
                    name: bindparam(name) if bound else value for name, value in values.items()
                })
                compiled = [condition.compile(dialect=compiler_dialect) for condition in conditions]
                sql = " AND ".join(str(condition) for condition in compiled)
                assert "CAST(ai_state_records.kind AS BINARY) <=>" in sql
                assert "CAST(ai_state_records.sort_key AS BINARY) <=>" in sql
                processed = []
                for condition in compiled:
                    parameters = condition.construct_params(values if bound else None)
                    for name, value in parameters.items():
                        processor = condition._bind_processors.get(name)
                        processed.append(processor(value) if processor is not None else value)
                assert [value for value in processed if value is not None] == [values["kind"], values["sort_key"]]
        await store.mutate(inspect)
