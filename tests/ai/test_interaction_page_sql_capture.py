#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded interaction pages capture their watermark without a second read."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, insert, select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence.state import state_store_key

from .test_model_interaction_direct_archive import _record
from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
async def test_sql_interaction_page_is_bounded_and_scope_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        values = tuple(_record(seq, direct=True) for seq in range(1, 5))
        await archive.sync_prepared_projection(run, events=(), checkpoints=(), interactions=values,
                                              execution_id="execution", producer_generation=1)
        other = replace(run, agent_run_id="other")
        await archive.sync_prepared_projection(other, events=(), checkpoints=(),
            interactions=tuple(replace(value, agent_run_id="other") for value in values),
            execution_id="execution", producer_generation=1)
        engine = archive.state_store.context.engine
        table = build_runtime_sql_metadata(frozenset({archive.runtime_domain})).tables["ai_state_records"]
        async with engine.begin() as connection:
            row = dict((await connection.execute(select(table).where(
                table.c.store_key == archive.state_store.store_key,
                table.c.key == archive._interaction_key("run", 2),
            ))).mappings().one())
            row.pop("id")
            row["store_key"] = state_store_key("foreign-namespace", "foreign-tenant", "trace")
            await connection.execute(insert(table), row)
        statements: list[str] = []
        def capture(_connection, _cursor, sql, _parameters, _context, _many):
            if sql.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(sql)
        original = _SqlTransaction.read_sql_rows
        async def inspect_plan(self, context, statement):
            for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
                compiled = str(statement.compile(dialect=dialect))
                assert "LEFT OUTER JOIN" in compiled and "LIMIT" in compiled
                assert not compiled.lstrip().upper().startswith("WITH")
            return await original(self, context, statement)
        monkeypatch.setattr(_SqlTransaction, "read_sql_rows", inspect_plan)
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            assert await archive.list_model_interactions(agent_run_id="run", after_model_request_seq=1, limit=2) == list(values[1:3])
            assert len(statements) == 1
            statements.clear()
            assert await archive.list_model_interactions(agent_run_id="run", after_model_request_seq=4, limit=2) == []
            assert len(statements) == 1
            statements.clear()
            assert await archive.list_model_interactions(agent_run_id="absent", limit=2) == []
            assert len(statements) == 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await archive.state_store.mutate(lambda tx: tx.delete_records((archive._interaction_key("run", 2),)))
        with pytest.raises(AIError) as raised:
            await archive.list_model_interactions(agent_run_id="run", after_model_request_seq=1, limit=2)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_sql_interaction_page_seeks_parent_sort_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        values = tuple(_record(seq, direct=True) for seq in range(1, 513))
        async def populate(transaction):
            await transaction.insert_records(tuple(archive._stored_interaction(value) for value in values))
            await transaction.reserve_sequence(archive._stream("run", "interaction"), len(values))
        await archive.state_store.mutate(populate)
        plans: list[str] = []
        original = _SqlTransaction.read_sql_rows
        async def inspect_plan(self, context, statement):
            compiled = statement.compile(dialect=context.engine.dialect,
                                         compile_kwargs={"render_postcompile": True})
            parameters = tuple(compiled.params[name] for name in compiled.positiontup)
            async with context.engine.connect() as connection:
                plan = await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + str(compiled), parameters)
                plans.extend(str(row[-1]) for row in plan)
            return await original(self, context, statement)
        monkeypatch.setattr(_SqlTransaction, "read_sql_rows", inspect_plan)
        assert await archive.list_model_interactions(agent_run_id="run", after_model_request_seq=499, limit=3) == list(values[499:502])
        assert any("SEARCH ai_state_records USING INDEX" in detail
                   and "parent_locator=?" in detail and "sort_prefix>?" in detail
                   and "sort_prefix<?" in detail for detail in plans), plans
        # A small final LEFT JOIN may sort its bounded output; the source page
        # must seek the parent/prefix index, never scan all interaction rows.
        assert not any(detail.startswith("SCAN ai_state_records") for detail in plans), plans
