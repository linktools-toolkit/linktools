#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual native mutation paths compile the audit clock for each SQL backend."""

from dataclasses import replace
from pathlib import Path
import re

import pytest
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import delete
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.persistence import SqlStepStore
from tests.ai.test_native_started_source import managed_commit, records
from tests.ai.test_native_step_backup import _Reader


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("promote", "restore", "prune"))
async def test_native_update_paths_explicitly_stamp_mysql_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit-clock.db'}")
    store = SqlStepStore(engine, namespace="clock", tenant_id="test", max_snapshots_per_run=1)
    cold = None
    run, started = records()
    updates = []
    execute = AsyncSession.execute

    async def observe(self, statement, *args, **kwargs):
        if getattr(statement, "is_update", False):
            updates.append(statement)
        return await execute(self, statement, *args, **kwargs)

    try:
        await store.initialize(local=True)
        if operation == "promote":
            await store.import_run(run)
            with monkeypatch.context() as patch:
                patch.setattr(AsyncSession, "execute", observe)
                await managed_commit(store, run=run, events=(started,))
            assert store.confirmed_started(run.run_id) == started
        elif operation == "restore":
            await managed_commit(store, run=run, events=(started,))
            async with store.storage_context.sessions() as session, session.begin():
                backup = await store.capture_backup(_Reader(session))
            async with engine.begin() as connection:
                await connection.execute(delete(store._table("events")))
                await connection.execute(delete(store._table("runs")))
            cold = SqlStepStore(engine, namespace="clock", tenant_id="test")
            await cold.import_run(run)
            with monkeypatch.context() as patch:
                patch.setattr(AsyncSession, "execute", observe)
                async with cold.storage_context.sessions() as session, session.begin():
                    await cold.restore_backup(session, backup)
            assert await cold.list_events(run_id=run.run_id) == [started]
        else:
            first = ContinuableSnapshot(run_id=run.run_id, step_index=0, messages=[],
                                        state="interrupted", idempotency_key="first")
            final = replace(first, step_index=1, idempotency_key="final")
            await store.save_snapshot(first)
            with monkeypatch.context() as patch:
                patch.setattr(AsyncSession, "execute", observe)
                await store.save_snapshot(final)
            assert await store.list_snapshots(run_id=run.run_id, include_interrupted=True) == [final]
        expected = "ai_step_snapshots" if operation == "prune" else "ai_step_runs"
        statements = [statement for statement in updates if statement.table.name == expected]
        assert statements, f"did not exercise the {operation} UPDATE"
        for dialect, clock in ((mysql.dialect(), "UTC_TIMESTAMP(6)"),
                               (sqlite.dialect(), "CURRENT_TIMESTAMP"),
                               (postgresql.dialect(), "CURRENT_TIMESTAMP")):
            for statement in statements:
                sql = str(statement.compile(dialect=dialect))
                assignments = sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
                assert re.search(r"\bupdated_at\s*=\s*" + re.escape(clock), assignments)
    finally:
        if cold is not None:
            await cold.close()
        await store.close()
        await engine.dispose()
