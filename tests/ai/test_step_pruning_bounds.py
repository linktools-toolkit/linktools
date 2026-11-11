#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot retention returns bounded metadata and preserves recovery/replay."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import select, text
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.steps import encode_native_replay_key


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_index", (None, 1, 11))
async def test_reducing_snapshot_retention_reads_bounded_metadata_and_keeps_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete_index: int | None,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retention.db'}")
    unlimited = SqlStepStore(engine, namespace="steps", tenant_id="tenant")
    limited = SqlStepStore(engine, namespace="steps", tenant_id="tenant", max_snapshots_per_run=2)
    snapshots = [ContinuableSnapshot(
        run_id="run", step_index=index, idempotency_key=str(index), messages=[],
        state="complete" if index == complete_index else "interrupted",
    ) for index in range(12)]
    final = replace(snapshots[-1], step_index=12, idempotency_key="12", state="interrupted")
    result_sizes = []
    compiled_statements = []
    execute = AsyncSession.execute

    async def observe(self, statement, *args, **kwargs):
        if (getattr(statement, "is_select", False) or getattr(statement, "is_update", False)) and "ai_step_snapshots" in str(statement):
            dialect = mysql.dialect()
            dialect.server_version_info = (5, 7, 44)
            compiled_statements.append(str(statement.compile(dialect=dialect)))
        result = await execute(self, statement, *args, **kwargs)
        if getattr(statement, "is_select", False) and "ai_step_snapshots" in str(statement):
            captured = result.freeze()
            result_sizes.append(len(captured.data))
            return captured()
        return result

    try:
        await unlimited.initialize(local=True)
        for snapshot in snapshots:
            await unlimited.save_snapshot(snapshot)
        await limited.initialize()
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "execute", observe)
            await limited.save_snapshot(final)
            await limited.latest_snapshot(run_id="run")
        assert result_sizes and max(result_sizes) <= 128
        assert any("pruned = false" in statement for statement in compiled_statements)
        assert not any("pruned IS false" in statement for statement in compiled_statements)
        retained = ([snapshots[complete_index]] if complete_index is not None and complete_index < 11 else [])
        retained += [snapshots[-1], final]
        assert await limited.list_snapshots(run_id="run", include_interrupted=True) == retained
        assert await limited.latest_snapshot(run_id="run") == (
            None if complete_index is None else snapshots[complete_index]
        )
        # A pruned replay remains a tombstone rather than resurrecting history.
        await limited.save_snapshot(snapshots[0])
        assert await limited.list_snapshots(run_id="run", include_interrupted=True) == retained
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots")) == 13
            assert await connection.scalar(text(
                "SELECT count(*) FROM ai_step_snapshots "
                "WHERE pruned = 1 AND messages = '{}' AND sequence_root IS NULL"
            )) == 13 - len(retained)
    finally:
        await unlimited.close()
        await limited.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_complete", (False, True))
async def test_pruning_preserves_complete_point_appearing_after_retention_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_complete: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'late-complete.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="tenant")
    limited = SqlStepStore(engine, namespace="steps", tenant_id="tenant", max_snapshots_per_run=1)
    first = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="first", messages=[],
                                state="complete" if prior_complete else "interrupted")
    late = replace(first, step_index=1, idempotency_key="late", state="complete")
    newest = replace(first, step_index=2, idempotency_key="newest", state="interrupted")
    execute = AsyncSession.execute
    inserted = False
    try:
        await store.initialize(local=True)
        for snapshot in (first, late, newest):
            await store.save_snapshot(snapshot)
        table = limited._table("snapshots")
        async with engine.begin() as connection:
            delayed = dict((await connection.execute(select(table).where(
                table.c.idempotency_key == encode_native_replay_key(late.idempotency_key),
            ))).mappings().one())
            await connection.execute(table.delete().where(table.c.id == delayed["id"]))
        await limited.initialize()

        async def appear_after_read(self, statement, *args, **kwargs):
            nonlocal inserted
            result = await execute(self, statement, *args, **kwargs)
            if not inserted and getattr(statement, "is_select", False) and "ai_step_snapshots" in str(statement):
                captured = result.freeze()
                inserted = True
                # Simulate a current-read UPDATE seeing an earlier allocated ID
                # that was absent from the retention SELECT's consistent view.
                await execute(self, table.insert().values(**delayed))
                return captured()
            return result

        final = replace(newest, step_index=3, idempotency_key="final")
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "execute", appear_after_read)
            await limited.save_snapshot(final)
        assert inserted
        assert await limited.latest_snapshot(run_id="run") == late
        assert await limited.latest_snapshot(run_id="run", include_interrupted=True) == final
        # The next boundary observes the new complete point and retires the old one.
        final = replace(final, step_index=4, idempotency_key="final-2")
        await limited.save_snapshot(final)
        assert await limited.list_snapshots(run_id="run", include_interrupted=True) == [late, final]
    finally:
        await store.close()
        await limited.close()
        await engine.dispose()
