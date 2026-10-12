#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint materialization keeps replay and optimistic ownership checks."""

import asyncio
import getpass
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._sql import _SqlTransaction
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.runtime.state._store import FactQuery, StoredFact


@pytest.mark.asyncio
async def test_checkpoint_materialization_reads_latest_fact_once_and_replays(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    await state.initialize(namespace="checkpoint-io", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
    statements: list[str] = []

    def capture_sql(
        _connection: object, _cursor: object, statement: str,
        _parameters: object, _context: object, _executemany: bool,
    ) -> None:
        if statement.startswith("SELECT") and "ORDER BY ai_state_facts.sequence DESC" in statement:
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    try:
        await archive.materialize_checkpoint(run, checkpoint)
        assert len(statements) == 1
        statements.clear()
        await archive.materialize_checkpoint(run, checkpoint)
        assert len(statements) == 1
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.messages == [] and latest.step_index == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("postgresql",), indirect=True)
async def test_checkpoint_owner_fences_commit_after_absent_fact_read(
    _server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, command = _server
    engine = create_async_engine(URL.create(
        "postgresql+asyncpg", username=getpass.getuser(), database="postgres",
        query={"host": command[command.index("-h") + 1]},
    ), isolation_level="READ COMMITTED")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    await state.initialize(namespace="checkpoint-race", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
    await archive.register_agent_run(run)
    observed = asyncio.Event()
    release = asyncio.Event()
    original = _SqlTransaction.list_facts
    first: asyncio.Task[None] | None = None

    async def pause_after_absent_fact(
        transaction: _SqlTransaction, query: FactQuery,
    ) -> tuple[StoredFact, ...]:
        result = await original(transaction, query)
        if asyncio.current_task() is first and query.latest and not observed.is_set():
            assert not result
            observed.set()
            await release.wait()
        return result

    monkeypatch.setattr(_SqlTransaction, "list_facts", pause_after_absent_fact)
    try:
        first = asyncio.create_task(archive.materialize_checkpoint(run, checkpoint))
        await asyncio.wait_for(observed.wait(), 5)
        await archive.materialize_checkpoint(run, checkpoint)
        release.set()
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(first, 5)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        await archive.materialize_checkpoint(run, checkpoint)
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.messages == [] and latest.step_index == 1
    finally:
        release.set()
        if first is not None and not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await state.close()
        await engine.dispose()
