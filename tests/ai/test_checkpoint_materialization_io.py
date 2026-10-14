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
        "postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", (False, True))
async def test_recovery_preparation_bounds_metadata_read_transactions(tmp_path: Path, replay: bool) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "preparation.db")
    await state.initialize(namespace="checkpoint-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
    engine = archive.state_store.storage_group.context.engine.sync_engine
    begins = []

    def begin(connection):
        begins.append(connection)

    try:
        if replay:
            await archive.materialize_checkpoint(run, checkpoint)
        event.listen(engine, "begin", begin)
        try:
            prepared, interactions, sequence = await state.run_store._prepare_recovery_checkpoint(
                archive, run, checkpoint, (),
            )
        finally:
            event.remove(engine, "begin", begin)
        assert prepared.stored.step_index == checkpoint.step_index
        assert interactions == () and sequence == 1
        assert len(begins) <= 3
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_checkpoint_relocation_scopes_metadata_without_holding_body_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.runtime.state._store import active_state_scope
    from .test_incremental_projection_boundaries import _storage

    state = _storage(backend, tmp_path)
    await state.initialize(namespace="checkpoint-relocation", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint(
        "run", 1, [ModelRequest(parts=[UserPromptPart("retained message")])],
        transcript_message_count_before=0,
    )
    try:
        await archive.materialize_checkpoint(run, checkpoint)
        get, count = archive.get_agent_run, archive.transcript_message_count_for_run
        load = archive.transcript_repository.load_message_span
        scopes, bodies = [], []

        async def tracked_get(*args, **kwargs):
            scopes.append(active_state_scope())
            return await get(*args, **kwargs)

        async def tracked_count(*args, **kwargs):
            scopes.append(active_state_scope())
            return await count(*args, **kwargs)

        async def tracked_load(*args, **kwargs):
            assert active_state_scope() is None
            bodies.append(True)
            return await load(*args, **kwargs)

        monkeypatch.setattr(archive, "get_agent_run", tracked_get)
        monkeypatch.setattr(archive, "transcript_message_count_for_run", tracked_count)
        monkeypatch.setattr(archive.transcript_repository, "load_message_span", tracked_load)
        relocated = await archive.relocate_run_checkpoint(run, checkpoint)
        assert relocated.transcript_message_count_before == 1
        assert scopes and all(scope is not None and scope is scopes[0] for scope in scopes)
        assert bodies == [True]
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_recovery_metadata_cancellation_releases_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from linktools.ai.runtime.state._store import active_state_scope
    from .test_incremental_projection_boundaries import _storage

    state = _storage(backend, tmp_path)
    await state.initialize(namespace="checkpoint-cancel", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
    count = archive.model_interaction_count

    async def cancelled_count(*args, **kwargs):
        assert active_state_scope() is not None
        raise asyncio.CancelledError

    try:
        monkeypatch.setattr(archive, "model_interaction_count", cancelled_count)
        with pytest.raises(asyncio.CancelledError):
            await state.run_store._prepare_recovery_checkpoint(archive, run, checkpoint, ())
        assert active_state_scope() is None
        monkeypatch.setattr(archive, "model_interaction_count", count)
        _, interactions, sequence = await state.run_store._prepare_recovery_checkpoint(archive, run, checkpoint, ())
        assert interactions == () and sequence == 1
    finally:
        await state.close()
