#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write admission prefetch retains its existing same-transaction counters."""
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._native_run_reference import NativeRunReferences
from linktools.ai.persistence._run_counters import run_counter_streams
from linktools.ai.persistence.state import StateTransaction, record_key

from .test_native_run_references import _source, _flush


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", (False, True))
async def test_run_write_prefetch_counters_and_reserved_values(tmp_path: Path, cached: bool) -> None:
    async with _source(tmp_path) as (state, _, _, _):
        await _flush(state)
        store = state.run_store.read_store(PersistenceDomain.TRACE).state_store
        reader = NativeRunReferences(store, namespace="native-run-facts", tenant_id="tenant")
        key = record_key("native-run-facts", "tenant", "trace", "agent_run", ("run",))
        missing = record_key("native-run-facts", "tenant", "trace", "agent_run", ("missing",))
        foreign = record_key("native-run-facts", "other", "trace", "agent_run", ("run",))
        streams = run_counter_streams(key)
        statements: list[str] = []

        def capture(_connection, _cursor, statement, *_args) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        event.listen(store.context.engine.sync_engine, "before_cursor_execute", capture)
        try:
            async def write(tx: StateTransaction) -> None:
                if cached:
                    await reader.read(tx, keys=(key, missing, foreign))
                statements.clear()
                assert set(await reader.read_physical(tx, (key, missing, foreign))) == {key}
                assert len(statements) == 1
                assert await tx.get_sequences(streams) == dict(zip(streams, (1, 0, 0), strict=True))
                assert len(statements) == 1
                assert await tx.get_sequences(run_counter_streams(missing)) == dict.fromkeys(run_counter_streams(missing), 0)
                assert len(statements) == 1
                assert run_counter_streams(foreign, store_key=store.store_key) == ()
                assert await tx.reserve_sequence(streams[2], 2) == 2
                statements.clear()
                assert set(await reader.read_physical(tx, (key,))) == {key}
                assert await tx.get_sequence(streams[2]) == 2
                assert not statements
            await store.mutate(write)
            assert await store.read(lambda tx: tx.get_sequence(streams[2])) == 2
        finally:
            event.remove(store.context.engine.sync_engine, "before_cursor_execute", capture)
