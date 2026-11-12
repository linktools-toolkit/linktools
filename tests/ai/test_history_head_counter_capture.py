#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run headers and their counters share one snapshot, never a later one."""
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence import PersistenceDomain

from .test_native_run_references import _source, _flush


@pytest.mark.asyncio
async def test_history_head_captures_same_owner_counters_and_refreshes_next_transaction(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, _, _, _run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        statements = []

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        engine = archive.state_store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await archive.execution_history_heads(()) == {}
            assert not statements
            heads = await archive.execution_history_heads(("run", "missing"))
            assert set(heads) == {"run", "missing"}
            assert heads["missing"].event_count == 0 and heads["missing"].projection_digest == "empty"
            assert heads["run"].event_count == 1
            assert heads["run"].checkpoint_count == 0 and heads["run"].interaction_count == 0
            assert len(statements) == 1
            assert all(name in statements[0] for name in (
                "event_high_water", "checkpoint_high_water", "interaction_high_water"))
            await _flush(state, step=1)
            statements.clear()
            fresh = await archive.execution_history_head_record("run")
            assert fresh.event_count == 2
            assert len(statements) == 1
        finally:
            event.remove(engine, "before_cursor_execute", capture)
