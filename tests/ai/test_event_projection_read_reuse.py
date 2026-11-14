#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event-only publication keeps its actual seal-head check in the write scope."""

from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.persistence.state import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@pytest.mark.asyncio
async def test_event_only_seal_head_reuses_projection_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="event-projection", tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        await store.register_agent_run(AgentRunRecord("run"))
        await store.flush_execution_projection("run", execution_id="execution")
        archive = store.read_store(PersistenceDomain.TRACE)
        original = archive.execution_history_head_record
        checking = False
        checked = []
        reads = []
        async def check(*args, **kwargs):
            nonlocal checking
            scope = active_state_scope()
            checking = scope is not None and scope.writable
            try:
                result = await original(*args, **kwargs)
                if checking:
                    checked.append(result)
                return result
            finally:
                checking = False
        def before(connection, cursor, statement, parameters, context, executemany):
            if checking and statement.lstrip().upper().startswith(("SELECT", "WITH")):
                reads.append(statement)
        monkeypatch.setattr(archive, "execution_history_head_record", check)
        engine = archive.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", before)
        try:
            await store.append_event(StepEvent("run", "TOOL_CALL_SUCCEEDED", 1))
            await store.flush_execution_projection("run", execution_id="execution")
        finally:
            event.remove(engine, "before_cursor_execute", before)
        assert checked and checked[-1].event_count == 1
        assert reads == []
        assert len(await archive.list_events(agent_run_id="run")) == 1
    finally:
        await state.close()
