#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded association reads retain index scopes and merge duplicate events."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.persistence._trace_event_store import SqlTraceEventStore
from linktools.ai.persistence.state import StateTransaction

from .test_trace_event_domain import _append, _archive, _event


@pytest.mark.asyncio
async def test_history_associations_merge_bounded_index_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _archive(tmp_path, "sqlite") as (archive, run_id):
        await _append(archive,
            _event(run_id, "MODEL_REQUEST_STARTED"),
            _event(run_id, "MODEL_REQUEST_SUCCEEDED", message="2"),
            _event(run_id, "TOOL_CALL_STARTED", tool="call"),
            _event(run_id, "TOOL_CALL_FAILED", tool="unknown", unknown=True),
            _event(run_id, "TOOL_CALL_SUCCEEDED", tool="call", second=1))
        statements: list[Select] = []
        original = SqlTraceEventStore._rows

        async def capture(self: SqlTraceEventStore, transaction: StateTransaction,
                          statement: Select) -> tuple[Mapping[str, object], ...]:
            statements.append(statement)
            for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
                compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                assert len(compiled.params) < 999
            return await original(self, transaction, statement)

        monkeypatch.setattr(SqlTraceEventStore, "_rows", capture)
        store = archive._trace_events
        assert store is not None

        async def select(**kwargs: object) -> list[int]:
            facts = await archive.state_store.read(lambda tx: store.select(
                tx, run_id, history=True, high_water=4, **kwargs))
            return [fact.sequence for fact in facts]

        assert await select(request_identities=("1",), message_identities=("2",),
                            tool_call_ids=("call", "unknown")) == [1, 2, 3]
        assert len(statements) == 1
        assert "UNION ALL" in str(statements[0])
        statements.clear()
        assert await select(request_identities=("1",), message_identities=("2",),
                            tool_call_ids=("call",), limit=2) == [1, 2]
        assert len(statements) == 1
        statements.clear()
        assert await select(message_identities=("2",)) == [2]
        assert len(statements) == 1 and "UNION" not in str(statements[0])
        statements.clear()
        assert await select(tool_call_ids=("call",)) == [3]
        assert len(statements) == 1 and "UNION" not in str(statements[0])
        statements.clear()
        assert await select() == []
        assert statements == []
        assert await select(request_identities=("1", *(f"missing-{i}" for i in range(64))),
                            message_identities=("2", *(str(i + 100) for i in range(64)))) == [1, 2]
        assert len(statements) == 3
        assert all("UNION ALL" in str(statement) for statement in statements)
        statements.clear()
        foreign = await archive.state_store.read(lambda tx: store.select(
            tx, "other-run", history=True, high_water=4,
            request_identities=("1",), message_identities=("2",)))
        assert foreign == ()


@pytest.mark.asyncio
async def test_history_associations_join_counter_and_dependency_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    async with _archive(tmp_path, "sqlite") as (archive, run_id):
        await _append(archive,
            _event(run_id, "MODEL_REQUEST_STARTED"),
            _event(run_id, "MODEL_REQUEST_SUCCEEDED", message="2"),
            _event(run_id, "TOOL_CALL_STARTED", tool="call"),
            _event(run_id, "TOOL_CALL_FAILED", tool="unknown", unknown=True),
            _event(run_id, "TOOL_CALL_SUCCEEDED", tool="call", second=1))
        statements: list[Select] = []
        original = SqlTraceEventStore._rows

        async def capture(self: SqlTraceEventStore, transaction: StateTransaction,
                          statement: Select) -> tuple[Mapping[str, object], ...]:
            statements.append(statement)
            for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
                compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                assert len(compiled.params) < 999
            return await original(self, transaction, statement)

        monkeypatch.setattr(SqlTraceEventStore, "_rows", capture)
        arguments = dict(agent_run_id=run_id, message_seqs=(2,), tool_call_ids=("call", "unknown"))
        events = await archive.read_history_associations(**arguments, event_high_water=4)
        assert [event.event_type for event in events] == [
            "MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "TOOL_CALL_STARTED"]
        assert len(statements) == 1
        assert "ai_step_run_admissions.event_high_water AS counter_value" in str(statements[0])
        statements.clear()
        assert await archive.read_history_associations(
            agent_run_id=run_id, message_seqs=(2, *range(100, 164)),
            tool_call_ids=("call", "unknown"), event_high_water=4) == events
        assert len(statements) == 3
        statements.clear()
        assert await archive.read_history_associations(
            agent_run_id=run_id, message_seqs=(999,), tool_call_ids=(), event_high_water=5) == []
        assert len(statements) == 1
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(999,), tool_call_ids=(), event_high_water=6)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(
                agent_run_id="missing", message_seqs=(2,), tool_call_ids=(), event_high_water=1)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
