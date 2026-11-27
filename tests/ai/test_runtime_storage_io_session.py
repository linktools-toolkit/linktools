#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical SQLite I/O contracts for the canonical Session owner."""

from contextlib import contextmanager

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._ordinary_lineage import decode_session_lineage
from linktools.ai.persistence.turns import SessionTurnKey

from .test_session_turn_sql import _TRANSCRIPT, accept, store

pytestmark = pytest.mark.asyncio


@contextmanager
def _session_statements(engine):
    statements = []

    def capture(connection, _cursor, statement, _parameters, _context, _executemany):
        sql = statement.lstrip().upper()
        if sql.startswith(("SELECT", "UPDATE", "INSERT", "DELETE")):
            statements.append((sql, connection.get_transaction()))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_session_list_reads_each_keyset_page_without_generation_probe(store):
    engine, _metadata, turns = store
    await turns.create("next", config={})
    await turns.create("excluded", config={})

    with _session_statements(engine) as first_sql:
        first = await turns.session_page(session_ids=("chat", "next"), before=None, limit=1)
    assert [item["session_key"] for item in first] == [b"next"]
    with _session_statements(engine) as next_sql:
        next_page = await turns.session_page(session_ids=("chat", "next"), before=first[0]["id"], limit=1)
    assert [item["session_key"] for item in next_page] == [b"chat"]
    for queries in (first_sql, next_sql):
        [(statement, _)] = queries
        assert statement.startswith("SELECT")
        assert "AI_SESSIONS" in statement and "AI_STATE_COUNTERS" not in statement


async def test_session_fork_reads_two_headers_and_writes_once_atomically(store):
    engine, _metadata, turns = store
    with _session_statements(engine) as queries:
        assert await turns.fork_session("chat", "target", expected_revision=0, authorized_revision=0, request_id="fork")

    assert [sql.split()[0] for sql, _ in queries] == ["SELECT", "SELECT", "UPDATE", "INSERT"]
    assert all("AI_SESSIONS" in sql for sql, _ in queries)
    first_transaction = queries[0][1]
    assert first_transaction is not None
    assert all(transaction is first_transaction for _, transaction in queries)
    assert (await turns.get_session("chat"))["revision"] == 1
    assert (await turns.get_session("target"))["lineage_json"]["source"] == "chat"


async def test_session_fork_replay_only_reads_existing_target(store):
    engine, _metadata, turns = store
    assert await turns.fork_session("chat", "target", expected_revision=0, authorized_revision=0, request_id="fork")
    target = await turns.get_session("target")
    before = await turns.get_session("chat")

    with _session_statements(engine) as queries:
        assert await turns.fork_session("chat", "target", expected_revision=0,
                                        authorized_revision=before["revision"], request_id="fork")
    [(statement, _)] = queries
    assert statement.startswith("SELECT") and "AI_SESSIONS" in statement
    assert await turns.get_session("target") == target
    assert await turns.get_session("chat") == before

    with pytest.raises(AIError) as conflict:
        await turns.fork_session("chat", "target", expected_revision=1, authorized_revision=1, request_id="fork")
    assert conflict.value.code is ErrorCode.STORAGE_CONFLICT


async def test_session_fork_replay_uses_frozen_source_prefix_without_copying_turns(store):
    engine, _metadata, turns = store
    assert await accept(turns, "first", request="first")
    assert await turns.finish(SessionTurnKey("chat", "first"), generation=1, turn_no=1,
                              state="succeeded", transcript=_TRANSCRIPT)
    source = await turns.get_session("chat")
    with _session_statements(engine) as queries:
        assert await turns.fork_session("chat", "target", expected_revision=source["revision"],
                                        authorized_revision=source["revision"], request_id="fork")
    assert all("AI_SESSIONS" in sql for sql, _ in queries)

    after_fork = await turns.get_session("chat")
    assert await accept(turns, "later", revision=after_fork["revision"], through=1, request="later")
    assert await turns.finish(SessionTurnKey("chat", "later"), generation=1, turn_no=2,
                              state="succeeded", transcript=_TRANSCRIPT)
    current = await turns.get_session("chat")
    assert await turns.fork_session("chat", "target", expected_revision=source["revision"],
                                    authorized_revision=current["revision"], request_id="fork")
    target = await turns.get_session("target")
    assert target["last_turn_no"] == 1
    inherited, = await turns.turn_page("target", through=target["last_turn_no"], limit=10,
                                       lineage=decode_session_lineage(target["lineage_json"]))
    assert inherited["session_key"] == b"chat" and inherited["turn_id"] == b"first"
