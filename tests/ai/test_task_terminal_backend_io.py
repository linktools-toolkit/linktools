#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure terminal publication separately from provisioning and graph preparation."""

from collections import Counter

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._task_test_helpers import admit_graph


async def _publish(engine, width, outcome, wire):
    namespace = f"terminal-{width}-{outcome}"
    counts = Counter()
    disposed = []
    def on_dispose(_engine):
        disposed.append(True)
    event.listen(engine.sync_engine, "engine_disposed", on_dispose)
    async with Persistence.open(namespace, storage=PersistenceStorage.sql(engine), tenant_id="tenant") as state:
        repository = state.task.tasks
        await admit_graph(state, TaskGraph("graph", tuple(TaskNode(f"node-{i}") for i in range(width))))
        for i in range(width):
            lease = await repository.claim("graph", f"node-{i}", tenant_id="tenant", owner="worker", lease_seconds=30)
            if i != width - 1:
                await repository.complete(lease, tenant_id="tenant", execution_id=f"execution-{i}", result_digest="a" * 64)
        def sql(_connection, _cursor, statement, _params, _context, many):
            counts[statement.lstrip().split()[0].upper()] += 1
            counts["executemany"] += bool(many)
        def commit(_connection):
            counts["commit"] += 1
        def checkout(*_args):
            counts["checkout"] += 1
        listeners = (("before_cursor_execute", sql), ("commit", commit), ("checkout", checkout))
        for name, callback in listeners:
            event.listen(engine.sync_engine, name, callback)
        wire.clear()
        try:
            if outcome == "fail":
                await repository.fail(lease, tenant_id="tenant", error_code=ErrorCode.TASK_NODE_FAILED.value, error_digest="b" * 64)
            else:
                await repository.complete(lease, tenant_id="tenant", execution_id="last-execution", result_digest="a" * 64)
        finally:
            for name, callback in listeners:
                event.remove(engine.sync_engine, name, callback)
        terminal_wire = Counter(wire)
        expected = await repository.graph_state("graph", tenant_id="tenant")
        expected_events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=1000)
        expected_results = await repository.get_results("graph", tuple(f"node-{i}" for i in range(width)), tenant_id="tenant")
        assert expected.status is (TaskStatus.FAILED if outcome == "fail" else TaskStatus.SUCCEEDED)
        assert len(expected_results) == width - (outcome == "fail")
        assert counts["commit"] == 1
    # A borrowed engine remains usable and the entire terminal publication survives reopen.
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT 1")) == 1
    async with Persistence.open(namespace, storage=PersistenceStorage.sql(engine), tenant_id="tenant") as reopened:
        repository = reopened.task.tasks
        assert await repository.graph_state("graph", tenant_id="tenant") == expected
        assert await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=1000) == expected_events
        assert await repository.get_results("graph", tuple(f"node-{i}" for i in range(width)), tenant_id="tenant") == expected_results
    event.remove(engine.sync_engine, "engine_disposed", on_dispose)
    assert not disposed
    if engine.dialect.name == "mysql":
        assert terminal_wire["UPDATE"] > 0
        assert terminal_wire["COMMIT"] == 1
    print("TASK_BACKEND_IO", engine.dialect.name, width, outcome, dict(counts), "wire", dict(terminal_wire))


@pytest.mark.asyncio
@pytest.mark.parametrize("width", (1, 32))
@pytest.mark.parametrize("outcome", ("complete", "fail"))
async def test_sqlite_terminal_io_survives_reopen(tmp_path, width, outcome):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}", pool_size=1, max_overflow=0)
    try:
        await provision_database(engine)
        await _publish(engine, width, outcome, [])
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("width", (1, 32))
@pytest.mark.parametrize("outcome", ("complete", "fail"))
async def test_mysql_terminal_wire_io_survives_reopen(_server, monkeypatch, width, outcome):
    from asyncmy.connection import Connection
    from asyncmy.constants.COMMAND import COM_QUERY

    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
    engine = create_async_engine(url, pool_size=1, max_overflow=0)
    owned = set()
    wire = []
    event.listen(engine.sync_engine, "connect", lambda connection, _record: owned.add(connection.driver_connection))
    original = Connection._execute_command
    async def execute_command(connection, command, sql):
        result = await original(connection, command, sql)
        if connection in owned and command == COM_QUERY:
            statement = sql.decode("utf-8") if isinstance(sql, bytes) else sql
            wire.append(statement.lstrip().split()[0].upper())
        return result
    monkeypatch.setattr(Connection, "_execute_command", execute_command)
    try:
        await provision_database(engine)
        await _publish(engine, width, outcome, wire)
    finally:
        await engine.dispose()
