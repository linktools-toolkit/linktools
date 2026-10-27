#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measured SQL write budgets retain replay, conflict, and shared-usage truth."""

import asyncio
import getpass
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.persistence import Persistence
from linktools.ai.core import RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, Metrics, MetricSource, MetricType, Observation,
)
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordReplacement, StateTransaction, StoredRecord


async def _assert_writes(engine: AsyncEngine) -> None:
    await provision_database(engine)
    metrics = Metrics.sql(engine, namespace="write-io")
    state = Persistence(PersistenceStorage.sql(engine))
    peer = Persistence(PersistenceStorage.sql(engine))
    await state.initialize(namespace="write-io", tenant_id="tenant")
    await peer.initialize(namespace="write-io", tenant_id="tenant")
    await metrics.get_observation("warmup")
    statements: list[str] = []

    def capture(conn, cursor, sql, params, context, many):
        if sql.split()[0] in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
            if many:
                assert engine.dialect.name == "mysql" and sql.startswith("UPDATE ai_state_records")
                statements.extend([sql] * len(params))
            else:
                statements.append(sql)

    def counted() -> Counter[str]:
        return Counter(sql.split()[0] for sql in statements)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        definition = MetricDefinition(
            name="app.events", revision=1, observation_kind="app.event",
            source=MetricSource.observation_count(), metric_type=MetricType.COUNTER,
            unit="1", default_aggregation=MetricAggregation.COUNT, description="original",
        )
        assert await metrics.define(definition) == definition
        assert counted() == {"INSERT": 1}
        assert await metrics.define(replace(definition, description="display-only")) == definition
        with pytest.raises(AIError, match="STORAGE_CONFLICT"):
            await metrics.define(replace(definition, unit="changed"))
        observations = tuple(Observation(
            version=1, observation_id=str(index), kind="app.event",
            occurred_at=datetime(2026, 10, 11, tzinfo=timezone.utc),
            source_namespace="source", tenant_id="tenant", status="SUCCEEDED",
            error_code=None, correlation={}, dimensions={}, measurements=(),
        ) for index in range(8))
        statements.clear()
        await metrics.record_observations(observations)
        assert counted() == {"INSERT": 1}
        statements.clear()
        await metrics.record_observations(observations)
        replay_inserts = 2 if engine.dialect.name == "mysql" else 1
        assert counted() == {"INSERT": replay_inserts, "SELECT": 1}
        statements.clear()
        mixed = replace(observations[0], observation_id="mixed-new")
        await metrics.record_observations((observations[0], mixed))
        assert counted() == {"INSERT": replay_inserts, "SELECT": 1}
        assert await metrics.get_observation("mixed-new") == mixed
        statements.clear()
        with pytest.raises(AIError, match="STORAGE_CONFLICT"):
            await metrics.record_observations((
                replace(observations[0], observation_id="rolled-back"),
                replace(observations[0], status="FAILED"),
            ))
        assert counted() == {"INSERT": replay_inserts, "SELECT": 1}
        assert await metrics.get_observation("rolled-back") is None
        await asyncio.gather(*(metrics.record_observations(observations) for _ in range(3)))
        for observation in observations:
            assert await metrics.get_observation(observation.observation_id) == observation

        concurrent = replace(observations[0], observation_id="concurrent-new")
        await asyncio.gather(*(metrics.record_observations((concurrent,)) for _ in range(3)))
        assert await metrics.get_observation(concurrent.observation_id) == concurrent
        contested = replace(observations[0], observation_id="concurrent-conflict")
        alternatives = (contested, replace(contested, status="FAILED"))
        outcomes = await asyncio.gather(*(
            metrics.record_observations((candidate,)) for candidate in alternatives
        ), return_exceptions=True)
        assert sum(outcome is None for outcome in outcomes) == 1
        failures = [outcome for outcome in outcomes if isinstance(outcome, AIError)]
        assert len(failures) == 1 and failures[0].code is ErrorCode.STORAGE_CONFLICT
        winner = alternatives[outcomes.index(None)]
        assert await metrics.get_observation(contested.observation_id) == winner

        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(model_requests=10, tool_calls=10, total_tokens=100))
        statements.clear()
        await budgets.admit_model("scope", "request")
        assert counted() == {"SELECT": 1, "INSERT": 1, "UPDATE": 1}
        statements.clear()
        settled = await budgets.settle_model("scope", "request", 7)
        settlement_updates = 2 if engine.dialect.name == "mysql" else 1
        assert counted() == {"SELECT": 1, "UPDATE": settlement_updates}
        statements.clear()
        assert await budgets.settle_model("scope", "request", 7) == settled
        assert counted() == {"SELECT": 1}
        with pytest.raises(AIError, match="STORAGE_CONFLICT"):
            await budgets.settle_model("scope", "request", 8)
        statements.clear()
        await budgets.admit_tool("scope", "tool")
        assert counted() == {"SELECT": 1, "INSERT": 1, "UPDATE": 1}
        await budgets.admit_model("scope", "unknown")
        statements.clear()
        unknown = await budgets.settle_model("scope", "unknown", None)
        assert counted() == {"SELECT": 1, "UPDATE": settlement_updates}
        assert (unknown.total_tokens, unknown.unknown_model_requests,
                unknown.in_flight_model_requests) == (7, 1, 0)
        with pytest.raises(AIError) as blocked:
            await peer.execution.budgets.admit_model("scope", "blocked")
        assert blocked.value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
        assert blocked.value.safe_details["reason"] == "unknown_usage"

        await budgets.ensure("concurrent", RunBudget(model_requests=6))
        owners = (budgets, peer.execution.budgets)
        await asyncio.gather(*(owners[index % 2].admit_model("concurrent", str(index))
                               for index in range(6)))
        await asyncio.gather(*(owners[index % 2].settle_model("concurrent", str(index), index)
                               for index in range(6)))
        await asyncio.gather(*(owner.settle_model("concurrent", "0", 0) for owner in owners))
        usage = await peer.execution.budgets.read("concurrent")
        assert (usage.model_requests, usage.total_tokens, usage.in_flight_model_requests) == (6, 15, 0)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await peer.close()
        await state.close()


async def _assert_record_cas(engine: AsyncEngine) -> None:
    store = SqlStateStore(engine)
    await store.initialize()
    records = tuple(StoredRecord(
        index.to_bytes(32, "big"), None, None, "owner", str(index), None,
        0, None, 0, None, {"index": index},
    ) for index in range(70))
    async def seed(transaction: StateTransaction) -> None:
        await transaction.insert_records(records)
    await store.mutate(seed)
    replacements = tuple(RecordReplacement(replace(
        record, state="updated", storage_revision=1,
        lease_owner="owner" if index % 2 else None, lease_fence=index,
        lease_expires_at=datetime(2026, 10, 11, tzinfo=timezone.utc) if index % 2 else None,
        data={"index": index, "updated": True},
    ), 0) for index, record in enumerate(records))
    updates: list[int] = []
    def capture_updates(conn, cursor, sql, params, context, many):
        if sql.startswith("UPDATE ai_state_records"):
            if many:
                assert engine.dialect.name == "mysql"
            updates.append(len(params) if many else 1)
    try:
        event.listen(engine.sync_engine, "before_cursor_execute", capture_updates)
        try:
            await store.mutate(lambda transaction: transaction.replace_records(replacements))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture_updates)
        if engine.dialect.name == "mysql":
            assert sum(updates) == len(replacements)
        else:
            assert 1 < sum(updates) < len(replacements)
        actual = await store.read(lambda transaction: transaction.get_records([r.key_digest for r in records]))
        assert tuple(actual[r.key_digest] for r in records) == tuple(r.record for r in replacements)

        injected = False
        def stale_one_version(conn, cursor, sql, params, context, many):
            nonlocal injected
            if sql.startswith("UPDATE ai_state_records") and not injected:
                injected = True
                conn.exec_driver_sql(
                    "UPDATE ai_state_records SET storage_revision = 2 WHERE key_digest = '"
                    + records[-1].key_digest.hex() + "'"
                )
        event.listen(engine.sync_engine, "before_cursor_execute", stale_one_version)
        try:
            with pytest.raises(AIError, match="STORAGE_CONFLICT"):
                await store.mutate(lambda transaction: transaction.replace_records(tuple(
                    RecordReplacement(replace(value.record, storage_revision=2, state="lost"), 1)
                    for value in replacements
                )))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", stale_one_version)
        assert injected
        assert await store.read(lambda transaction: transaction.get_records(list(actual))) == actual
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_write_budgets_and_atomic_replay(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    try:
        await _assert_writes(engine)
        await _assert_record_cas(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_write_budgets_and_atomic_replay(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket})
    engine = create_async_engine(url, isolation_level="READ COMMITTED")
    try:
        await _assert_writes(engine)
        await _assert_record_cas(engine)
    finally:
        await engine.dispose()
