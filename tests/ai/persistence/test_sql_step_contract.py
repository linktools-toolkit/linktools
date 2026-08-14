#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialect-independent SQL StepStore checks."""

import pytest
from linktools.ai.adapter import SqlStepArchive
from linktools.ai.runtime import RuntimeDomain
from pydantic_ai_harness.step_persistence import StepStore
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine


def test_one_sql_step_archive_implements_the_public_harness_protocol() -> None:
    assert isinstance(SqlStepArchive.__new__(SqlStepArchive), StepStore)


def test_sql_table_names_and_namespace_columns_are_separate() -> None:
    source = open("linktools-ai/src/linktools/ai/adapter/_step.py", encoding="utf-8").read()
    assert '"step_runs"' in source
    assert '"step_events"' in source
    assert "namespace_key" in source
    assert "namespace=self._namespace" not in source


@pytest.mark.asyncio
async def test_sql_step_archive_provisions_its_owner_schema() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    store = SqlStepArchive(
        engine,
        namespace="namespace",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION,
    )
    try:
        await store.initialize()
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda sync_connection: inspect(sync_connection).get_table_names())
        assert {"step_runs", "step_events", "step_snapshots"} <= set(tables)
        await store.close()
    finally:
        await engine.dispose()
