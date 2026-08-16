#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialect-independent SQL StepStore checks."""

import pytest
from linktools.ai.runtime.state._steps import SqlStepArchive
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state import build_step_sql_metadata
from linktools.ai.migrate import provision_runtime_database
from pydantic_ai_harness.step_persistence import StepStore
from sqlalchemy import MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine


def test_one_sql_step_archive_implements_the_public_harness_protocol() -> None:
    assert isinstance(SqlStepArchive.__new__(SqlStepArchive), StepStore)


def test_sql_table_names_and_namespace_columns_are_separate() -> None:
    source = open("linktools-ai/src/linktools/ai/runtime/state/_steps.py", encoding="utf-8").read()
    assert '"ai_step_runs"' in source
    assert '"ai_step_events"' in source
    assert "namespace_digest" in source
    assert "namespace=self._namespace" not in source


def test_step_schema_registers_only_owned_tables_in_shared_metadata() -> None:
    metadata = MetaData()
    result = build_step_sql_metadata(
        RuntimeDomain.EXECUTION,
        metadata=metadata,
    )

    assert result is metadata
    assert set(metadata.tables) == {
        "ai_step_runs",
        "ai_step_events",
        "ai_step_snapshots",
    }


@pytest.mark.asyncio
async def test_sql_step_archive_validates_its_owner_schema() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await provision_runtime_database(engine, domains=(RuntimeDomain.EXECUTION,))
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
        assert {"ai_step_runs", "ai_step_events", "ai_step_snapshots"} <= set(tables)
        await store.close()
    finally:
        await engine.dispose()
