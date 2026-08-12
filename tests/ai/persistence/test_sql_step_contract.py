#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialect-independent SQL StepStore checks."""

import pytest
from linktools.ai.adapter import SqlMediaStore, SqlStepStore
from linktools.ai.errors import AIError, ErrorCode
from pydantic_ai_harness.step_persistence import StepStore
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def test_one_sql_step_store_implements_the_public_harness_protocol() -> None:
    assert isinstance(SqlStepStore.__new__(SqlStepStore), StepStore)
    assert hasattr(SqlMediaStore, "put")


def test_sql_table_names_and_namespace_columns_are_separate() -> None:
    source = open("linktools-ai/src/linktools/ai/adapter/_step.py", encoding="utf-8").read()
    assert 'storage_name("step_runs")' in source
    assert 'storage_name("step_events")' in source
    assert "namespace_key" in source
    assert "namespace=self._namespace" not in source


@pytest.mark.asyncio
async def test_sql_step_store_requires_preprovisioned_schema() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    store = SqlStepStore(session_factory, "namespace")
    try:
        with pytest.raises(AIError) as error:
            await store.initialize()
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda sync_connection: inspect(sync_connection).get_table_names())
        assert tables == []
    finally:
        await engine.dispose()
