#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fixtures for the SQLAlchemy storage dialect contracts.

Each parametrized case builds a fresh schema against one supported dialect:

- ``sqlite`` runs unconditionally (the in-repo verified dialect).
- ``mysql`` / ``postgresql`` run only when the operator provides a live DSN via
  ``LINKTOOLS_AI_TEST_MYSQL_DSN`` / ``LINKTOOLS_AI_TEST_POSTGRESQL_DSN``. When a
  DSN is absent the case is SKIPPED (not silently dropped) so CI, which sets
  those vars against MySQL 8.4 / PostgreSQL 16, exercises every dialect while a
  local run without databases still covers SQLite."""

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.sqlalchemy.models import Base

_DIALECTS = ["sqlite", "mysql", "postgresql"]


async def _build(dialect: str, tmp_path):
    if dialect == "sqlite":
        url = f"sqlite+aiosqlite:///{tmp_path}/{dialect}.db"
    elif dialect == "mysql":
        dsn = os.environ.get("LINKTOOLS_AI_TEST_MYSQL_DSN")
        if not dsn:
            pytest.skip("LINKTOOLS_AI_TEST_MYSQL_DSN not set; set it to a MySQL 8.4 DSN to run this case")
        url = dsn
    elif dialect == "postgresql":
        dsn = os.environ.get("LINKTOOLS_AI_TEST_POSTGRESQL_DSN")
        if not dsn:
            pytest.skip("LINKTOOLS_AI_TEST_POSTGRESQL_DSN not set; set it to a PostgreSQL 16 DSN to run this case")
        url = dsn
    else:  # pragma: no cover - param is constrained
        raise AssertionError(f"unknown dialect: {dialect}")

    engine = create_async_engine(url, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


@pytest_asyncio.fixture(params=_DIALECTS)
async def sql_asset_backend(request, tmp_path):
    from linktools.ai.storage.sqlalchemy.asset import SqlAlchemyAssetBackend

    engine, session_factory = await _build(request.param, tmp_path)
    backend = SqlAlchemyAssetBackend(session_factory=session_factory)
    yield backend
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception:
        pass
    await engine.dispose()
