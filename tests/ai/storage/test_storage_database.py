#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageDatabase: construction does no I/O; initialize_storage is the only
schema-creating path. SQLite is single-process."""

from __future__ import annotations

from pathlib import Path

import pytest

from linktools.ai.storage.database import (
    CoordinationScope,
    StorageDatabase,
    build_sqlite_storage,
    scope_for_url,
)
from linktools.ai.storage.initialization import initialize_storage


@pytest.mark.asyncio
async def test_sqlite_construction_creates_no_file(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    storage = build_sqlite_storage(db_path)
    try:
        assert storage.coordination_scope is CoordinationScope.PROCESS
        assert storage.table_prefix == "ai_"
        # Building the database must not create the file (lazy engine).
        assert not db_path.exists()
    finally:
        await storage.engine.dispose()


@pytest.mark.asyncio
async def test_initialize_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    storage = build_sqlite_storage(db_path)
    try:
        await initialize_storage(storage)
        assert db_path.exists()
        # A second initialize must not fail (create_all is idempotent).
        await initialize_storage(storage)
    finally:
        await storage.engine.dispose()


def test_server_url_defaults_to_shared_database() -> None:
    # Pure URL inspection -- no driver import/connection (the deployment brings
    # its own server driver; storage declares none).
    assert scope_for_url("postgresql+asyncpg://u:p@h/db") is CoordinationScope.SHARED_DATABASE
    assert scope_for_url("mysql+asyncmy://u:p@h/db") is CoordinationScope.SHARED_DATABASE


def test_sqlite_url_defaults_to_process() -> None:
    assert scope_for_url("sqlite+aiosqlite:///:memory:") is CoordinationScope.PROCESS


@pytest.mark.asyncio
async def test_storage_database_carries_shared_factory_and_metadata(tmp_path: Path) -> None:
    storage = build_sqlite_storage(tmp_path / "state.sqlite3")
    try:
        assert isinstance(storage, StorageDatabase)
        assert storage.session_factory is not None
        assert storage.metadata is not None
    finally:
        await storage.engine.dispose()
