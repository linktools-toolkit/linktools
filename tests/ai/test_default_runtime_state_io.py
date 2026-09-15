#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Runtime state I/O layout."""

import hashlib
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._factory import _default_runtime_state
from linktools.ai.runtime.state import RuntimeDomain, RuntimeRetentionMode
from linktools.ai.storage import FilesystemObjectStore, SqlObjectStore
from linktools.ai.workspace import Workspace
from linktools.commands.ai.run import _open_runtime_state


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


@pytest.mark.asyncio
async def test_local_sqlite_uses_builtin_object_store_and_self_provisions_schema(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite"
    state = RuntimeState.sqlite(database)

    await state.initialize(namespace="sqlite-io", tenant_id="tenant")
    try:
        store = state.object_store(RuntimeDomain.EXECUTION)
        assert isinstance(store, SqlObjectStore)

        payload = b"runtime-object"
        digest = hashlib.sha256(payload).hexdigest()
        stored = await store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
        assert stored.digest == digest
        assert database.exists()

        with sqlite3.connect(database) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        assert "ai_state_records" in tables
        assert "ai_objects" in tables
        assert "ai_object_chunks" in tables
        assert not (tmp_path / "objects").exists()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_existing_incompatible_sqlite_is_not_implicitly_migrated(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY)")

    state = RuntimeState.sqlite(database)
    with pytest.raises(AIError) as error:
        await state.initialize(namespace="sqlite-existing", tenant_id="tenant")

    assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {"marker"}


@pytest.mark.asyncio
async def test_cli_sqlite_state_uses_runtime_filesystem_objects(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    runtime_root = workspace.storage_root / "runtime"
    database = runtime_root / "runtime.db"
    objects_path = runtime_root / "objects"

    async with _open_runtime_state(workspace, "sqlite") as state:
        await state.initialize(namespace="sqlite-layout", tenant_id="tenant")
        try:
            store = state.object_store(RuntimeDomain.EXECUTION)
            assert isinstance(store, FilesystemObjectStore)

            payload = b"runtime-object"
            digest = hashlib.sha256(payload).hexdigest()
            await store.put(
                "payload",
                _chunks(payload),
                expected_size=len(payload),
                expected_digest=digest,
            )
        finally:
            await state.close()

    assert database.is_file()
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
    assert "ai_objects" not in tables
    assert "ai_object_chunks" not in tables
    assert objects_path.is_dir()
    assert list(objects_path.glob("*/*.bin"))
    assert list(objects_path.glob("*/*.json"))
    assert not (workspace.storage_root / "runtime.db.objects").exists()


def test_default_runtime_state_keeps_filesystem_durable_domains(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    state = _default_runtime_state(workspace)
    durable = (
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.RECOVERY,
        RuntimeDomain.TASK,
    )

    routes = tuple(state.plan.route(domain) for domain in durable)
    assert {route.kind for route in routes} == {"filesystem"}
    assert all(
        route.retention is RuntimeRetentionMode.DURABLE
        for route in routes
    )
    assert state.plan.route(RuntimeDomain.MEMORY).retention is RuntimeRetentionMode.VOLATILE
    assert state.plan.route(RuntimeDomain.ARTIFACT).retention is RuntimeRetentionMode.VOLATILE
    assert state.plan.route(RuntimeDomain.EVALUATION).retention is RuntimeRetentionMode.VOLATILE
