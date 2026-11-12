#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Writable archive preparation reuses its atomic initialization boundary."""

from pathlib import Path

import pytest
from sqlalchemy import event, insert, select

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._sql import _record_values
from linktools.ai.persistence._archive import ContentRecord, archive_scope_key
from linktools.ai.persistence._step_payload_owner import payload_record_values
from linktools.ai.persistence.state import STATE_GROUP_ARCHIVE_KEY, StoredRecord, record_key


@pytest.mark.asyncio
async def test_archive_startup_uses_one_initialization_transaction(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    scope = None
    for expected_reads in (3, 1):
        async with Persistence.open("archive-startup", storage=storage) as state:
            group = state.execution.executions.state_store.storage_group
            engine = group.context.engine.sync_engine
            statements: list[str] = []
            commits: list[object] = []

            def capture(connection, cursor, statement, parameters, context, many):
                if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                    statements.append(statement)

            def committed(connection):
                commits.append(connection)

            event.listen(engine, "before_cursor_execute", capture)
            event.listen(engine, "commit", committed)
            try:
                opened = await group.open_archive(create=True)
            finally:
                event.remove(engine, "before_cursor_execute", capture)
                event.remove(engine, "commit", committed)
            assert opened is not None
            assert len(statements) == expected_reads
            assert len(commits) == 1
            assert scope is None or opened[0] == scope
            scope = opened[0]
            assert await group.open_archive() == opened


@pytest.mark.asyncio
async def test_archive_startup_does_not_publish_rolled_back_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("archive-startup", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        group = state.execution.executions.state_store.storage_group
        initialize = group._initialize_archive_scope
        attempted: list[str] = []

        async def fail_after_insert(session, clock):
            scope, existing = await initialize(session, clock)
            assert not existing
            attempted.append(scope)
            raise AIError(ErrorCode.INTERNAL_ERROR, "rollback archive initialization")

        with monkeypatch.context() as patch:
            patch.setattr(group, "_initialize_archive_scope", fail_after_insert)
            with pytest.raises(AIError) as error:
                await group.open_archive(create=True)
            assert error.value.code is ErrorCode.INTERNAL_ERROR
        assert len(attempted) == 1 and group.opened_archive is None
        assert await group.open_archive() is None
        opened = await group.open_archive(create=True)
        assert opened is not None and opened[0] != attempted[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", (False, True))
async def test_archive_startup_rejects_content_without_its_owner(tmp_path: Path, typed: bool) -> None:
    async with Persistence.open("archive-startup", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        group = state.execution.executions.state_store.storage_group
        table = group.metadata.tables["ai_state_records"]
        orphan = StoredRecord(b"orphan", None, None, "orphan", b"orphan", None, 0, None, 0, None, {})
        async with group.context.engine.begin() as connection:
            if typed:
                content = ContentRecord("execution-payload", "a" * 64, 1, "raw", body=b"x")
                value = StoredRecord(
                    record_key("archive-startup", "default", "archive", "archive_content", (content.identity,)),
                    archive_scope_key("archive-startup", "default"), None, "archive_content",
                    content.identity.encode(), None, 0, None, 0, None, content.to_json(), payload_bytes=b"x",
                )
                await connection.execute(insert(group.metadata.tables["ai_step_messages"]).values(**payload_record_values(value)))
            else:
                await connection.execute(insert(table).values(store_key=STATE_GROUP_ARCHIVE_KEY, **_record_values(orphan)))
        with pytest.raises(AIError) as error:
            await group.open_archive(create=True)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert group.opened_archive is None
        async with group.context.engine.connect() as connection:
            assert (await connection.execute(select(table.c.kind))).scalars().all() == ([] if typed else ["orphan"])
