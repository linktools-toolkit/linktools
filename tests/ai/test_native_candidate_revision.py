#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline closure respects owner CAS and rollback on every SQL dialect."""

import getpass
from pathlib import Path

import pytest
from dataclasses import replace
from sqlalchemy import inspect, select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice, SqlStepStore
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import SqlObjectStore
from .test_native_online_collection import _record
from .test_step_content_cleanup import _Offline, _snapshot
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata


async def _assert_offline_owner_revision(engine: AsyncEngine) -> None:
    await provision_database(engine)
    state = SqlStateStore(engine)
    await state.initialize()
    group, archive_store = await state.storage_group.open_archive(create=True)
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    native.bind_message_archive(ContentArchive(archive_store, group, namespace="steps", tenant_id="one",
                                               object_store=SqlObjectStore.from_context(state.storage_group.context)))
    try:
        await native.save_snapshot(_snapshot("run", 1, "retained"))
        table = build_step_sql_metadata().tables["ai_step_snapshots"]
        async with engine.connect() as connection:
            first = await connection.scalar(select(table.c.sequence_root))
            assert not await connection.run_sync(lambda sync: inspect(sync).has_table("ai_native_content_edges"))
        root = NativeMessageRoot(group, native.store_key, (NativeMessageSlice(first, 1, 0, 1),))
        record = _record(encode_domain(ArchivedContextBlock(root, "c" * 64, 2)))
        await state.mutate(lambda tx: tx.insert_record(record))
        await native.save_snapshot(_snapshot("run", 2, "current"))
        assert not await state.mutate(lambda tx: tx.replace_record(
            replace(record, storage_revision=2, data={}), expected_storage_revision=1))
        async def abort(tx):
            assert await tx.delete_record(record.key, expected_storage_revision=0)
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with pytest.raises(AIError):
            await state.mutate(abort)
        retained = await native.purge_unreferenced_content(exclusive=_Offline())
        assert retained.retained_messages == 2
        assert (await native.read_message_roots((root,)))[0]
        assert await state.mutate(lambda tx: tx.delete_record(record.key, expected_storage_revision=0))
        removed = await native.purge_unreferenced_content(exclusive=_Offline())
        assert removed.deleted_messages == 1
        assert removed.retained_messages == 1
        assert (await native.purge_unreferenced_content(exclusive=_Offline())).deleted_messages == 0
    finally:
        await native.close()
        await state.close()


@pytest.mark.asyncio
async def test_sqlite_offline_closure_preserves_owner_revision(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'candidates.db'}")
    try:
        await _assert_offline_owner_revision(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_offline_closure_preserves_owner_revision(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url)
    try:
        await _assert_offline_owner_revision(engine)
    finally:
        await engine.dispose()
