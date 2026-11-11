#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native collection candidates preserve revision fencing on every SQL dialect."""

import getpass
from pathlib import Path

import pytest
from sqlalchemy import insert, literal, select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._native_content_index import _enqueue_select
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata


async def _assert_candidate_revision(engine: AsyncEngine) -> None:
    await provision_database(engine)
    state = SqlStateStore(engine)
    await state.initialize()
    edges = build_step_sql_metadata().tables["ai_native_content_edges"]
    context = state.storage_group.context
    try:
        async with context.sessions() as session:
            async with session.begin():
                await session.execute(insert(edges).values(
                    edge_kind="record", source_id=11, target_id=22, revision=0,
                ))
                source = select(literal("candidate_sequence"), literal(0), edges.c.target_id, literal(0)).where(
                    edges.c.edge_kind == "record", edges.c.source_id == 11,
                )
                await _enqueue_select(context, session, source)
                first = (await session.execute(select(edges.c.id, edges.c.revision).where(
                    edges.c.edge_kind == "candidate_sequence", edges.c.target_id == 22,
                ))).one()
                await _enqueue_select(context, session, source)
                second = (await session.execute(select(edges.c.id, edges.c.revision).where(
                    edges.c.edge_kind == "candidate_sequence", edges.c.target_id == 22,
                ))).one()
                assert second.id == first.id
                assert second.revision == first.revision + 1
                assert await session.scalar(select(edges.c.revision).where(
                    edges.c.edge_kind == "record", edges.c.source_id == 11,
                )) == 0
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_sqlite_candidate_reenqueue_preserves_target_revision(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'candidates.db'}")
    try:
        await _assert_candidate_revision(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_candidate_reenqueue_preserves_target_revision(_server: tuple[str, list[str]]) -> None:
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
        await _assert_candidate_revision(engine)
    finally:
        await engine.dispose()
