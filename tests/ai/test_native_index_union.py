#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-store Native existence checks batch kinds without relaxing restore rules."""

from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import delete, event, insert, select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence._native_content_index import NativeContentUnavailable, index_nodes
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.steps import native_store_key, native_store_locator
from linktools.ai.storage import create_sql_storage_context


def _identity(value: str) -> str:
    return sha256(value.encode()).hexdigest()


@pytest.mark.asyncio
async def test_native_index_union_batches_both_kinds_with_bounded_parameters(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch.db'}")
    context = create_sql_storage_context(engine)
    metadata = build_step_sql_metadata()
    sequences, messages = (metadata.tables[name] for name in ("ai_step_sequences", "ai_step_messages"))
    store = native_store_key("index", "tenant")
    identity = {"store_key": store, "store_locator": native_store_locator(store)}
    rows = tuple({"sha256": _identity(str(i)), "message_sha256": _identity(str(i)),
                  "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None} for i in range(129))
    statements = []

    def capture(connection: object, cursor: object, statement: str, parameters: object, execution: object, many: bool) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(execution.compiled.statement)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
            await connection.execute(insert(sequences), [{**identity, **row} for row in rows])
            await connection.execute(insert(messages), [{**identity, "sha256": row["sha256"], "message": b"{}",
                "compression": "raw", "raw_size": 2} for row in rows])
            expected = dict((await connection.execute(select(sequences.c.sha256, sequences.c.id))).all())
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        async with context.sessions() as session:
            assert await index_nodes(context, session, store, ()) == {}
            assert statements == []
            assert await index_nodes(context, session, store, rows) == expected
        assert len(statements) == 2
        for statement in statements:
            for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
                compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                assert "UNION ALL" in str(compiled)
                assert len(compiled.params) <= 262
    finally:
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "message", "child", "own", "foreign_store", "raw_store"))
@pytest.mark.parametrize("partial", (False, True))
async def test_native_index_union_preserves_required_partial_and_store_boundaries(tmp_path: Path, damage: str, partial: bool) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'source.db'}")
    context = create_sql_storage_context(engine)
    metadata = build_step_sql_metadata()
    sequences, messages = (metadata.tables[name] for name in ("ai_step_sequences", "ai_step_messages"))
    store = native_store_key("index", "tenant")
    actual = native_store_key("index", "other") if damage in {"foreign_store", "raw_store"} else store
    identity = {"store_key": actual, "store_locator": native_store_locator(store if damage == "raw_store" else actual)}
    own, child, message = (_identity(name) for name in ("own", "child", "message"))
    row = {"sha256": own, "message_sha256": message, "tree_count": 2,
           "left_sha256": child, "right_sha256": None, "next_sha256": None}
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
            await connection.execute(insert(sequences), [{**identity, **row},
                {**identity, **row, "sha256": child, "tree_count": 1, "left_sha256": None}])
            await connection.execute(insert(messages), {**identity, "sha256": message, "message": b"{}",
                "compression": "raw", "raw_size": 2})
            if damage == "message":
                await connection.execute(delete(messages))
            elif damage in {"child", "own"}:
                await connection.execute(delete(sequences).where(sequences.c.sha256 == (child if damage == "child" else own)))
        async with context.sessions() as session:
            if damage in {"own", "foreign_store", "raw_store"} or (not partial and damage != "none"):
                with pytest.raises(NativeContentUnavailable):
                    await index_nodes(context, session, store, (row,), partial_restore=partial)
            else:
                found = await index_nodes(context, session, store, (row,), partial_restore=partial)
                assert own in found
                assert (child in found) is (damage != "child")
    finally:
        await context.close()
        await engine.dispose()
