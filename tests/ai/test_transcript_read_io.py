#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded transcript reads retain their captured range and integrity checks."""

from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain

from .test_incremental_projection_boundaries import _storage
from .test_runtime_storage_io_session import _listen_sql, _record_selects
from .test_session_history_projection import _materialize, _session

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("reader", ("owner", "session", "raw_context"))
async def test_bounded_transcript_read_has_a_metadata_io_budget(
    tmp_path: Path, reader: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'transcript.db'}")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    await state.initialize(namespace="transcript-read-io", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "run", ("one", "two"))
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        repository = archive.transcript_repository
        statements: list[tuple[str, object]] = []
        listener = _listen_sql(engine, statements)
        try:
            if reader == "raw_context":
                context = await repository.load_session_raw_model_context(
                    session.history_id, tenant_id="tenant", message_count=4,
                )
                messages = context.model_messages()
                assert all(value.source is not None for value in context.messages)
            else:
                stream = (
                    repository.iter_message_range(session.history_id, start=0, end=4)
                    if reader == "owner"
                    else repository.iter_session_message_range(
                        session.history_id, tenant_id="tenant", start=0, end=4,
                    )
                )
                messages = tuple([message async for message in stream])
            assert [message.parts[0].content for message in messages] == [
                "one", "answer:one", "two", "answer:two",
            ]
            # One captured owner head plus its seek index; a session also resolves its history.
            assert len(_record_selects(statements)) <= (2 if reader == "owner" else 3)
            assert sum(statement == "BEGIN" for statement, _ in statements) <= 4
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", listener)
    finally:
        await state.close()
        await engine.dispose()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("reader", ("owner", "session"))
async def test_transcript_range_excludes_appends_after_reading_starts(
    tmp_path: Path, backend: str, reader: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="transcript-read-cut", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await _materialize(state, "first", ("one", "two"))
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        stream = (
            repository.iter_message_range(session.history_id, start=1, end=4)
            if reader == "owner"
            else repository.iter_session_message_range(
                session.history_id, tenant_id="tenant", start=1, end=4,
            )
        )
        first = await anext(stream)
        await _materialize(state, "later", ("three",))
        messages = [first, *[message async for message in stream]]
        assert [message.parts[0].content for message in messages] == [
            "answer:one", "two", "answer:two",
        ]
        fresh = [message async for message in repository.iter_message_range(
            session.history_id, start=4, end=6,
        )]
        assert [message.parts[0].content for message in fresh] == ["three", "answer:three"]
        with pytest.raises(AIError) as error:
            _ = [message async for message in repository.iter_message_range(
                session.history_id, start=0, end=7,
            )]
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
