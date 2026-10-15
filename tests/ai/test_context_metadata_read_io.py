#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context metadata is captured together before loading any payloads."""

from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_session_history_projection import _materialize, _session

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("reader", ("owner", "session"))
async def test_empty_context_batches_projection_and_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    state = _storage("sqlite", tmp_path)
    await state.initialize(namespace="context-metadata", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        repository = archive.transcript_repository
        owner = session.history_id
        assert owner is not None
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.split()[0].upper())

        async def separate_reads(owner_id):
            projection = await repository.load_projection(owner_id)
            return projection, await repository.get_head(owner_id) if projection is None else None

        results, counts = [], []
        for baseline in (True, False):
            statements.clear()
            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(repository, "_load_context_metadata", separate_reads)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    result = await repository.load_session_model_context(owner, tenant_id="tenant") if reader == "session" else await repository.load_model_context(owner)
                    results.append(result)
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            counts.append((statements.count("SELECT"), statements.count("BEGIN")))
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
        assert results[0] == results[1]
        assert results[1].model_messages() == ()
        assert counts == ([(3, 3), (2, 2)] if reader == "session" else [(2, 2), (1, 1)])
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("reader", ("owner", "session"))
async def test_context_metadata_keeps_projection_precedence_and_releases_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, reader: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="context-metadata", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        repository = archive.transcript_repository
        owner = session.history_id
        assert owner is not None
        await _materialize(state, "run", ("one",))
        load = repository._load_model_context_from_projection

        async def outside_scope(*args):
            assert active_state_scope() is None
            return await load(*args)

        monkeypatch.setattr(repository, "_load_model_context_from_projection", outside_scope)
        context = await repository.load_session_model_context(owner, tenant_id="tenant") if reader == "session" else await repository.load_model_context(owner)
        assert [message.parts[0].content for message in context.model_messages()] == ["one", "answer:one"]

        # A projection remains authoritative; the unused head is not decoded.
        decode = repository._decode_head
        def reject_head(_record):
            raise AssertionError("the projection path must not decode an unused head")
        monkeypatch.setattr(repository, "_decode_head", reject_head)
        projection, head = await repository._load_context_metadata(owner)
        assert projection is not None and head is None
        monkeypatch.setattr(repository, "_decode_head", decode)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("reader", ("owner", "session"))
async def test_empty_context_requires_head_before_session_descriptor(
    tmp_path: Path, backend: str, reader: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="context-metadata", tenant_id="tenant")
    try:
        repository = state.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        with pytest.raises(AIError) as error:
            if reader == "session":
                await repository.load_session_model_context("missing", tenant_id="tenant")
            else:
                await repository.load_model_context("missing")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert active_state_scope() is None
    finally:
        await state.close()
