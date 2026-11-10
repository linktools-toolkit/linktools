#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse context references do not read the intervening transcript pages."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence.contracts import ConversationHistoryRecord, TranscriptMessageRef

pytestmark = pytest.mark.asyncio


async def _history(state: Persistence, count: int):
    await state.initialize(namespace="sparse-windows", tenant_id="tenant")
    await state.conversation.histories.create(ConversationHistoryRecord("history", "session", None, None, 0))
    repository = state.run_store.read_store(PersistenceDomain.CONVERSATION).transcript_repository
    prepared = PreparedArchive()
    chunks = []
    for index in range(count):
        chunks.extend(await repository.prepare_chunks(
            "history", (ModelResponse(parts=[TextPart(str(index))]),),
            first_message_index=index, prepared=prepared,
        ))
    await repository.publish_archive(prepared)
    async def publish(transaction):
        await repository.append_chunks(transaction, "history", chunks)
        await repository.publish_native_archive(transaction, prepared, owner_written=True)

    await repository._store.mutate(publish)
    return repository


async def test_sparse_windows_skip_unrequested_sql_fact_pages(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "sparse.db"))
    try:
        repository = await _history(state, 513)
        engine = state.conversation.histories.state_store.storage_group.context.engine.sync_engine
        statements = []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        event.listen(engine, "before_cursor_execute", capture)
        try:
            values = await repository.load_message_spans("history", ((512, 513), (0, 1), (512, 513)))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert [[message.parts[0].content for message in span] for span in values] == [["512"], ["0"], ["512"]]
        assert sum("FROM ai_state_facts" in statement for statement in statements) == 2
        refs = tuple(TranscriptMessageRef(PersistenceDomain.CONVERSATION, "history", index)
                     for index in (512, 0, 512))
        resolved = await repository.resolve_transcript_message_refs(refs)
        assert [value.message.parts[0].content for value in resolved] == ["512", "0", "512"]
        assert tuple(value.source for value in resolved) == refs
    finally:
        await state.close()


@pytest.mark.parametrize("hint", ("present", "missing", "invalid"))
async def test_sparse_windows_keep_overlaps_empty_ranges_and_hint_fallback(
    hint: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import _history as history_module
    monkeypatch.setattr(history_module, "_TRANSCRIPT_PAGE_SIZE", 2)
    monkeypatch.setattr(history_module, "_TRANSCRIPT_SEEK_BLOCK", 4)
    state = Persistence(PersistenceStorage.in_memory())
    try:
        repository = await _history(state, 13)
        if hint != "present":
            async def remove_hint(transaction):
                key = repository._seek_key("history", 8)
                if hint == "missing":
                    await transaction.delete_record(key)
                else:
                    stored = await transaction.get_record(key)
                    assert stored is not None
                    await transaction.replace_record(replace(stored, data={}, storage_revision=stored.storage_revision + 1),
                                                     expected_storage_revision=stored.storage_revision)
            await repository._store.mutate(remove_hint)
        seeks = []
        original_seek = repository._seek_fact_sequence
        async def track_seek(transaction, owner_id, index):
            seeks.append(index)
            return await original_seek(transaction, owner_id, index)
        monkeypatch.setattr(repository, "_seek_fact_sequence", track_seek)
        windows = ((10, 12), (0, 0), (1, 3), (11, 13), (13, 13))
        values = await repository.load_message_spans("history", windows)
        assert [[message.parts[0].content for message in span] for span in values] == [
            ["10", "11"], [], ["1", "2"], ["11", "12"], [],
        ]
        assert seeks == [1, 10]
        with pytest.raises(AIError) as error:
            await repository.load_message_spans("history", ((0, 1), (14, 14)))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
