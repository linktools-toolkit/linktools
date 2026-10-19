#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History counts capture the standard descriptor and local head together."""

from linktools.ai.runtime.state import RuntimeState
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import ConversationHistoryRecord
from linktools.ai.runtime.state._conversation_repositories import ConversationHistoryRepositoryImpl
from linktools.ai.runtime.state._history import TranscriptRepository


@pytest.mark.asyncio
async def test_history_count_uses_one_physical_select_and_refreshes_each_call(tmp_path: Path) -> None:
    storage = RuntimeState(RuntimeStorage.sqlite(tmp_path / "count.db"))
    await storage.initialize(namespace="count", tenant_id="tenant")
    statements = []

    def count(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.split()[0].upper())

    try:
        histories = storage.conversation.histories
        await histories.create(ConversationHistoryRecord("history", "session", "parent", "prefix", 3))
        transcript = storage.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        event.listen(Engine, "before_cursor_execute", count)
        assert await transcript.history_message_count("history", tenant_id="tenant") == 3
        assert statements == ["BEGIN", "SELECT"]
        event.remove(Engine, "before_cursor_execute", count)
        head = await transcript.get_head("history")
        assert head is not None

        async def advance(transaction):
            key = transcript.head_key("history")
            record = await transaction.get_record(key)
            replacement = replace(transcript._new_head_record(replace(head, message_count=2)),
                                  storage_revision=record.storage_revision + 1)
            assert await transaction.replace_record(replacement, expected_storage_revision=record.storage_revision)

        await histories.state_store.mutate(advance)
        statements.clear()
        event.listen(Engine, "before_cursor_execute", count)
        assert await transcript.history_message_count("history", tenant_id="tenant") == 5
        assert statements == ["BEGIN", "SELECT"]
    finally:
        if event.contains(Engine, "before_cursor_execute", count):
            event.remove(Engine, "before_cursor_execute", count)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("missing", "wrong_tenant", "missing_head"))
async def test_history_count_retains_missing_and_tenant_errors(tmp_path: Path, case: str) -> None:
    storage = RuntimeState(RuntimeStorage.sqlite(tmp_path / "invalid.db"))
    await storage.initialize(namespace="count", tenant_id="tenant")
    try:
        histories = storage.conversation.histories
        transcript = storage.run_store.read_store(RuntimeDomain.CONVERSATION).transcript_repository
        if case != "missing":
            await histories.create(ConversationHistoryRecord("history", "session", None, None, 0))
        if case == "missing_head":
            await histories.state_store.mutate(lambda tx: tx.delete_record(transcript.head_key("history")))
        with pytest.raises(AIError) as error:
            await transcript.history_message_count("history", tenant_id="other" if case == "wrong_tenant" else "tenant")
        assert error.value.code is (ErrorCode.STORAGE_INTEGRITY_ERROR if case == "missing_head"
                                     else ErrorCode.SESSION_HISTORY_UNAVAILABLE)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_custom_history_subclass_keeps_scalar_get_semantics(tmp_path: Path) -> None:
    storage = RuntimeState(RuntimeStorage.sqlite(tmp_path / "custom.db"))
    await storage.initialize(namespace="count", tenant_id="tenant")
    calls = []

    class CustomHistory(ConversationHistoryRepositoryImpl):
        async def get(self, history_id, *, tenant_id):
            calls.append(history_id)
            record = await super().get(history_id, tenant_id=tenant_id)
            return replace(record, parent_history_id="parent", prefix_index_head_id="prefix", inherited_message_count=7)

        async def get_with_head_in_transaction(self, transaction, history_id, *, tenant_id):
            pytest.fail("custom scalar contract must not be bypassed")

    try:
        store = storage.conversation.histories.state_store
        histories = CustomHistory(store, namespace="count", tenant_id="tenant")
        await histories.create(ConversationHistoryRecord("history", "session", None, None, 0))
        transcript = TranscriptRepository(store, object_store=None, namespace="count", tenant_id="tenant",
                                          runtime_domain=RuntimeDomain.CONVERSATION, history_repository=histories)
        assert await transcript.history_message_count("history", tenant_id="tenant") == 7
        assert calls == ["history"]
    finally:
        await storage.close()
