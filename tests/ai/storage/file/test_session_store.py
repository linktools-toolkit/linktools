#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/file/test_session_store.py"""
from datetime import datetime, timezone

import pytest

from linktools.ai.session.models import MessageRole, SessionMessage, SessionRecord, SessionStatus
from linktools.ai.storage.file.session import FileSessionStore


def _record(session_id="session-1") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(id=session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1, created_at=now, updated_at=now)


def _message(session_id="session-1", sequence=1, role=MessageRole.USER, content="hi") -> SessionMessage:
    return SessionMessage(
        id=f"{session_id}-{sequence}", session_id=session_id, sequence=sequence, role=role,
        content=content, run_id=None, created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_create_then_get_roundtrip(tmp_path):
    store = FileSessionStore(root=tmp_path)
    created = await store.create(_record())
    fetched = await store.get("session-1")
    assert fetched is not None
    assert fetched.id == "session-1"
    assert created == fetched


@pytest.mark.asyncio
async def test_get_missing_returns_none(tmp_path):
    store = FileSessionStore(root=tmp_path)
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_append_then_list_messages_in_order(tmp_path):
    store = FileSessionStore(root=tmp_path)
    await store.create(_record())
    await store.append_messages("session-1", (_message(sequence=1, content="hi"), _message(sequence=2, role=MessageRole.ASSISTANT, content="hello")))
    messages = await store.list_messages("session-1")
    assert [m.content for m in messages] == ["hi", "hello"]
    assert messages[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_list_messages_after_sequence_filters(tmp_path):
    store = FileSessionStore(root=tmp_path)
    await store.create(_record())
    await store.append_messages("session-1", (_message(sequence=1), _message(sequence=2), _message(sequence=3)))
    messages = await store.list_messages("session-1", after_sequence=1)
    assert [m.sequence for m in messages] == [2, 3]


@pytest.mark.asyncio
async def test_update_status_and_metadata(tmp_path):
    store = FileSessionStore(root=tmp_path)
    await store.create(_record())
    updated = await store.update("session-1", status=SessionStatus.ARCHIVED, metadata={"k": "v"})
    assert updated.status == SessionStatus.ARCHIVED
    assert dict(updated.metadata) == {"k": "v"}
    assert updated.version == 2


@pytest.mark.asyncio
async def test_sessions_are_isolated(tmp_path):
    store = FileSessionStore(root=tmp_path)
    await store.create(_record(session_id="session-a"))
    await store.create(_record(session_id="session-b"))
    await store.append_messages("session-a", (_message(session_id="session-a", sequence=1),))
    messages_a = await store.list_messages("session-a")
    messages_b = await store.list_messages("session-b")
    assert len(messages_a) == 1
    assert len(messages_b) == 0
