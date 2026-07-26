#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Commit-scoped idempotency for EventStore + SessionStore (P5 commit-scoped idempotency).

append_once / append_messages_once reserve (stream_id, commit_id) /
(session_id, commit_id, batch_index) atomically via unique constraints, so a
retried call returns the originally-persisted record instead of re-appending.
The store does NOT dedupe via a full list-then-filter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.events.persistence.sqlalchemy import SqlAlchemyEventStore
from linktools.ai.events.payloads import RunStarted
from linktools.ai.run.models import RunnableType
from linktools.ai.session.models import (
    NewSessionMessage,
    SessionRecord,
    SessionStatus,
    MessageRole,
)
from linktools.ai.session.persistence.sqlalchemy import SqlAlchemySessionStore
from linktools.ai.storage.sqlalchemy.models import Base


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def session_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'cid.db'}")

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return async_sessionmaker(engine, expire_on_commit=False)

    return _run(_setup())


def _seed_session(session_factory, session_id="sess-1"):
    """Create a session record directly so append_messages_once can target it."""
    async def _seed():
        store = SqlAlchemySessionStore(session_factory=session_factory)
        record = SessionRecord(
            id=session_id,
            parent_id=None,
            status=SessionStatus.ACTIVE,
            version=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await store.create(record)

    _run(_seed())


# --- EventStore.append_once -------------------------------------------------


def test_append_once_replay_returns_same_envelope(session_factory):
    """A retried append_once with the same commit_id returns the SAME
    envelope (same event_id, same sequence) -- no re-append."""
    store = SqlAlchemyEventStore(session_factory=session_factory)
    payload = RunStarted(run_id="run-1", runnable_id="agent-1")

    common = dict(
        stream_id="run-1",
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="sess-1",
        runnable_id="agent-1",
        payload=payload,
    )

    async def _run_async():
        first = await store.append_once(commit_id="commit-A", **common)
        second = await store.append_once(commit_id="commit-A", **common)
        return first, second

    first, second = _run(_run_async())
    assert first.event_id == second.event_id
    assert first.sequence == second.sequence


def test_append_once_different_commit_ids_append_separately(session_factory):
    """Two different commit_ids are two distinct appends -- both sequences
    assigned, no collision."""
    store = SqlAlchemyEventStore(session_factory=session_factory)
    payload = RunStarted(run_id="run-1", runnable_id="agent-1")

    common = dict(
        stream_id="run-1",
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="sess-1",
        runnable_id="agent-1",
        payload=payload,
    )

    async def _run_async():
        first = await store.append_once(commit_id="commit-A", **common)
        second = await store.append_once(commit_id="commit-B", **common)
        return first, second

    first, second = _run(_run_async())
    assert first.event_id != second.event_id
    assert second.sequence == first.sequence + 1


# --- SessionStore.append_messages_once --------------------------------------


def test_append_messages_once_replay_returns_same_batch(session_factory):
    """A retried append_messages_once with the same commit_id returns the SAME
    message ids/sequences -- no duplicate rows."""
    _seed_session(session_factory)
    store = SqlAlchemySessionStore(session_factory=session_factory)
    messages = (
        NewSessionMessage(role=MessageRole.USER, content="hello", run_id="run-1"),
        NewSessionMessage(role=MessageRole.ASSISTANT, content="hi", run_id="run-1"),
    )

    async def _run_async():
        first = await store.append_messages_once(
            commit_id="commit-batch-1", session_id="sess-1", messages=messages
        )
        second = await store.append_messages_once(
            commit_id="commit-batch-1", session_id="sess-1", messages=messages
        )
        return first, second

    first, second = _run(_run_async())
    assert len(first) == len(second) == 2
    assert [m.id for m in first] == [m.id for m in second]
    assert [m.sequence for m in first] == [m.sequence for m in second]


def test_append_messages_once_different_commit_ids_append_separately(session_factory):
    _seed_session(session_factory)
    store = SqlAlchemySessionStore(session_factory=session_factory)
    messages = (
        NewSessionMessage(role=MessageRole.USER, content="hello", run_id="run-1"),
    )

    async def _run_async():
        first = await store.append_messages_once(
            commit_id="batch-A", session_id="sess-1", messages=messages
        )
        second = await store.append_messages_once(
            commit_id="batch-B", session_id="sess-1", messages=messages
        )
        return first, second

    first, second = _run(_run_async())
    assert first[0].id != second[0].id
    assert second[0].sequence == first[0].sequence + 1


def test_append_messages_once_then_list_returns_single_batch(session_factory):
    """After a replay, list_messages returns ONE batch (2 rows), not 4 -- the
    reserve worked."""
    _seed_session(session_factory)
    store = SqlAlchemySessionStore(session_factory=session_factory)
    messages = (
        NewSessionMessage(role=MessageRole.USER, content="hello", run_id="run-1"),
        NewSessionMessage(role=MessageRole.ASSISTANT, content="hi", run_id="run-1"),
    )

    async def _run_async():
        await store.append_messages_once(
            commit_id="batch-1", session_id="sess-1", messages=messages
        )
        # Replay.
        await store.append_messages_once(
            commit_id="batch-1", session_id="sess-1", messages=messages
        )
        return await store.list_messages("sess-1")

    listed = _run(_run_async())
    assert len(listed) == 2  # not 4
