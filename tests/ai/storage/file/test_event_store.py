#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/file/test_event_store.py"""
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import EventSequenceConflictError
from linktools.ai.events.envelope import EventEnvelope
from linktools.ai.events.payloads import RunStarted
from linktools.ai.storage.file.event import FileEventStore


def _event(run_id="run-1", sequence=1) -> EventEnvelope:
    return EventEnvelope(
        event_id=f"evt-{sequence}", sequence=sequence, occurred_at=datetime.now(timezone.utc),
        run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", payload=RunStarted(run_id=run_id, runnable_id="agent-1"),
    )


@pytest.mark.asyncio
async def test_append_then_list_roundtrip(tmp_path):
    store = FileEventStore(root=tmp_path)
    await store.append(_event(sequence=1))
    await store.append(_event(sequence=2))
    page = await store.list("run-1")
    assert [e.sequence for e in page.items] == [1, 2]
    assert isinstance(page.items[0].payload, RunStarted)


@pytest.mark.asyncio
async def test_list_after_sequence_filters(tmp_path):
    store = FileEventStore(root=tmp_path)
    await store.append(_event(sequence=1))
    await store.append(_event(sequence=2))
    await store.append(_event(sequence=3))
    page = await store.list("run-1", after_sequence=1)
    assert [e.sequence for e in page.items] == [2, 3]


@pytest.mark.asyncio
async def test_list_respects_limit(tmp_path):
    store = FileEventStore(root=tmp_path)
    for seq in range(1, 6):
        await store.append(_event(sequence=seq))
    page = await store.list("run-1", limit=2)
    assert [e.sequence for e in page.items] == [1, 2]


@pytest.mark.asyncio
async def test_append_with_expected_sequence_conflict_raises(tmp_path):
    store = FileEventStore(root=tmp_path)
    await store.append(_event(sequence=1))
    with pytest.raises(EventSequenceConflictError):
        await store.append(_event(sequence=1), expected_sequence=5)


@pytest.mark.asyncio
async def test_events_for_different_runs_are_isolated(tmp_path):
    store = FileEventStore(root=tmp_path)
    await store.append(_event(run_id="run-a", sequence=1))
    await store.append(_event(run_id="run-b", sequence=1))
    page_a = await store.list("run-a")
    page_b = await store.list("run-b")
    assert len(page_a.items) == 1
    assert len(page_b.items) == 1
    assert page_a.items[0].run_id == "run-a"
