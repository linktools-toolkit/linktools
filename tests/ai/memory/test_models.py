#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for memory.models (MemoryRecord, MemoryMatch), the MemoryStore
Protocol, and the MemoryError family in errors.py. Pure data/Protocol checks
-- no I/O."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import (
    LinktoolsAIError,
    MemoryConflictError,
    MemoryError,
    MemoryNotFoundError,
)
from linktools.ai.memory.models import MemoryMatch, MemoryRecord
from linktools.ai.memory.store import MemoryStore, _UNSET


# --- MemoryRecord ------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _make_record(**overrides):
    now = _now()
    defaults = dict(
        id="m-1",
        tenant_id="t-1",
        owner_id="user-1",
        content="remember to deploy",
        category=None,
        confidence=None,
        version=0,
        created_at=now,
        updated_at=now,
        metadata={"k": "v"},
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def test_memory_record_construct():
    now = _now()
    record = MemoryRecord(
        id="m-1",
        tenant_id="t-1",
        owner_id="user-1",
        content="remember to deploy",
        category=None,
        confidence=None,
        version=0,
        created_at=now,
        updated_at=now,
        metadata={"k": "v"},
    )
    assert record.id == "m-1"
    assert record.tenant_id == "t-1"
    assert record.owner_id == "user-1"
    assert record.content == "remember to deploy"
    assert record.category is None
    assert record.confidence is None
    assert record.version == 0
    assert record.created_at == now
    assert record.updated_at == now
    assert record.metadata == {"k": "v"}


def test_memory_record_construct_with_optionals():
    now = _now()
    record = MemoryRecord(
        id="m-2",
        tenant_id="t-1",
        owner_id="user-2",
        content="fact",
        category="preference",
        confidence=0.9,
        version=3,
        created_at=now,
        updated_at=now,
        metadata={},
        user_id="user-2",
        workspace_id="ws-1",
        session_id="sess-1",
    )
    assert record.category == "preference"
    assert record.confidence == 0.9
    assert record.version == 3
    assert record.user_id == "user-2"
    assert record.workspace_id == "ws-1"
    assert record.session_id == "sess-1"


def test_memory_record_frozen():
    record = _make_record()
    with pytest.raises(FrozenInstanceError):
        record.content = "mutated"  # type: ignore[misc]


def test_memory_record_field_equality():
    now = _now()
    a = _make_record(id="m-x", created_at=now, updated_at=now)
    b = _make_record(id="m-x", created_at=now, updated_at=now)
    assert a == b


def test_memory_record_inequality():
    now = _now()
    a = _make_record(id="m-x", created_at=now, updated_at=now)
    b = _make_record(id="m-y", created_at=now, updated_at=now)
    assert a != b


# --- MemoryMatch -------------------------------------------------------------


def test_memory_match_defaults_score_to_none():
    # A keyword backend constructs MemoryMatch with no score rather than a
    # fabricated value.
    record = _make_record()
    match = MemoryMatch(record=record)
    assert match.record is record
    assert match.score is None


def test_memory_match_carries_score():
    record = _make_record()
    match = MemoryMatch(record=record, score=0.87)
    assert match.score == 0.87


def test_memory_match_frozen():
    match = MemoryMatch(record=_make_record(), score=0.5)
    with pytest.raises(FrozenInstanceError):
        match.score = 0.9  # type: ignore[misc]


def test_memory_match_equality():
    record = _make_record()
    assert MemoryMatch(record=record, score=1.0) == MemoryMatch(record=record, score=1.0)
    assert MemoryMatch(record=record, score=1.0) != MemoryMatch(record=record, score=0.5)
    assert MemoryMatch(record=record, score=None) != MemoryMatch(record=record, score=0.0)


# --- MemoryError family ------------------------------------------------------


def test_memory_error_is_linktools_ai_error():
    assert issubclass(MemoryError, LinktoolsAIError)


@pytest.mark.parametrize("exc_cls", [MemoryNotFoundError, MemoryConflictError])
def test_memory_error_subclasses(exc_cls):
    assert issubclass(exc_cls, MemoryError)


def test_memory_not_found_raises_as_memory_error():
    with pytest.raises(MemoryError):
        raise MemoryNotFoundError("nope")


def test_memory_conflict_raises_as_memory_error():
    with pytest.raises(MemoryError):
        raise MemoryConflictError("dup")


# --- MemoryStore Protocol ----------------------------------------------------


class _StubStore:
    async def get(self, memory_id): ...

    async def search(self, query, *, scope, limit=10, category=None): ...

    async def remember(self, record): ...

    async def update(
        self,
        memory_id,
        *,
        expected_version,
        content=_UNSET,
        category=_UNSET,
        confidence=_UNSET,
        metadata=_UNSET,
    ): ...

    async def forget(self, memory_id, *, expected_version): ...


def test_memory_store_is_runtime_checkable():
    assert isinstance(_StubStore(), MemoryStore)


def test_memory_store_rejects_non_implementor():
    class _Incomplete:
        async def get(self, memory_id): ...

    assert not isinstance(_Incomplete(), MemoryStore)


# --- _UNSET sentinel ---------------------------------------------------------


def test_unset_sentinel_is_distinct_from_none():
    assert _UNSET is not None
