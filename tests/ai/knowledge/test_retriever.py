#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for knowledge.retriever: Retriever Protocol + MemoryRetriever.
MemoryRetriever projects MemoryRecord -> Document over an async MemoryStore."""

import asyncio

from linktools.ai.knowledge.document import Document
from linktools.ai.knowledge.retriever import MemoryRetriever, Retriever
from linktools.ai.knowledge.scope import RetrievalScope
from linktools.ai.memory.models import MemoryRecord


def _make_record(**overrides):
    defaults = dict(
        id="m-1",
        tenant_id="t1",
        owner_id="u1",
        content="hello world",
        category=None,
        confidence=None,
        version=0,
        created_at=None,  # not exercised by retriever
        updated_at=None,
        metadata={"k": "v"},
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _scope(*, tenant_id="t1", user_id="u1"):
    return RetrievalScope(tenant_id=tenant_id, user_id=user_id)


class _StubStore:
    """Dict-backed async MemoryStore stub: tenant-scoped (hard tenant filter
    + NULL-or-equal user sub-scope) + content substring, mirroring the real
    backends."""

    def __init__(self, records):
        # records: iterable of MemoryRecord
        self._records = list(records)

    async def search(self, query, *, scope, limit=10, category=None):
        hits = []
        for r in self._records:
            if r.tenant_id != scope.tenant_id:
                continue
            if (
                scope.user_id is not None
                and r.user_id is not None
                and r.user_id != scope.user_id
            ):
                continue
            if query and query not in r.content:
                continue
            hits.append(r)
        return tuple(hits[:limit])


def _run_search_basic():
    store = _StubStore([_make_record(id="m-1", owner_id="u1", user_id="u1", content="hello world")])
    retriever = MemoryRetriever(store)
    docs = asyncio.run(retriever.search("hello", scope=_scope()))
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.id == "m-1"
    assert doc.content == "hello world"
    assert doc.score is None
    assert doc.source == "memory"
    assert doc.metadata == {"k": "v"}


def test_search_basic():
    _run_search_basic()


def _run_scope_user_filters():
    # The scope's user_id is the filter: a u2 scope sees only m-2 even though
    # m-1 shares the tenant.
    store = _StubStore(
        [
            _make_record(id="m-1", owner_id="u1", user_id="u1", content="hello a"),
            _make_record(id="m-2", owner_id="u2", user_id="u2", content="hello b"),
        ]
    )
    retriever = MemoryRetriever(store)
    docs = asyncio.run(retriever.search("hello", scope=_scope(user_id="u2")))
    assert len(docs) == 1
    assert docs[0].id == "m-2"
    assert docs[0].content == "hello b"


def test_scope_user_filters():
    _run_scope_user_filters()


def _run_limit_honored():
    store = _StubStore(
        [
            _make_record(id=f"m-{i}", owner_id="u1", user_id="u1", content=f"hello {i}")
            for i in range(5)
        ]
    )
    retriever = MemoryRetriever(store)
    docs = asyncio.run(retriever.search("hello", scope=_scope(), limit=2))
    assert len(docs) == 2


def test_limit_honored():
    _run_limit_honored()


def _run_empty_result():
    store = _StubStore([_make_record(id="m-1", owner_id="u1", user_id="u1", content="hello")])
    retriever = MemoryRetriever(store)
    docs = asyncio.run(retriever.search("missing-keyword", scope=_scope()))
    assert docs == ()


def test_empty_result():
    _run_empty_result()


def _run_retriever_protocol():
    store = _StubStore([])
    retriever = MemoryRetriever(store)
    assert isinstance(retriever, Retriever)


def test_retriever_protocol_runtime_checkable():
    _run_retriever_protocol()
