#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for knowledge.retriever: Retriever Protocol + MemoryRetriever.
MemoryRetriever projects MemoryRecord -> Document over an async MemoryStore."""

import asyncio

from linktools.ai.knowledge.document import Document
from linktools.ai.knowledge.retriever import MemoryRetriever, Retriever
from linktools.ai.memory.models import MemoryRecord


def _make_record(**overrides):
    defaults = dict(
        id="m-1",
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


class _StubStore:
    """Dict-backed async MemoryStore stub: filters by owner_id + content substring."""

    def __init__(self, records):
        # records: iterable of MemoryRecord
        self._records = list(records)

    async def search(self, query, *, owner_id=None, category=None, limit=10):
        hits = []
        for r in self._records:
            if owner_id is not None and r.owner_id != owner_id:
                continue
            if query and query not in r.content:
                continue
            hits.append(r)
        return tuple(hits[:limit])


def _run_search_basic():
    store = _StubStore([_make_record(id="m-1", owner_id="u1", content="hello world")])
    retriever = MemoryRetriever(store, owner_id="u1")
    docs = asyncio.run(retriever.search("hello"))
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


def _run_filter_overrides_owner():
    store = _StubStore([
        _make_record(id="m-1", owner_id="u1", content="hello a"),
        _make_record(id="m-2", owner_id="u2", content="hello b"),
    ])
    retriever = MemoryRetriever(store, owner_id="u1")
    docs = asyncio.run(retriever.search("hello", filters={"owner_id": "u2"}))
    assert len(docs) == 1
    assert docs[0].id == "m-2"
    assert docs[0].content == "hello b"


def test_filter_overrides_owner():
    _run_filter_overrides_owner()


def _run_limit_honored():
    store = _StubStore([
        _make_record(id=f"m-{i}", owner_id="u1", content=f"hello {i}")
        for i in range(5)
    ])
    retriever = MemoryRetriever(store, owner_id="u1")
    docs = asyncio.run(retriever.search("hello", limit=2))
    assert len(docs) == 2


def test_limit_honored():
    _run_limit_honored()


def _run_empty_result():
    store = _StubStore([_make_record(id="m-1", owner_id="u1", content="hello")])
    retriever = MemoryRetriever(store, owner_id="u1")
    docs = asyncio.run(retriever.search("missing-keyword"))
    assert docs == ()


def test_empty_result():
    _run_empty_result()


def _run_retriever_protocol():
    store = _StubStore([])
    retriever = MemoryRetriever(store)
    assert isinstance(retriever, Retriever)


def test_retriever_protocol_runtime_checkable():
    _run_retriever_protocol()
