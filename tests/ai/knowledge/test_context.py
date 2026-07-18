#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for knowledge.context: KnowledgeContext.format + format_memory."""

from datetime import datetime, timezone

from linktools.ai.knowledge.context import KnowledgeContext, format_memory
from linktools.ai.knowledge.document import Document
from linktools.ai.memory.models import MemoryRecord


def _make_document(**overrides):
    defaults = dict(
        id="d-1",
        content="first fact",
        score=None,
        source="memory",
        metadata={},
    )
    defaults.update(overrides)
    return Document(**defaults)


def _make_record(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="m-1",
        tenant_id="t1",
        owner_id="u1",
        content="remembered fact",
        category=None,
        confidence=None,
        version=0,
        created_at=now,
        updated_at=now,
        metadata={},
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def test_knowledge_context_empty():
    ctx = KnowledgeContext(documents=())
    assert ctx.format() == ""


def test_knowledge_context_formats_docs():
    doc_a = _make_document(id="d-1", content="first fact")
    doc_b = _make_document(id="d-2", content="second fact")
    out = KnowledgeContext((doc_a, doc_b)).format()
    assert out.startswith("## Knowledge")
    assert "- first fact" in out
    assert "- second fact" in out
    # one per line, prefixed with "- "
    lines = out.splitlines()
    assert lines[0] == "## Knowledge"
    assert lines[1] == "- first fact"
    assert lines[2] == "- second fact"


def test_format_memory_empty():
    assert format_memory(()) == ""


def test_format_memory_renders_content():
    rec = _make_record(content="remembered fact")
    out = format_memory((rec,))
    assert out.startswith("## Memory")
    assert "- remembered fact" in out
    lines = out.splitlines()
    assert lines[0] == "## Memory"
    assert lines[1] == "- remembered fact"
