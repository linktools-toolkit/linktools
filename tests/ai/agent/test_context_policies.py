#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for agent.context_policies: the default Memory/Retrieval policies build
a tenant-bound scope from the RunContext and FAIL CLOSED when the context has
no tenant (empty result, never a global search)."""

import asyncio

from linktools.ai.agent.context_policies import (
    DefaultMemoryPolicy,
    DefaultRetrievalPolicy,
)
from linktools.ai.retrieval.scope import RetrievalScope
from linktools.ai.memory.scope import MemoryScope
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunnableType


def _ctx(*, tenant_id=None, user_id=None, session_id="sess-1", workspace_key=None):
    metadata = {} if workspace_key is None else {"workspace_key": workspace_key}
    return RunContext(
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace=None,
        metadata=metadata,
    )


class _RecordingStore:
    """Records the scope handed to search so the test can assert it, and
    returns a fixed tuple of records."""

    def __init__(self):
        self.last_scope = None

    async def search(self, query, *, scope, limit=10, category=None):
        self.last_scope = scope
        return ()


class _RecordingRetriever:
    def __init__(self):
        self.last_scope = None

    async def search(self, query, *, scope, limit=10):
        self.last_scope = scope
        return ()


def test_memory_policy_fails_closed_without_tenant():
    # : a run with no tenant gets NO memories -- never a global search.
    store = _RecordingStore()
    policy = DefaultMemoryPolicy(store=store)
    hits = asyncio.run(policy.select_memories(_ctx(tenant_id=None), "hello"))
    assert hits == ()
    assert store.last_scope is None  # search was never called


def test_memory_policy_builds_tenant_scope_when_tenant_present():
    store = _RecordingStore()
    policy = DefaultMemoryPolicy(store=store)
    asyncio.run(
        policy.select_memories(
            _ctx(tenant_id="t1", user_id="alice", workspace_key="ws-1"), "hello"
        )
    )
    scope = store.last_scope
    assert isinstance(scope, MemoryScope)
    assert scope.tenant_id == "t1"
    assert scope.user_id == "alice"
    assert scope.workspace_id == "ws-1"
    assert scope.session_id == "sess-1"


def test_retrieval_policy_fails_closed_without_tenant():
    retriever = _RecordingRetriever()
    policy = DefaultRetrievalPolicy(retriever=retriever)
    items = asyncio.run(policy.retrieve(_ctx(tenant_id=None), "hello"))
    assert items == ()
    assert retriever.last_scope is None


def test_retrieval_policy_builds_tenant_scope_when_tenant_present():
    retriever = _RecordingRetriever()
    policy = DefaultRetrievalPolicy(retriever=retriever)
    asyncio.run(
        policy.retrieve(
            _ctx(tenant_id="t1", user_id="alice", workspace_key="ws-1"), "hello"
        )
    )
    scope = retriever.last_scope
    assert isinstance(scope, RetrievalScope)
    assert scope.tenant_id == "t1"
    assert scope.user_id == "alice"
    assert scope.workspace_id == "ws-1"
    assert scope.session_id == "sess-1"
