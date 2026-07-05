#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_context.py"""
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunnableType


def test_run_context_construction():
    ctx = RunContext(
        run_id="run-1", root_run_id="run-1", parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None,
        workspace=None,
    )
    assert ctx.run_id == "run-1"
    assert ctx.runnable_type == RunnableType.AGENT
    assert ctx.workspace is None
    assert dict(ctx.metadata) == {}


def test_run_context_is_frozen():
    import pytest
    ctx = RunContext(
        run_id="run-1", root_run_id="run-1", parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", runnable_type=RunnableType.AGENT, user_id=None, tenant_id=None,
        workspace=None,
    )
    with pytest.raises(Exception):
        ctx.run_id = "run-2"
