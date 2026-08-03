#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``lt ai history`` console-logic tests.

Drives :func:`run_history` through a :class:`FakeRuntimeClient` (no real
Runtime/storage) to verify the three views render the recorded state. Also
covers the public enumeration contract (list_sessions/list_runs/list_session_turns/get_run_detail)."""

import io
import json
import contextlib

import pytest

from linktools.ai.cli.client import FakeRuntimeClient
from linktools.ai.cli.console.history import _history_async


async def _run(client: FakeRuntimeClient, **kwargs) -> str:
    """Invoke the async history logic with an injected client, capturing stdout."""
    kwargs.setdefault("session", None)
    kwargs.setdefault("run_id", None)
    kwargs.setdefault("turn", None)
    kwargs.setdefault("project", None)
    kwargs.setdefault("remote", None)
    kwargs.setdefault("json_output", False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = await _history_async(client=client, **kwargs)
    assert rc == 0
    return buf.getvalue()


@pytest.mark.asyncio
async def test_list_sessions_renders_table() -> None:
    client = FakeRuntimeClient(sessions=[_Session("main"), _Session("demo")])
    out = await _run(client)
    assert "main" in out
    assert "demo" in out
    assert "SESSION" in out


@pytest.mark.asyncio
async def test_list_sessions_empty() -> None:
    client = FakeRuntimeClient(sessions=[])
    out = await _run(client)
    assert "no sessions" in out


@pytest.mark.asyncio
async def test_list_turns_renders_table() -> None:
    client = FakeRuntimeClient(
        session_turns=[
            _Turn(1, "run-a", "completed", "hello"),
            _Turn(2, "run-b", "failed", "world"),
        ]
    )
    out = await _run(client, session="main")
    assert "hello" in out
    assert "world" in out
    assert "completed" in out
    assert "failed" in out


@pytest.mark.asyncio
async def test_list_turns_empty() -> None:
    client = FakeRuntimeClient(session_turns=[])
    out = await _run(client, session="main")
    assert "no turns" in out


@pytest.mark.asyncio
async def test_show_turn_emits_messages() -> None:
    messages = (({"kind": "request", "parts": [{"type": "text", "content": "hi"}]},),)
    client = FakeRuntimeClient(
        session_turns=[_Turn(1, "run-a", "completed", "hi")],
        session_messages=messages,
    )
    out = await _run(client, session="main", turn=1)
    assert "[user]" in out
    assert "hi" in out


@pytest.mark.asyncio
async def test_show_run_emits_run_detail_messages() -> None:
    detail = _Detail(
        interactions=[_Interaction([{"type": "text", "content": "answer"}])]
    )
    client = FakeRuntimeClient(
        run_record=_Run("run-a", "main", "completed"),
        run_detail=detail,
    )
    out = await _run(client, session="main", run_id="run-a")
    assert "run-a" in out
    assert "[assistant]" in out
    assert "answer" in out


@pytest.mark.asyncio
async def test_json_list_sessions() -> None:
    client = FakeRuntimeClient(sessions=[_Session("main")])
    out = await _run(client, json_output=True)
    parsed = json.loads(out.strip().splitlines()[0])
    assert parsed["id"] == "main"


# -- helpers / fakes


class _Status:
    def __init__(self, value):
        self.value = value


class _Session:
    def __init__(self, sid):
        self.id = sid
        self.tenant_id = "local"
        self.updated_at = "2026-08-03T00:00:00+00:00"


class _Turn:
    def __init__(self, seq, run_id, status, input_text):
        self.sequence = seq
        self.run_id = run_id
        self.status = _Status(status)
        self.capture_state = "complete"
        self.input = input_text


class _Run:
    def __init__(self, rid, sid, status):
        self.id = rid
        self.session_id = sid
        self.status = _Status(status)


class _Interaction:
    def __init__(self, response_parts):
        self.request = {}
        self.response = {"parts": response_parts}
        self.status = "completed"


class _Detail:
    def __init__(self, interactions):
        self.interactions = interactions
        self.tool_calls = ()
        self.final_output = None
        self.status = "completed"
