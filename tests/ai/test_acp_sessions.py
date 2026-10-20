#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP session listings preserve Runtime authorization and pagination."""

from types import SimpleNamespace

import pytest

from linktools.ai.acp import ACPAgent
from linktools.ai.core import Page, Principal, SessionStatus
from linktools.ai.runtime import ListSessionRequest, SessionView


def _schema_object(**values: object) -> dict[str, object]:
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", (False, True))
@pytest.mark.parametrize("cwd", (None, "/selected"))
async def test_acp_session_listing_advertises_and_preserves_pagination(
    monkeypatch: pytest.MonkeyPatch,
    sdk: bool,
    cwd: str | None,
) -> None:
    if sdk:
        acp = pytest.importorskip("acp")
        schema = pytest.importorskip("acp.schema")
    else:
        acp = SimpleNamespace(PROTOCOL_VERSION=1)
        schema = SimpleNamespace(**{
            name: _schema_object
            for name in (
                "InitializeResponse", "AgentCapabilities", "SessionCapabilities",
                "SessionListCapabilities", "Implementation", "ListSessionsResponse",
                "SessionInfo",
            )
        })
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))
    principal = Principal("caller", "tenant", "service")
    requests: list[ListSessionRequest] = []

    async def list_sessions(request: ListSessionRequest) -> Page[SessionView]:
        requests.append(request)
        if request.cursor is None:
            return Page(
                (SessionView("first", "agent", SessionStatus.OPEN, cwd="/first"),),
                "opaque-next-page",
            )
        assert request.cursor == "opaque-next-page"
        return Page((
            SessionView("second", "agent", SessionStatus.OPEN, cwd="/selected"),
            SessionView("unknown", "agent", SessionStatus.OPEN),
        ), None)

    agent = ACPAgent(
        SimpleNamespace(sessions=SimpleNamespace(list=list_sessions)),
        principal=principal,
        memory_scope="memory",
    )
    initialized = await agent.initialize(acp.PROTOCOL_VERSION)
    initialize_payload = initialized.model_dump(by_alias=True) if sdk else initialized
    capabilities = initialize_payload["agentCapabilities"]
    assert capabilities["loadSession"] is True
    assert capabilities["sessionCapabilities"]["list"] is not None

    first = await agent.list_sessions(cwd=cwd)
    first_payload = first.model_dump(by_alias=True) if sdk else first
    assert first_payload["nextCursor"] == "opaque-next-page"
    assert [(item["sessionId"], item["cwd"]) for item in first_payload["sessions"]] == (
        [("first", "/first")] if cwd is None else []
    )

    second = await agent.list_sessions(cwd=cwd, cursor=first_payload["nextCursor"])
    second_payload = second.model_dump(by_alias=True) if sdk else second
    assert second_payload["nextCursor"] is None
    assert [(item["sessionId"], item["cwd"]) for item in second_payload["sessions"]] == (
        [("second", "/selected"), ("unknown", "")] if cwd is None else [("second", "/selected")]
    )
    assert requests == [
        ListSessionRequest(principal, cursor=None, limit=200),
        ListSessionRequest(principal, cursor="opaque-next-page", limit=200),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", ("stream", "combined", "none"))
@pytest.mark.parametrize("retain_history", (False, True))
async def test_acp_workspace_uses_explicit_execution_coordination(
    tmp_path, monkeypatch: pytest.MonkeyPatch, coordination: str, retain_history: bool,
) -> None:
    from pydantic_ai.models.test import TestModel

    from linktools.ai.acp import ACPApplication
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import RuntimeStorage
    from linktools.ai.workspace import Workspace

    from ._runtime_test_helpers import _Models
    from .test_runtime_shared_stream import _CombinedCache, _CountingStreamBackend

    cache = _CombinedCache()
    streams = _CountingStreamBackend()
    declaration = (
        None if coordination == "none" else ExecutionCapability(
            retain_history=retain_history, coordination=streams if coordination == "stream" else cache,
        )
    )
    application = ACPApplication.for_workspace(
        Workspace.load(tmp_path), models=_Models(TestModel(call_tools=[])),
        storage=RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache),
        execution=declaration,
    )
    served = False

    async def serve(agent: ACPAgent) -> None:
        nonlocal served
        session = await agent._runtime.agents.get().create_session("acp-session")
        result = await session.run("hello", timeout_seconds=5)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        record = await agent._runtime._state.execution.executions.get(
            result.result.execution_id, tenant_id=agent._runtime.tenant_id,
        )
        assert record is not None
        assert record.conversation is (True if declaration is None else retain_history)
        assert (await agent._runtime.sessions.get(
            session.session_id, principal=agent._principal,
        )).session_id == session.session_id
        served = True

    monkeypatch.setattr("linktools.ai.acp.serve_stdio", serve)
    await application.serve(memory_scope="memory")
    assert served
    assert bool(streams.open_keys) is (coordination == "stream")
    assert bool(cache.open_keys) is (coordination == "combined")
    assert not cache.closed
