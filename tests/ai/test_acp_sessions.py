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
