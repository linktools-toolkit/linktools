#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP session listings preserve Runtime authorization and pagination."""
from ._runtime_test_helpers import model_features

from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.acp import ACPAgent
from linktools.ai.core import Page, Principal, SessionStatus
from linktools.ai.execution import ListSessionRequest, SessionView


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
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.workspace import Workspace

    from ._runtime_test_helpers import _Models
    from .test_runtime_shared_stream import _CombinedCache, _CountingStreamBackend

    cache = _CombinedCache()
    streams = _CountingStreamBackend()
    declaration = (
        None if coordination == "none" else ExecutionCapability(
            retain_history=retain_history,
        )
    )
    application = ACPApplication.for_workspace(
        Workspace.load(tmp_path), features=model_features(_Models(TestModel(call_tools=[]))),
        storage=PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache,
                                          coordination=None if coordination == "none" else streams if coordination == "stream" else cache),
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
    await application.serve()
    assert served
    assert bool(streams.open_keys) is (coordination == "stream")
    assert bool(cache.open_keys) is (coordination == "combined")
    assert not cache.closed


@pytest.mark.asyncio
async def test_acp_application_can_serve_again_after_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    import linktools.ai.acp as acp_module
    from linktools.ai.core import AuthorizationPolicy, service_principal
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.workspace import Workspace

    from ._runtime_test_helpers import RuntimeUsageModels

    features = RuntimeFeatures()
    binding = RuntimeUsageModels()
    features.model("default", binding)
    application = acp_module.ACPApplication.for_workspace(
        Workspace.load(tmp_path), features=features,
        storage=PersistenceStorage.in_memory(),
    )
    captures = []
    closed = []
    transport_failed = False

    @asynccontextmanager
    async def open_runtime(
        namespace: str, *, storage: PersistenceStorage, features: RuntimeFeatures,
        authorization: AuthorizationPolicy | None = None,
    ) -> AsyncIterator[SimpleNamespace]:
        assert namespace == "default"
        assert storage is application.storage
        assert authorization is application.authorization
        captures.append(await features.capture())
        try:
            yield SimpleNamespace(default_principal=service_principal("default", "acp"))
        finally:
            closed.append(True)

    async def serve(agent: ACPAgent) -> None:
        nonlocal transport_failed
        if not transport_failed:
            transport_failed = True
            raise RuntimeError("transport disconnected")

    monkeypatch.setattr(acp_module.Runtime, "open", open_runtime)
    monkeypatch.setattr(acp_module, "serve_stdio", serve)
    with pytest.raises(RuntimeError, match="transport disconnected"):
        await application.serve()
    await application.serve()

    assert len(captures) == len(closed) == 2
    assert captures[0].contributions == captures[1].contributions
    assert all(capture.models["default"] is binding for capture in captures)
    assert all(capture.workspace is application.workspace for capture in captures)
