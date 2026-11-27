#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP session listings preserve Runtime authorization and pagination."""
from ._runtime_test_helpers import model_features

from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.acp import ACPAgent
from linktools.ai.core import Page, Principal, SessionStatus
from linktools.ai.session import ListSessionRequest, SessionView
from linktools.ai.workspace import Workspace


def _schema_object(**values: object) -> dict[str, object]:
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", (False, True))
@pytest.mark.parametrize("filter_cwd", (False, True))
async def test_acp_session_listing_advertises_and_preserves_pagination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sdk: bool,
    filter_cwd: bool,
) -> None:
    workspace = Workspace(tmp_path, {})
    cwd = str(workspace.root / "selected") if filter_cwd else None
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
                (SessionView("first", "agent", SessionStatus.OPEN, cwd="first"),),
                "opaque-next-page",
            )
        assert request.cursor == "opaque-next-page"
        return Page((
            SessionView("second", "agent", SessionStatus.OPEN, cwd="selected"),
            SessionView("unknown", "agent", SessionStatus.OPEN),
        ), None)

    agent = ACPAgent(
        SimpleNamespace(sessions=SimpleNamespace(list=list_sessions)),
        principal=principal,
        workspace=workspace,
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
        [("first", str(workspace.root / "first"))] if cwd is None else []
    )

    second = await agent.list_sessions(cwd=cwd, cursor=first_payload["nextCursor"])
    second_payload = second.model_dump(by_alias=True) if sdk else second
    assert second_payload["nextCursor"] is None
    assert [(item["sessionId"], item["cwd"]) for item in second_payload["sessions"]] == (
        [("second", str(workspace.root / "selected")), ("unknown", str(workspace.root))]
        if cwd is None else [("second", cwd)]
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
    from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.workspace import Workspace

    from ._runtime_test_helpers import _Models
    from .test_runtime_shared_stream import _CombinedCache, _CountingStreamBackend

    class HostAuthorization(TenantAuthorizationPolicy):
        """Explicitly authorize native workspace sessions on this host."""

    class RequestError(Exception):
        @classmethod
        def invalid_params(cls, detail):
            return cls(detail)

        internal_error = invalid_params

    acp = SimpleNamespace(PROTOCOL_VERSION=1, RequestError=RequestError)
    schema = SimpleNamespace(**{
        name: _schema_object
        for name in (
            "InitializeResponse", "AgentCapabilities", "SessionCapabilities",
            "SessionListCapabilities", "Implementation", "NewSessionResponse", "PromptResponse",
        )
    })
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))

    cache = _CombinedCache()
    streams = _CountingStreamBackend()
    declaration = (
        None if coordination == "none" else ExecutionCapability(
            retain_history=retain_history,
        )
    )
    features = model_features(_Models(TestModel(call_tools=[])))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    application = ACPApplication.for_workspace(
        Workspace.load(tmp_path), features=features,
        storage=PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache,
                                          coordination=None if coordination == "none" else streams if coordination == "stream" else cache),
        execution=declaration, authorization=HostAuthorization(),
    )
    served = False

    async def serve(agent: ACPAgent) -> None:
        nonlocal served
        await agent.initialize(acp.PROTOCOL_VERSION)
        session_id = (await agent.new_session(cwd=str(tmp_path)))["sessionId"]
        response = await agent.prompt(session_id, [SimpleNamespace(type="text", text="hello")])
        assert response["stopReason"] == "end_turn"
        runtime = agent._runtime
        session = await runtime.sessions.get(session_id, principal=agent._principal)
        assert session.session_id == session_id
        timeline = await session.timeline()
        [turn] = timeline["items"]
        assert await runtime._state.execution.executions.get(
            turn["turn_id"], tenant_id=runtime.tenant_id,
        ) is None

        # A workspace's Execution declaration still configures standalone
        # Execution; the ACP session turn has its own transcript owner.
        execution = await runtime.agents.get().start("standalone")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None
        assert record.conversation is (True if declaration is None else retain_history)
        served = True

    monkeypatch.setattr("linktools.ai.acp.serve_stdio", serve)
    await application.serve()
    assert served
    assert bool(streams.open_keys) is (coordination == "stream")
    assert bool(cache.open_keys) is (coordination == "combined")
    assert not cache.closed


@pytest.mark.asyncio
async def test_embedded_sql_acp_replays_default_workspace_tool_with_session_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

    from linktools.ai.acp import ACPApplication
    from linktools.ai.capability import SessionCapability
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.persistence import PersistenceStorage

    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    class HostAuthorization(TenantAuthorizationPolicy):
        """Explicitly authorize the SQL session API on this test host."""

    class RequestError(Exception):
        @classmethod
        def invalid_params(cls, detail):
            return cls(detail)

        internal_error = invalid_params

    class Connection:
        async def session_update(self, session_id, update):
            updates.append((session_id, update))

    acp = SimpleNamespace(PROTOCOL_VERSION=1, RequestError=RequestError)
    schema = SimpleNamespace(**{
        name: _schema_object
        for name in (
            "InitializeResponse", "AgentCapabilities", "SessionCapabilities", "SessionListCapabilities",
            "Implementation", "NewSessionResponse", "PromptResponse", "LoadSessionResponse",
            "ToolCallStart", "ToolCallProgress", "TextContentBlock", "AgentMessageChunk", "UserMessageChunk",
        )
    })
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))
    updates = []
    calls = 0
    (tmp_path / "visible-note.txt").write_text("workspace fixture")

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert "list_files" in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[ToolCallPart(
                "list_files", {"path": ".", "include_glob": "visible-note.txt"}, tool_call_id="list",
            )])
        assert calls == 2
        [result] = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        assert result.tool_call_id == "list" and result.outcome == "success"
        assert "visible-note.txt" in result.content
        return ModelResponse(parts=[TextPart("I found the workspace note.")])

    features = model_features(_Models(_UsageFunctionModel(model)))
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    app = ACPApplication.for_workspace(
        Workspace.load(tmp_path), features=features,
        storage=PersistenceStorage.sqlite(tmp_path / "state.db"), authorization=HostAuthorization(),
    )

    async def serve(agent):
        agent.on_connect(Connection())
        await agent.initialize(acp.PROTOCOL_VERSION)
        session_id = (await agent.new_session(str(tmp_path)))["sessionId"]
        prompt = "Find the workspace note"
        assert (await agent.prompt(session_id, [SimpleNamespace(type="text", text=prompt)]))["stopReason"] == "end_turn"
        assert calls == 2
        updates.clear()
        await agent.load_session(str(tmp_path), session_id)
        assert calls == 2
        assert [sid for sid, _ in updates] == [session_id] * 3
        user, tool, assistant = [event for _, event in updates]
        assert user["sessionUpdate"] == "user_message_chunk" and user["content"]["text"] == prompt
        assert (tool["sessionUpdate"], tool["title"], tool["status"]) == ("tool_call", "list_files", "completed")
        assert assistant["sessionUpdate"] == "agent_message_chunk"
        assert assistant["content"]["text"] == "I found the workspace note."

    monkeypatch.setattr("linktools.ai.acp.serve_stdio", serve)
    await app.serve()


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
        captures.append(await features.resolve())
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
    assert tuple((item.kind, item.id, item.revision, item.contract) for item in captures[0].contributions) == tuple(
        (item.kind, item.id, item.revision, item.contract) for item in captures[1].contributions
    )
    assert all(capture.models["default"] is binding for capture in captures)
    assert all(capture.workspace is application.workspace for capture in captures)


@pytest.mark.asyncio
async def test_acp_native_host_listing_requires_scope_and_preserves_acl_and_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from linktools.ai.core import AuthorizationAction, PrincipalKind, TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime
    from linktools.ai.session import CreateSessionRequest

    from ._runtime_test_helpers import RuntimeUsageModels

    class RequestError(Exception):
        @classmethod
        def internal_error(cls, detail):
            return cls(detail["reason"])

    acp = SimpleNamespace(PROTOCOL_VERSION=1, RequestError=RequestError)
    schema = SimpleNamespace(**{
        name: _schema_object
        for name in (
            "InitializeResponse", "AgentCapabilities", "SessionCapabilities",
            "SessionListCapabilities", "Implementation", "ListSessionsResponse", "SessionInfo",
        )
    })
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))

    actor = Principal("caller", "default", "service")
    local = Principal("local", "default", PrincipalKind.LOCAL_TRUSTED.value)
    scoped = [f"scope-{index:02d}" for index in range(65)]
    denied = scoped[-1]
    unlisted = "different-session"
    checked = []

    class HostPolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if principal == actor and action is AuthorizationAction.SESSION_READ:
                checked.append(resource.id)
                if resource.id == denied:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    calls = []
    current_scope = scoped.copy()

    async def list_session_ids(principal):
        calls.append(principal)
        return tuple(current_scope)

    async with Runtime.open(
        "acp-list", storage=PersistenceStorage.from_root(tmp_path / "state"),
        features=model_features(RuntimeUsageModels()), authorization=HostPolicy(),
    ) as runtime:
        for session_id in (*scoped, unlisted):
            await runtime.sessions.create(CreateSessionRequest(actor, session_id, session_id))

        def can_list(initialized):
            return initialized["agentCapabilities"]["sessionCapabilities"]["list"] is not None

        unconfigured = ACPAgent(runtime, principal=actor)
        assert not can_list(await unconfigured.initialize(acp.PROTOCOL_VERSION))
        with pytest.raises(RequestError, match="host_session_scope_required"):
            await unconfigured.list_sessions()
        assert checked == []

        embedded = ACPAgent(runtime, principal=actor, list_session_ids=list_session_ids)
        assert can_list(await embedded.initialize(acp.PROTOCOL_VERSION))
        first = await embedded.list_sessions()
        first_ids = {entry["sessionId"] for entry in first["sessions"]}
        assert first_ids == set(scoped[1:-1])
        assert set(checked) == set(scoped[1:])
        cursor = first["nextCursor"]
        assert cursor is not None

        current_scope.append(unlisted)
        with pytest.raises(AIError) as changed:
            await embedded.list_sessions(cursor=cursor)
        assert changed.value.code is ErrorCode.CURSOR_INVALID
        current_scope.pop()
        second = await embedded.list_sessions(cursor=cursor)
        assert [entry["sessionId"] for entry in second["sessions"]] == [scoped[0]]
        assert second["nextCursor"] is None
        assert calls == [actor, actor, actor]

        trusted = ACPAgent(runtime, principal=local)
        assert can_list(await trusted.initialize(acp.PROTOCOL_VERSION))
        trusted_first = await trusted.list_sessions()
        trusted_ids = {entry["sessionId"] for entry in trusted_first["sessions"]}
        assert {denied, unlisted} <= trusted_ids
