#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The default ai-web Agent delegates through a native Step-backed turn."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel


_SESSION = "delegation"
_HEADERS = {"X-LinkTools-Console": "1"}
_CHILD_TASK = "Look up fixture entry A"
_CHILD_OUTPUT = "fixture entry A exists"
_ANSWER = "The reviewer found fixture entry A."
_DELEGATE_CALL = "delegate-reviewer"


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize this SQL session through the host policy."""


@asynccontextmanager
async def _events(app):
    # HTTPX's ASGITransport buffers streaming responses. Use the actual SSE
    # ASGI request so the initial subscription can precede the POST.
    path = "/api/session/events"
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "GET", "scheme": "http", "http_version": "1.1",
        "path": path, "raw_path": path.encode(), "root_path": "",
        "query_string": f"session_id={_SESSION}".encode(), "headers": [],
        "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345),
    }
    disconnect = asyncio.Event()
    frames = asyncio.Queue()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200, message
        elif message["type"] == "http.response.body":
            for line in message.get("body", b"").splitlines():
                if line.startswith(b"data: "):
                    frames.put_nowait(json.loads(line[6:]))

    task = asyncio.create_task(app(scope, receive, send))
    try:
        yield frames
    finally:
        disconnect.set()
        await task


def _model(role, fn):
    binding = _Models(_UsageFunctionModel(fn))
    binding.route_id = role
    binding.model_identity = f"test:subagent-{role}"
    binding.model_digest = ("a" if role == "parent" else "b") * 64
    binding.contract = {"provider": "test", "model": f"subagent-{role}"}
    return binding


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel,trace", ((False, False), (True, False), (False, True)),
                         ids=("success", "cancel", "cli-default-trace"))
async def test_ai_web_native_delegation_replays_exact_turn_without_relaunch(
    tmp_path: Path, cancel: bool, trace: bool,
) -> None:
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    entered, release, tool_cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls: list[str] = []
    child_ids: list[str] = []
    returned_ids: list[str] = []

    async def lookup(context: AgentContext[None], entry: str) -> str:
        assert entry == "A"
        calls.append("child tool")
        child_ids.append(context.execution_id)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return _CHILD_OUTPUT

    async def child(messages, info):
        calls.append("child model")
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        assert prompts == [_CHILD_TASK]
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            assert [(part.tool_call_id, part.content) for part in results] == [("lookup-A", _CHILD_OUTPUT)]
            return ModelResponse(parts=[TextPart(_CHILD_OUTPUT)])
        assert "lookup" in {tool.name for tool in info.function_tools}
        assert "delegate_task" not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[ToolCallPart("lookup", {"entry": "A"}, tool_call_id="lookup-A")])

    async def parent(messages, info):
        calls.append("parent model")
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            [result] = results
            assert result.tool_call_id == _DELEGATE_CALL and result.outcome == "success"
            assert result.content["status"] == "SUCCEEDED"
            assert result.content["output"] == {"text": _CHILD_OUTPUT}
            returned_ids.append(result.content["execution_id"])
            return ModelResponse(parts=[TextPart(_ANSWER)])
        assert "delegate_task" in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "reviewer", "task": _CHILD_TASK}, tool_call_id=_DELEGATE_CALL,
        )])

    name = "canonical-web-subagent"
    features = RuntimeFeatures(name)
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="parent", allow_subagents=("reviewer",), allow_tools=(), allow_skills=())
    features.agent("reviewer", model="child", allow_subagents=(), allow_tools=("lookup",), allow_skills=())
    if trace:
        # Like the CLI, use the default unbuffered Trace policy with SQL Step.
        features.capability(TraceCapability(), scope="runtime")
    storage = PersistenceStorage.from_root(tmp_path)
    params = {"session_id": _SESSION}
    payload = {"prompt": "Use the reviewer to inspect fixture A", "request_id": "turn-once"}
    expected_calls = ["parent model", "child model", "child tool"]
    if not cancel:
        expected_calls += ["child model", "parent model"]

    async def request(http):
        response = await http.post("/api/session/messages", params=params, headers=_HEADERS, json=payload)
        assert response.status_code == 202, response.text
        return response.json()["turn_id"]

    @asynccontextmanager
    async def open_host():
        async with (
            Persistence.open(name, storage=storage) as state,
            Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
        ):
            app = create_app(runtime=runtime, local_workspace=True)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://127.0.0.1:8765") as http:
                yield state, runtime, app, http

    def assert_answer(frame):
        if "output" in frame:
            assert frame["output"] == {"text": _ANSWER}
        else:
            assert any(item["item_kind"] == "assistant" and item["content"] == _ANSWER for item in frame["items"])

    async with open_host() as (state, runtime, app, http):
        created = await http.post("/api/sessions", headers=_HEADERS,
                                  json={"session_id": _SESSION, "request_id": "create"})
        assert created.status_code == 201, created.text
        async with _events(app) as frames:
            try:
                async with asyncio.timeout(10):
                    snapshot = await frames.get()
                    assert snapshot["type"] == "snapshot" and snapshot["session"]["active_turn_id"] is None
                    turn_id = await request(http)
                    await entered.wait()
                    assert await request(http) == turn_id
                    if cancel:
                        stopped = await http.post("/api/session/cancel", params=params, headers=_HEADERS,
                                                  json={"request_id": "stop", "turn_id": turn_id})
                        assert stopped.status_code == 200 and stopped.json()["accepted"], stopped.text
                        await tool_cancelled.wait()
                        assert not release.is_set()
                    else:
                        release.set()
                    tool_statuses = []
                    while True:
                        frame = await frames.get()
                        if frame.get("turn_id") != turn_id:
                            continue
                        if frame["type"] == "tool" and frame["item"]["key"] == f"tool:{_DELEGATE_CALL}":
                            tool_statuses.append(frame["item"]["status"])
                        if frame["type"] == "terminal":
                            assert frame["state"] == ("cancelled" if cancel else "succeeded"), frame
                            if not cancel:
                                assert_answer(frame)
                            break
                    if cancel:
                        assert tool_statuses and tool_statuses[0] == "RUNNING" and "SUCCEEDED" not in tool_statuses
                    else:
                        assert tool_statuses == ["RUNNING", "SUCCEEDED"]
            finally:
                release.set()
        assert calls == expected_calls
        [child_id] = child_ids
        assert child_id != turn_id and returned_ids == ([] if cancel else [child_id])
        assert await state.execution.executions.get(turn_id, tenant_id=runtime.tenant_id) is None
        assert await state.conversation.sessions.get(_SESSION, tenant_id=runtime.tenant_id) is None

    async with open_host() as (_, runtime, _, http):
        assert await request(http) == turn_id
        stored = await http.get("/api/session", params={**params, "turn_id": turn_id})
        assert stored.status_code == 200, stored.text
        published = stored.json()
        assert published["status"] == ("CANCELLED" if cancel else "SUCCEEDED") and published["conversation_committed"]
        if not cancel:
            assert_answer(published)
        assert calls == expected_calls and child_ids == [child_id]
        assert returned_ids == ([] if cancel else [child_id])
        if trace:
            principal = runtime.default_principal
            parent_trace = await runtime.history.turn_trace(_SESSION, turn_id, principal=principal)
            child_trace = await runtime.history.trace(child_id, principal=principal)
            assert parent_trace.items and all(item.turn_id == turn_id for item in parent_trace.items)
            assert child_trace.items and all(item.execution_id == child_id for item in child_trace.items)
            parent_tools = {item.payload.get("tool_call_id") for item in parent_trace.items
                            if item.payload["kind"] == "TOOL_RESULT"}
            child_tools = {item.payload.get("tool_call_id") for item in child_trace.items
                           if item.payload["kind"] == "TOOL_RESULT"}
            assert parent_tools == {_DELEGATE_CALL}
            assert child_tools == {"lookup-A"}



@pytest.mark.asyncio
async def test_approved_parent_tool_continues_to_native_subagent(tmp_path: Path) -> None:
    calls: list[str] = []
    child_ids: list[str] = []

    async def confirm_fixture(_context: AgentContext[None]) -> str:
        if not calls:
            calls.append("approval requested")
            raise ApprovalRequired()
        assert calls == ["approval requested"]
        calls.append("approved parent effect")
        return "confirmed"

    async def lookup(context: AgentContext[None]) -> str:
        calls.append("child tool")
        child_ids.append(context.execution_id)
        return _CHILD_OUTPUT

    async def child(messages, _info):
        if isinstance(messages[-1].parts[-1], ToolReturnPart):
            assert messages[-1].parts[-1].content == _CHILD_OUTPUT
            return ModelResponse(parts=[TextPart(_CHILD_OUTPUT)])
        return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="lookup")])

    async def parent(messages, _info):
        result = messages[-1].parts[-1]
        if not isinstance(result, ToolReturnPart):
            return ModelResponse(parts=[ToolCallPart("confirm_fixture", {}, tool_call_id="approval")])
        if result.tool_call_id == "approval":
            assert result.content == "confirmed"
            return ModelResponse(parts=[ToolCallPart(
                "delegate_task", {"subagent_id": "reviewer", "task": _CHILD_TASK}, tool_call_id=_DELEGATE_CALL,
            )])
        assert result.tool_call_id == _DELEGATE_CALL and result.outcome == "success"
        assert result.content["output"] == {"text": _CHILD_OUTPUT}
        assert result.content["execution_id"] == child_ids[0]
        return ModelResponse(parts=[TextPart(_ANSWER)])

    name = "canonical-subagent-after-approval"
    features = RuntimeFeatures(name)
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.tool(confirm_fixture, effect_policy="none")
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="parent", allow_subagents=("reviewer",),
                   allow_tools=("confirm_fixture",), allow_skills=())
    features.agent("reviewer", model="child", allow_subagents=(), allow_tools=("lookup",), allow_skills=())
    async with (
        Persistence.open(name, storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session(_SESSION, principal=principal)
        turn = await runtime.sessions.start(
            _SESSION, "Confirm fixture A first", agent="default", principal=principal, idempotency_key="original",
        )
        assert (await turn.wait(timeout_seconds=10)).status is ExecutionStatus.WAITING_DEFERRED
        paused = await state.session_turns.get_turn(turn.key)
        assert paused is not None and paused["state"] == "waiting" and paused["generation"] == 1
        pending, = await runtime.approvals.list(turn.key, principal=principal)
        assert pending.status is ApprovalStatus.PENDING
        assert calls == ["approval requested"] and child_ids == []

        await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
            principal, pending.approval_id, "approve-original", ApprovalDecision.APPROVE,
        ))
        final = await turn.wait_final(timeout_seconds=10)
        assert final.key == turn.key and final.status is ExecutionStatus.SUCCEEDED
        assert final.view["output"] == {"text": _ANSWER}
        row = await state.session_turns.get_turn(turn.key)
        assert row is not None and row["generation"] == paused["generation"] + 1
        records = await state.checkpoint.tools.list_by_agent_run(
            turn.key.turn_id, execution_id=turn.key.turn_id, tenant_id=runtime.tenant_id,
        )
        assert {record.tool_call_id: record.status for record in records} == {
            "approval": ToolOperationStatus.COMPLETED, _DELEGATE_CALL: ToolOperationStatus.COMPLETED,
        }
        [child_id] = child_ids
        assert child_id != turn.key.turn_id
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=runtime.tenant_id) is None
        assert await state.conversation.sessions.get(_SESSION, tenant_id=runtime.tenant_id) is None
        replay = await runtime.sessions.start(
            _SESSION, "Confirm fixture A first", agent="default", principal=principal, idempotency_key="original",
        )
        assert replay.key == turn.key
        assert (await replay.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert calls == ["approval requested", "approved parent effect", "child tool"]
        assert child_ids == [child_id]
