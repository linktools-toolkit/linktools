#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The default ai-web Agent delegates through a native Step-backed turn."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from traceback import extract_tb

import httpx
import pytest
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import AgentContext, SessionCapability, TraceCapability
from linktools.ai.core import (
    ApprovalDecision, ApprovalStatus, ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.runs import ModelInteractionStateStepArchive
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import configure_execution
from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed


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
@pytest.mark.parametrize(
    "cancel,trace,recover,wrapped_cancel,ordinary,bare_cancel",
    ((False, False, False, False, False, False), (True, False, False, False, False, False),
     (True, False, False, True, False, False), (False, True, False, False, False, False),
     (False, True, True, False, False, False), (True, False, False, True, True, False),
     (True, False, False, True, True, True), (True, False, False, True, False, True)),
    ids=("success", "cancel", "cancel-wrapped", "cli-default-trace", "recover-without-trace",
         "ordinary-unknown", "ordinary-unknown-bare-stop", "delegate-unknown-bare-stop"),
)
async def test_ai_web_native_delegation_replays_exact_turn_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool, trace: bool, recover: bool, wrapped_cancel: bool,
    ordinary: bool, bare_cancel: bool,
) -> None:
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    if wrapped_cancel:
        from linktools.ai.runtime._tool import RuntimeToolOperationBridge

        unknown = RuntimeToolOperationBridge.unknown

        async def wrapped_unknown(bridge, decision, error):
            await unknown(bridge, decision, error)
            if bridge._canonical_event_writer is not None and isinstance(error, asyncio.CancelledError):
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error

        monkeypatch.setattr(RuntimeToolOperationBridge, "unknown", wrapped_unknown)

    if bare_cancel:
        from linktools.ai.runtime._agent_executor import AgentExecutor

        invoke = AgentExecutor.execute_regenerable

        async def drain_without_wrapped_error(executor, **kwargs):
            # Simulate outer cancellation discarding the error without
            # changing the already-retained receipt of the business tool.
            try:
                return await invoke(executor, **kwargs)
            except (AIError, asyncio.CancelledError) as error:
                if isinstance(error, AIError) and error.code is not ErrorCode.TOOL_EFFECT_UNKNOWN:
                    raise
                if not ordinary:
                    # The parent must not be the only task responsible for
                    # stopping its child while the provider is still draining.
                    await tool_cancelled.wait()
                raise asyncio.CancelledError from None

        monkeypatch.setattr(AgentExecutor, "execute_regenerable", drain_without_wrapped_error)

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

    async def ordinary_effect(_context: AgentContext[None]) -> str:
        calls.append("ordinary effect")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return "ordinary result"

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
        if ordinary:
            return ModelResponse(parts=[ToolCallPart("ordinary_effect", {}, tool_call_id="ordinary-call")])
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "reviewer", "task": _CHILD_TASK}, tool_call_id=_DELEGATE_CALL,
        )])

    name = "canonical-web-subagent"
    features = RuntimeFeatures(name, defaults={"execution": {"retain_history": True}})
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    configure_execution(features, retain_history=True)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.tool(lookup, effect_policy="none")
    if ordinary:
        features.tool(ordinary_effect, effect_policy="non_replay_safe")
    features.agent("default", model="parent", allow_subagents=("reviewer",),
                   allow_tools=("ordinary_effect",) if ordinary else (), allow_skills=())
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
        if recover:
            async def lose_terminal_commit(_key, *, generation, state, **_options):
                assert generation == 1 and state == "succeeded"
                return False

            # The worker completed native Step, but terminal Session publication
            # was never committed, so optional parent Trace never enqueues.
            monkeypatch.setattr(state.session_turns, "finish", lose_terminal_commit)
        created = await http.post("/api/sessions", headers=_HEADERS,
                                  json={"session_id": _SESSION, "request_id": "create"})
        assert created.status_code == 201, created.text
        async with _events(app) as frames:
            stage = "subscribe"
            try:
                async with asyncio.timeout(10):
                    snapshot = await frames.get()
                    assert snapshot["type"] == "snapshot" and snapshot["session"]["active_turn_id"] is None
                    stage = "admit turn"
                    turn_id = await request(http)
                    stage = "enter tool"
                    await entered.wait()
                    assert await request(http) == turn_id
                    if cancel:
                        stage = "Stop request"
                        stopped = await http.post("/api/session/cancel", params=params, headers=_HEADERS,
                                                  json={"request_id": "stop", "turn_id": turn_id})
                        assert stopped.status_code == 200 and stopped.json()["accepted"], stopped.text
                        stage = "cancel tool"
                        await tool_cancelled.wait()
                        assert not release.is_set()
                    else:
                        release.set()
                    stage = "terminal event"
                    tool_statuses = []
                    while True:
                        frame = await frames.get()
                        if frame.get("turn_id") != turn_id:
                            continue
                        if frame["type"] == "tool" and frame["item"]["key"] == f"tool:{_DELEGATE_CALL}":
                            tool_statuses.append(frame["item"]["status"])
                        if ordinary and frame["type"] == "gap" and frame.get("phase") == "producer_stopped":
                            break
                        if frame["type"] == "gap" and recover:
                            break
                        assert frame.get("phase") != "producer_stopped", frame
                        if frame["type"] == "terminal":
                            assert not recover and not ordinary, frame
                            assert frame["state"] == ("cancelled" if cancel else "succeeded"), frame
                            if not cancel:
                                assert_answer(frame)
                            break
                    if cancel and not ordinary:
                        assert tool_statuses and tool_statuses[0] == "RUNNING" and "SUCCEEDED" not in tool_statuses
                    elif not ordinary:
                        assert tool_statuses == ["RUNNING", "SUCCEEDED"]
            except TimeoutError:
                pytest.fail(f"native delegation timed out at {stage}", pytrace=False)
            finally:
                release.set()
        if ordinary:
            assert calls == ["parent model", "ordinary effect"]
            [operation] = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
            assert (operation.tool_call_id, operation.status) == ("ordinary-call", ToolOperationStatus.EFFECT_UNKNOWN)
            assert not await state.execution.executions.list_children(turn_id, tenant_id=runtime.tenant_id)
            assert (await state.session_turns.get_turn(SessionTurnKey(_SESSION, turn_id)))["state"] == "cancelling"
            return
        assert calls == expected_calls
        [child_id] = child_ids
        assert child_id != turn_id and returned_ids == ([] if cancel else [child_id])
        if cancel:
            stopped_child = await state.execution.executions.get(child_id, tenant_id=runtime.tenant_id)
            assert stopped_child.status.value in {"CANCELLED", "FAILED"}
        if wrapped_cancel:
            [original] = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
            assert original.tool_call_id == _DELEGATE_CALL and original.status.value == "EFFECT_UNKNOWN"
            # It is an intrinsic native delegate, not an externally registered
            # capability. Stop relies on the original claimed child proof.
            assert original.capability_ref is None

        assert await state.execution.executions.get(turn_id, tenant_id=runtime.tenant_id) is None
        assert state.conversation.sessions is None

        if trace and not recover:
            principal = runtime.default_principal
            parent_trace = await _wait_for_committed(
                lambda: runtime.history.turn_trace(_SESSION, turn_id, principal=principal),
                lambda page: bool(page.items),
            )
            assert all(item.turn_id == turn_id for item in parent_trace.items)
            parent_tools = {item.payload.get("tool_call_id") for item in parent_trace.items
                            if item.payload["kind"] == "TOOL_RESULT"}
            assert parent_tools == {_DELEGATE_CALL}

    async with open_host() as (state, runtime, _, http):
        if recover:
            key = SessionTurnKey(_SESSION, turn_id)
            turns = state.session_turns
            assert (await turns.get_turn(key))["state"] == "accepted"
            # Simulate expiration rather than sleep for the full lease period.
            async with turns._context.engine.begin() as connection:
                changed = await connection.execute(
                    turns._turns.update().filter_by(**turns._turn_values(key), generation=1).values(lease_until=0),
                )
                assert changed.rowcount == 1

            async def trace_forbidden(_archive, **_kwargs):
                pytest.fail("Recover does not read optional parent Trace")

            with monkeypatch.context() as unavailable:
                unavailable.setattr(ModelInteractionStateStepArchive, "list_model_interactions", trace_forbidden)
                unavailable.setattr(ModelInteractionStateStepArchive, "list_trace_events", trace_forbidden)
                unavailable.setattr(ModelInteractionStateStepArchive, "list_turn_trace_events", trace_forbidden)
                assert await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=runtime.default_principal)

            row = await turns.get_turn(key)
            assert row["state"] == "succeeded" and row["generation"] == 2
            assert not (await runtime.history.turn_trace(_SESSION, turn_id, principal=runtime.default_principal)).items
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
            child_trace = await runtime.history.trace(child_id, principal=runtime.default_principal)
            assert child_trace.items and all(item.execution_id == child_id for item in child_trace.items)
            child_tools = {item.payload.get("tool_call_id") for item in child_trace.items
                           if item.payload["kind"] == "TOOL_RESULT"}
            assert child_tools == {"lookup-A"}



@pytest.mark.asyncio
@pytest.mark.parametrize("effect_policy", ("none", "replay_safe"))
async def test_approved_parent_tool_continues_to_native_subagent(tmp_path: Path, effect_policy: str) -> None:
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
    features = RuntimeFeatures(name, defaults={"execution": {"retain_history": True}})
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    configure_execution(features, retain_history=True)
    features.capability(SessionCapability(checkpoints=True))
    features.tool(confirm_fixture, effect_policy=effect_policy)
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="parent", allow_subagents=("reviewer",),
                   allow_tools=("confirm_fixture",), allow_skills=())
    features.agent("reviewer", model="child", allow_subagents=(), allow_tools=("lookup",), allow_skills=())
    async with (
        Persistence.open(name, storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        principal = service_principal(runtime.tenant_id, "approval-fixture")
        await runtime.sessions.create_turn_session(_SESSION, principal=principal)
        turn = await runtime.sessions.start_turn(
            _SESSION, "Confirm fixture A first", agent="default", principal=principal, idempotency_key="original",
        )
        paused_result = await asyncio.wait_for(turn.wait(), timeout=10)
        assert paused_result.status is ExecutionStatus.WAITING_DEFERRED
        paused = await state.session_turns.get_turn(turn.key)
        assert paused is not None and paused["state"] == "waiting" and paused["generation"] == 1
        pending, = await runtime.approvals.list(turn.key, principal=principal)
        assert pending.status is ApprovalStatus.PENDING
        assert calls == ["approval requested"] and child_ids == []
        [original] = await state.checkpoint.tools.list_by_agent_run(
            turn.key.turn_id, execution_id=turn.key.turn_id, tenant_id=runtime.tenant_id,
        )
        assert original.tool_call_id == "approval" and original.status is ToolOperationStatus.PENDING

        phase = "approval decision and resume"
        try:
            await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
                principal, pending.approval_id, "approve-original", ApprovalDecision.APPROVE,
            ))
            phase = "wait for terminal"
            final = await turn.wait_final(timeout_seconds=10)
        except AIError as error:
            cause = error.__cause__ or error
            frame = extract_tb(cause.__traceback__)[-1]
            raise AssertionError(
                f"{phase}: {error.code.value} at {Path(frame.filename).name}:{frame.lineno} ({frame.name}); "
                f"calls={calls!r}, children={len(child_ids)}"
            ) from error
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
        approved = next(record for record in records if record.tool_call_id == "approval")
        assert approved.tool_operation_id == original.tool_operation_id and approved.fence == original.fence + 1
        assert approved.producer_generation == row["generation"]
        [child_id] = child_ids
        assert child_id != turn.key.turn_id
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=runtime.tenant_id) is None
        assert state.conversation.sessions is None
        replay = await runtime.sessions.start_turn(
            _SESSION, "Confirm fixture A first", agent="default", principal=principal, idempotency_key="original",
        )
        assert replay.key == turn.key
        assert (await replay.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert calls == ["approval requested", "approved parent effect", "child tool"]
        assert child_ids == [child_id]
