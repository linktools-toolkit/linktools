#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext, SessionCapability, TraceCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.session import CreateSessionRequest
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.persistence import PersistenceStorage, StateTurnResources
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import SessionTurnNotice

from ._runtime_test_helpers import _Models, _UsageFunctionModel


class _HostAuthorization(TenantAuthorizationPolicy):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("file", "memory", "borrowed", "coordinated"))
async def test_observe_turns_follows_owned_boundaries(tmp_path: Path, backend: str) -> None:
    async def lookup(context: AgentContext[None], value: str) -> str:
        return "not-for-notices"

    def respond(messages, info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("answer")])
        return ModelResponse(parts=[ToolCallPart("lookup", {"value": "not-for-notices"}, tool_call_id="call")])

    features = RuntimeFeatures("turn-notices", defaults={"execution": {"retain_history": True}})
    features.model("default", _Models(_UsageFunctionModel(respond)))
    if backend != "memory":
        features.capability(SessionCapability(checkpoints=True))
    features.capability(TraceCapability())
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="default", allow_tools=("lookup",))
    async with AsyncExitStack() as resources:
        if backend == "file":
            storage = PersistenceStorage.from_root(tmp_path)
        elif backend == "memory":
            storage = PersistenceStorage.in_memory()
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'borrowed.db'}")
            resources.push_async_callback(engine.dispose)
            await provision_runtime_database(engine)
            coordination = StateTurnResources() if backend == "coordinated" else None
            storage = PersistenceStorage.sql(engine, coordination=coordination)
        runtime = await resources.enter_async_context(Runtime.open(
            "default", storage=storage, features=features,
            authorization=None if backend == "memory" else _HostAuthorization(),
        ))
        principal = runtime.default_principal
        if backend != "memory":
            sessions = runtime.sessions
            await sessions.create_turn_session("session", principal=principal)
            lifecycle, frames = [], []

            def fail(_notice):
                raise RuntimeError("callback must not break the turn")

            unsubscribe_failed = sessions.observe_turn_lifecycle(fail)
            unsubscribe_lifecycle = sessions.observe_turn_lifecycle(lifecycle.append)
            unsubscribe_frames = sessions.observe_turn_session("session", frames.append)
            try:
                turn = await sessions.start_turn("session", "ask", agent="default", principal=principal, idempotency_key="turn")
                result = await turn.wait_final(timeout_seconds=10)
                assert result.view["output"] == {"text": "answer"}
                replay = await sessions.start_turn("session", "ask", agent="default", principal=principal, idempotency_key="turn")
                assert replay.key == turn.key
                assert [item["phase"] for item in lifecycle] == ["admitted", "terminal"]
                assert {item["turn_id"] for item in lifecycle} == {turn.key.turn_id}
                milestones = [item for item in frames if item["type"] in {"admitted", "tool", "terminal"}]
                assert [item["type"] for item in milestones] == ["admitted", "tool", "tool", "terminal"]
                assert {item["turn_id"] for item in milestones} == {turn.key.turn_id}
                assert [(frame["item"]["key"], frame["item"]["tool_name"], frame["item"]["status"])
                        for frame in milestones if frame["type"] == "tool"] == [
                    ("tool:call", "lookup", "RUNNING"), ("tool:call", "lookup", "SUCCEEDED"),
                ]
                assert "not-for-notices" not in repr(milestones)
                assert all(set(notice) == {"kind", "phase", "session_id", "turn_id", "revision"}
                           for notice in lifecycle)
            finally:
                unsubscribe_frames()
                unsubscribe_lifecycle()
                unsubscribe_failed()
            return
        await runtime.sessions.create("default", CreateSessionRequest(principal, "session", "create"))

        def fail(_notice):
            raise RuntimeError("callback must not break the turn")

        changes = asyncio.Event()
        remove_invalidation = runtime.sessions.observe_changes("session", changes.set)
        runtime.sessions.invalidate("session")
        assert changes.is_set()
        remove_invalidation()
        unsubscribe_failed = runtime.sessions.observe_turns(fail)
        notices: list[SessionTurnNotice] = []
        unsubscribe = runtime.sessions.observe_turns(handler=notices.append)
        turn = await runtime.executions.start_in_session("session", "ask", principal=principal, idempotency_key="turn")
        assert (await turn.wait(timeout_seconds=10)).result.output == {"text": "answer"}
        replay = await runtime.executions.start_in_session("session", "ask", principal=principal, idempotency_key="turn")
        assert replay.execution_id == turn.execution_id
        assert [notice.kind for notice in notices] == ["admitted", "tool", "tool", "terminal"]
        assert {(notice.session_id, notice.execution_id) for notice in notices} == {("session", turn.execution_id)}
        assert [(notice.process.key, notice.process.tool_name, notice.process.status)
                for notice in notices if notice.process is not None] == [
                    ("tool:call", "lookup", "RUNNING"),
                    ("tool:call", "lookup", "SUCCEEDED"),
                ]
        assert "not-for-notices" not in repr(notices)
        unsubscribe()
        unsubscribe_failed()


@pytest.mark.asyncio
async def test_explicit_same_id_recovery_notifies_terminal_once_without_new_admission(tmp_path: Path) -> None:
    from linktools.ai.workspace import Workspace
    from linktools.commands.ai._common import _open_local_runtime

    workspace = Workspace.initialize(tmp_path)
    entered = asyncio.Event()

    async def hold(_messages, _info):
        entered.set()
        await asyncio.Event().wait()

    async def answer(_messages, _info):
        return ModelResponse(parts=[TextPart("recovered answer")])

    from .test_session_tool_process_recovery import _expire

    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(hold))) as initial:
        runtime = initial.runtime
        actor = runtime.default_principal
        sessions = runtime.sessions
        await sessions.create_turn_session("session", principal=actor)
        turn = await sessions.start_turn("session", "ask", agent="default", principal=actor, idempotency_key="turn")
        await asyncio.wait_for(entered.wait(), 10)
        rejected_notices = []
        unsubscribe = sessions.observe_turn_lifecycle(rejected_notices.append)
        try:
            assert not await sessions.recover_turn("session", turn.key.turn_id, mode="resume", principal=actor)
            await asyncio.wait_for(runtime.handoff(), 10)
            assert not rejected_notices
        finally:
            unsubscribe()

    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(answer))) as successor:
        runtime = successor.runtime
        actor = runtime.default_principal
        sessions = runtime.sessions
        assert (await runtime.history.turn_session_state("session", principal=actor))["active_turn_id"] == turn.key.turn_id
        await _expire(runtime._state.session_turns, SessionTurnKey("session", turn.key.turn_id), generation=1)
        notices, frames = [], []
        unsubscribe = sessions.observe_turn_lifecycle(notices.append)
        unsubscribe_frames = sessions.observe_turn_session("session", frames.append)
        try:
            assert await sessions.recover_turn("session", turn.key.turn_id, mode="resume", principal=actor)
            replay = await sessions.start_turn("session", "ask", agent="default", principal=actor, idempotency_key="turn")
            assert replay.key == turn.key
            result = await replay.wait_final(timeout_seconds=10)
            assert result.view["output"] == {"text": "recovered answer"}
            assert not await sessions.recover_turn("session", turn.key.turn_id, mode="resume", principal=actor)
            current = await runtime.history.turn_session_state("session", principal=actor)
            assert current["active_turn_id"] is None and current["last_turn_no"] == 1
            saved = await runtime.history.turn_result("session", turn.key.turn_id, principal=actor)
            assert saved["conversation_committed"] and saved["turn_id"] == turn.key.turn_id
            assert [(notice["phase"], notice["session_id"], notice["turn_id"]) for notice in notices] == [
                ("admitted", "session", turn.key.turn_id),
                ("terminal", "session", turn.key.turn_id),
            ]
            assert [(frame["type"], frame["turn_id"], frame["turn_no"], frame["generation"])
                    for frame in frames if frame["type"] in {"admitted", "terminal"}] == [
                ("admitted", turn.key.turn_id, 1, 2),
                ("terminal", turn.key.turn_id, 1, 2),
            ]
        finally:
            unsubscribe_frames()
            unsubscribe()
