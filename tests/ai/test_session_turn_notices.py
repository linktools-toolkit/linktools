import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CreateSessionRequest
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.persistence import PersistenceStorage, StateTurnResources
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import SessionTurnNotice

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("file", "memory", "borrowed", "coordinated"))
async def test_observe_turns_follows_owned_boundaries(tmp_path: Path, backend: str) -> None:
    async def lookup(context: AgentContext[None], value: str) -> str:
        return "not-for-notices"

    def respond(messages, info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("answer")])
        return ModelResponse(parts=[ToolCallPart("lookup", {"value": "not-for-notices"}, tool_call_id="call")])

    features = RuntimeFeatures("turn-notices")
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.capability(ExecutionCapability(retain_history=True))
    if backend != "memory":
        features.capability(StepPersistenceCapability())
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
        runtime = await resources.enter_async_context(Runtime.open("default", storage=storage, features=features))
        principal = runtime.default_principal
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
        turn = await runtime.sessions.start("session", "ask", principal=principal, idempotency_key="turn")
        assert (await turn.wait(timeout_seconds=10)).result.output == {"text": "answer"}
        replay = await runtime.sessions.start("session", "ask", principal=principal, idempotency_key="turn")
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

    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(hold))) as initial:
        actor = initial.runtime.default_principal
        await initial.runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        turn = await initial.runtime.sessions.start("session", "ask", principal=actor, idempotency_key="turn")
        await asyncio.wait_for(entered.wait(), 10)
        rejected_notices = []
        unsubscribe = initial.runtime.sessions.observe_turns(rejected_notices.append)
        try:
            with pytest.raises(AIError) as error:
                await initial.runtime.executions.recover(
                    turn.execution_id, principal=actor, idempotency_key="rejected",
                )
            assert error.value.code is ErrorCode.STORAGE_CONFLICT
            await asyncio.wait_for(initial.runtime.handoff(), 10)
            assert not rejected_notices
        finally:
            unsubscribe()

    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(answer))) as successor:
        runtime = successor.runtime
        actor = runtime.default_principal
        assert (await runtime.executions.inspect(turn.execution_id, principal=actor)).status.value == "STARTED"
        notices = []
        unsubscribe = runtime.sessions.observe_turns(notices.append)
        try:
            recovered = await runtime.executions.recover(turn.execution_id, principal=actor, idempotency_key="recover")
            assert recovered.execution_id == turn.execution_id
            result = await runtime.executions.wait(turn.execution_id, principal=actor, timeout_seconds=10)
            assert result.result.output == {"text": "recovered answer"}
            replay = await runtime.executions.recover(turn.execution_id, principal=actor, idempotency_key="recover")
            assert replay.execution_id == turn.execution_id
            detail = await runtime.history.session_detail("session", principal=actor)
            assert detail.session.active_execution_id is None
            saved, = detail.timeline.items
            assert saved.conversation_committed and saved.execution_id == turn.execution_id
            assert [(notice.kind, notice.session_id, notice.execution_id) for notice in notices] == [
                ("terminal", "session", turn.execution_id),
            ]
        finally:
            unsubscribe()
