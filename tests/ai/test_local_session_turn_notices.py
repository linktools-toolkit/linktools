import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.execution import CreateSessionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
async def test_file_sqlite_local_turn_notices_follow_owned_boundaries(tmp_path: Path) -> None:
    async def lookup(context: AgentContext[None], value: str) -> str:
        return "not-for-notices"

    def respond(messages, info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("answer")])
        return ModelResponse(parts=[ToolCallPart("lookup", {"value": "not-for-notices"}, tool_call_id="call")])

    features = RuntimeFeatures("local-notices")
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(TraceCapability())
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="default", allow_tools=("lookup",))
    async with Runtime.open("default", storage=PersistenceStorage.from_root(tmp_path), features=features) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create("default", CreateSessionRequest(principal, "session", "create"))

        def fail(_notice):
            raise RuntimeError("callback must not break the turn")

        changes = asyncio.Event()
        remove_invalidation = runtime.sessions.observe_changes("session", changes.set)
        runtime.sessions.invalidate("session")
        assert changes.is_set()
        remove_invalidation()
        unsubscribe_failed = runtime.sessions.observe_local_turns(fail)
        notices = []
        unsubscribe = runtime.sessions.observe_local_turns(notices.append)
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
