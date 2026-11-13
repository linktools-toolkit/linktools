#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session model context is selected before old message bodies are loaded."""

from contextvars import ContextVar

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.capability import ExecutionCapability
from linktools.ai.persistence.runs import ContentArchive
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._conversation_persistence import ConversationPersistence
from linktools.ai.runtime import Runtime

from .test_runtime_session_start_preparation import _features as _base_features


def _features(model):
    features = _base_features(model)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    return features

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("selection,expected_messages", (("all", 3), ("current_turn", 1)))
async def test_session_selects_context_before_prior_body_read_and_preserves_timeline(
    tmp_path, monkeypatch, selection, expected_messages,
):
    seen = []
    calls = []
    sql = []
    bodies = []
    active = ContextVar("conversation_context_read", default=False)

    async def model(messages, info):
        seen.append(tuple(messages))
        return ModelResponse(parts=[TextPart("answer")])

    original_read = ContentArchive._read_content

    async def read_content(self, content):
        result = await original_read(self, content)
        if active.get():
            bodies.append(len(result))
        return result

    monkeypatch.setattr(ContentArchive, "_read_content", read_content)
    original_native_capture = SqlStepStore.capture_message_roots

    async def capture_native(self, roots, **kwargs):
        load = await original_native_capture(self, roots, **kwargs)

        async def read_native():
            result = await load()
            if active.get():
                bodies.extend(len(body) for group in result for body in group)
            return result

        return read_native

    monkeypatch.setattr(SqlStepStore, "capture_message_roots", capture_native)
    original = ConversationPersistence.load_committed_conversation_context

    async def load(self, **kwargs):
        token = active.set(True)
        try:
            result = await original(self, **kwargs)
            calls.append((kwargs.get("include_content", True), len(result.messages)))
            return result
        finally:
            active.reset(token)

    monkeypatch.setattr(ConversationPersistence, "load_committed_conversation_context", load)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("session-start-preparation", storage=storage) as state:
        engine = state.conversation.sessions.state_store.storage_group.context.engine.sync_engine

        def capture(_connection, _cursor, statement, *_args):
            if active.get() and statement.lstrip().upper().startswith(("SELECT", "WITH")):
                sql.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            async with Runtime.attach(persistence=state, features=_features(model), metrics=None) as runtime:
                session = runtime.agents.get().session("session")
                first = await session.start("old question", idempotency_key="old")
                assert (await first.wait(timeout_seconds=5)).result.output == {"text": "answer"}
                calls.clear()
                sql.clear()
                bodies.clear()
                second = await session.start("new question", session_context=selection, idempotency_key="new")
                assert (await second.wait(timeout_seconds=5)).result.output == {"text": "answer"}
                assert len(seen[-1]) == expected_messages
                assert calls == [(selection == "all", expected_messages - 1)]
                if selection == "current_turn":
                    assert bodies == []
                else:
                    assert bodies
                print("session context source reads", selection, {"selects": len(sql), "bodies": len(bodies), "bytes": sum(bodies)})
                timeline = await session.timeline()
                assert len(timeline.items) == 2
                assert {turn.execution_id for turn in timeline.items} == {first.execution_id, second.execution_id}
                assert {turn.user_input["prompt"]["text"] for turn in timeline.items} == {"old question", "new question"}
                again = await session.start("new question", session_context=selection, idempotency_key="new")
                assert again.execution_id == second.execution_id
                with pytest.raises(AIError) as error:
                    await session.start("new question", session_context="current_turn" if selection == "all" else "all", idempotency_key="new")
                assert error.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
        finally:
            event.remove(engine, "before_cursor_execute", capture)
    # Same-version reopen keeps both authoritative turns and the admitted policy.
    async with Persistence.open("session-start-preparation", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=_features(model), metrics=None) as runtime:
            record = await state.execution.executions.get(second.execution_id, tenant_id=runtime.default_principal.tenant_id)
            assert record.session_context == selection
            timeline = await runtime.agents.get().session("session").timeline()
            assert len(timeline.items) == 2


async def test_current_turn_requires_a_session_and_rejects_unknown_selection(tmp_path):
    async def model(messages, info):
        raise AssertionError("invalid context selection must fail before the model")

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(model), metrics=None) as runtime:
            for kwargs in ({"session_context": "current_turn"}, {"session_id": "session", "session_context": "unknown"}):
                with pytest.raises(AIError) as error:
                    await runtime.agents.get().start("question", **kwargs)
                assert error.value.code == ErrorCode.REQUEST_FIELD_INVALID
