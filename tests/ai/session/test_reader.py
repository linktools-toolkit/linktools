#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/session/test_reader.py"""

import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from linktools.ai.session.models import MessageRole, SessionMessage
from linktools.ai.session.reader import SessionReader


class _FakeSessionStore:
    def __init__(self, messages: "tuple[SessionMessage, ...]") -> None:
        self._messages = messages

    async def list_messages(self, session_id: str, *, after_sequence: int = 0, limit: int = 1000):
        assert session_id == "session-1"
        return self._messages


def _message(sequence: int, role: MessageRole, content) -> SessionMessage:
    return SessionMessage(
        id=f"msg-{sequence}",
        session_id="session-1",
        sequence=sequence,
        role=role,
        content=content,
        run_id="run-1",
        created_at=datetime.now(timezone.utc),
    )


def test_load_message_history_converts_user_and_assistant():
    store = _FakeSessionStore(
        (
            _message(1, MessageRole.USER, "hello"),
            _message(2, MessageRole.ASSISTANT, "hi there"),
        )
    )
    reader = SessionReader(store)

    history = asyncio.run(reader.load_message_history("session-1"))

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert history[0].parts == [UserPromptPart(content="hello", timestamp=history[0].parts[0].timestamp)]
    assert isinstance(history[1], ModelResponse)
    assert history[1].parts == [TextPart(content="hi there")]


def test_load_message_history_converts_system_role():
    store = _FakeSessionStore((_message(1, MessageRole.SYSTEM, "be terse"),))
    reader = SessionReader(store)

    history = asyncio.run(reader.load_message_history("session-1"))

    assert len(history) == 1
    assert isinstance(history[0], ModelRequest)
    part = history[0].parts[0]
    assert isinstance(part, SystemPromptPart)
    assert part.content == "be terse"


def test_load_message_history_stringifies_non_str_content():
    store = _FakeSessionStore((_message(1, MessageRole.ASSISTANT, {"a": 1}),))
    reader = SessionReader(store)

    history = asyncio.run(reader.load_message_history("session-1"))

    assert len(history) == 1
    assert isinstance(history[0], ModelResponse)
    assert history[0].parts == [TextPart(content=str({"a": 1}))]


def test_load_message_history_empty_when_no_prior_turns():
    reader = SessionReader(_FakeSessionStore(()))

    history = asyncio.run(reader.load_message_history("session-1"))

    assert history == ()
