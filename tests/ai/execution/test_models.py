#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/execution/test_models.py"""

from datetime import datetime, timezone

from linktools.ai.execution.session import MessageRole, SessionMessage


def test_message_role_values():
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"
    assert MessageRole.SYSTEM == "system"


def test_session_message_construction():
    now = datetime.now(timezone.utc)
    message = SessionMessage(
        id="msg-1",
        session_id="session-1",
        sequence=1,
        role=MessageRole.USER,
        content="hello",
        run_id=None,
        created_at=now,
    )
    assert message.content == "hello"
    assert message.role == MessageRole.USER
