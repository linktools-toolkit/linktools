#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/session/test_recorder.py"""

from linktools.ai.session.models import MessageRole
from linktools.ai.session.recorder import SessionRecorder


def test_build_turn_messages_includes_user_and_assistant():
    recorder = SessionRecorder()
    messages = recorder.build_turn_messages(
        user_prompt="hello", output="hi there", run_id="run-1"
    )
    assert [m.role for m in messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert messages[0].content == "hello"
    assert messages[1].content == "hi there"
    assert all(m.run_id == "run-1" for m in messages)


def test_build_turn_messages_omits_user_when_prompt_empty():
    """A resume continuation has no new user turn -- only the assistant
    output is recorded, matching the pre-extraction behavior."""
    recorder = SessionRecorder()
    messages = recorder.build_turn_messages(user_prompt="", output="continued", run_id="run-2")
    assert [m.role for m in messages] == [MessageRole.ASSISTANT]
    assert messages[0].content == "continued"


def test_build_turn_messages_stringifies_non_str_output():
    recorder = SessionRecorder()
    messages = recorder.build_turn_messages(user_prompt="q", output={"a": 1}, run_id="run-3")
    assistant = [m for m in messages if m.role == MessageRole.ASSISTANT][0]
    assert assistant.content == str({"a": 1})
