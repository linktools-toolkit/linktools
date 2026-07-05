#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/session/test_models.py"""
from datetime import datetime, timezone

from linktools.ai.session.models import MessageRole, SessionMessage, SessionRecord, SessionStatus


def test_session_status_values():
    assert SessionStatus.ACTIVE == "active"
    assert SessionStatus.ARCHIVED == "archived"


def test_message_role_values():
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"
    assert MessageRole.SYSTEM == "system"


def test_session_record_construction_has_no_root_or_store():
    now = datetime.now(timezone.utc)
    record = SessionRecord(id="session-1", parent_id=None, status=SessionStatus.ACTIVE, version=1, created_at=now, updated_at=now)
    assert record.id == "session-1"
    assert not hasattr(record, "root")
    assert not hasattr(record, "copy")
    assert dict(record.metadata) == {}


def test_session_message_construction():
    now = datetime.now(timezone.utc)
    message = SessionMessage(
        id="msg-1", session_id="session-1", sequence=1, role=MessageRole.USER,
        content="hello", run_id=None, created_at=now,
    )
    assert message.content == "hello"
    assert message.role == MessageRole.USER


def test_session_record_is_frozen():
    import pytest
    now = datetime.now(timezone.utc)
    record = SessionRecord(id="session-1", parent_id=None, status=SessionStatus.ACTIVE, version=1, created_at=now, updated_at=now)
    with pytest.raises(Exception):
        record.status = SessionStatus.ARCHIVED
