#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Session domain models -- SessionRecord/SessionTurn/SessionMessage carry
no Store reference, no physical root path, and no I/O methods."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from ..json import JsonValue
from .domain import RunStatus


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


MessageContent = "str | Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: "str | None"
    tenant_id: "str | None"
    next_turn_sequence: int
    latest_completed_run_id: "str | None"
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SessionTurn:
    session_id: str
    sequence: int
    run_id: str
    input: JsonValue
    assistant_summary: "JsonValue | None"
    status: RunStatus
    created_at: datetime
    completed_at: "datetime | None"


@dataclass(frozen=True, slots=True)
class SessionMessage:
    id: str
    session_id: str
    sequence: int
    role: MessageRole
    content: "str | Mapping[str, Any]"
    run_id: "str | None"
    created_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewSessionMessage:
    """Input shape for appending a message to a session's history via the
    commit pipeline. Deliberately carries no ``id``/``sequence``/``created_at``
    -- the session store is the SOLE authority for assigning those (mirroring
    how the event store owns sequence assignment for events), so two concurrent
    callers appending to the same session can never compute the same
    sequence number themselves. The caller supplies only the semantic
    content; the store returns the persisted :class:`SessionMessage` with
    the fields it assigned."""

    role: MessageRole
    content: "str | Mapping[str, Any]"
    run_id: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


__all__: "list[str]" = [
    "MessageRole",
    "NewSessionMessage",
    "SessionMessage",
    "SessionRecord",
    "SessionTurn",
]
