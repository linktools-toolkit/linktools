#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Session domain models -- SessionRecord/SessionMessage carry no Store
reference, no physical root path, and no I/O methods (contrast with the
pre-vNext FileSession/RemoteSession in this same package's types.py, which
this models.py does not touch or depend on)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


MessageContent = "str | Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    parent_id: "str | None"
    status: SessionStatus
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


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
    """Input shape for :meth:`SessionStore.append_messages`.
    Deliberately carries no ``id``/``sequence``/``created_at`` -- the
    SessionStore is the SOLE authority for assigning those (mirroring how
    EventStore owns sequence assignment for events), so two concurrent
    callers appending to the same session can never compute the same
    sequence number themselves. The caller supplies only the semantic
    content; the store returns the persisted :class:`SessionMessage` with
    the fields it assigned."""

    role: MessageRole
    content: "str | Mapping[str, Any]"
    run_id: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
