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
