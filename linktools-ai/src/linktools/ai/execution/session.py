#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure session record and turn models."""


from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from ..json import JsonValue
    from .domain import RunStatus

@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: "str | None"
    tenant_id: "str | None"
    next_turn_sequence: int
    latest_completed_run_id: "str | None"
    created_at: "datetime"
    updated_at: "datetime"


@dataclass(frozen=True, slots=True)
class SessionTurn:
    session_id: str
    sequence: int
    run_id: str
    input: "JsonValue"
    assistant_summary: "JsonValue | None"
    status: "RunStatus"
    created_at: "datetime"
    completed_at: "datetime | None"


__all__: "list[str]" = [
    "SessionRecord",
    "SessionTurn",
]
