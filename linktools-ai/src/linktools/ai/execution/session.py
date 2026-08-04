#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure session record and turn models."""


from dataclasses import dataclass
from typing import Literal

from typing import TYPE_CHECKING

from .domain import MessageCaptureState

if TYPE_CHECKING:
    from datetime import datetime
    from ..json import JsonValue
    from .domain import RunStatus


@dataclass(frozen=True, slots=True)
class SeedTurn:
    session_id: str
    sequence: int
    run_id: str
    input: "JsonValue"
    delta_messages: "tuple[JsonValue, ...]"
    status: "RunStatus"
    capture_state: MessageCaptureState


@dataclass(frozen=True, slots=True)
class SessionContextSeed:
    schema: Literal["session-context-seed.v1"]
    source_session_id: str
    source_updated_at: "datetime"
    turns: "tuple[SeedTurn, ...]"

    def __post_init__(self) -> None:
        if self.schema != "session-context-seed.v1":
            raise ValueError("unsupported session context seed schema")

@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: "str | None"
    tenant_id: "str | None"
    next_turn_sequence: int
    latest_completed_run_id: "str | None"
    created_at: "datetime"
    updated_at: "datetime"
    context_seed: "SessionContextSeed | None" = None


@dataclass(frozen=True, slots=True)
class SessionTurn:
    session_id: str
    sequence: int
    run_id: str
    input: "JsonValue"
    # TURN_DELTA: this turn's net-new messages (new_messages()), window-policy
    # immune. Accumulated across PAUSED/resume of the same turn. Empty for a
    # turn that produced no model messages (e.g. PENDING->CANCELLED).
    delta_messages: "tuple[JsonValue, ...]"
    status: "RunStatus"
    # Honesty marker for the audit read path: COMPLETE means the delta is a
    # faithful record of what the run produced; PARTIAL means it was salvaged
    # from an interrupted run (streaming/tool/cancel-before-commit) and may be
    # incomplete; UNAVAILABLE means no trustworthy delta exists.
    capture_state: MessageCaptureState
    created_at: "datetime"
    completed_at: "datetime | None"


__all__: "list[str]" = [
    "SessionRecord",
    "SessionTurn",
    "SeedTurn",
    "SessionContextSeed",
]
