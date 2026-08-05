#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure session record and turn models."""


from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Mapping

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


class SessionState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class SessionWorkspace:
    cwd: str
    additional_directories: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.cwd:
            raise ValueError("session workspace cwd is required")
        object.__setattr__(self, "cwd", str(Path(self.cwd).expanduser().resolve()))
        object.__setattr__(
            self,
            "additional_directories",
            tuple(str(Path(item).expanduser().resolve()) for item in self.additional_directories),
        )


@dataclass(frozen=True, slots=True)
class SessionSettings:
    agent_id: str
    options: "Mapping[str, JsonValue]" = field(default_factory=dict)
    tool_source_fingerprints: "tuple[str, ...]" = ()


@dataclass(frozen=True, slots=True)
class CreateSession:
    session_id: str
    user_id: "str | None"
    tenant_id: "str | None"
    workspace: SessionWorkspace
    settings: SessionSettings
    context_seed: "SessionContextSeed | None" = None


@dataclass(frozen=True, slots=True)
class UpdateSession:
    session_id: str
    expected_revision: int
    workspace: "SessionWorkspace | None" = None
    settings: "SessionSettings | None" = None
    title: "str | None" = None
    state: "SessionState | None" = None


@dataclass(frozen=True, slots=True)
class ForkSession:
    source_session_id: str
    target_session_id: str
    user_id: "str | None"
    tenant_id: "str | None"
    workspace: SessionWorkspace
    settings: SessionSettings


@dataclass(frozen=True, slots=True)
class SessionQuery:
    user_id: "str | None" = None
    tenant_id: "str | None" = None

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
    workspace: SessionWorkspace = field(
        default_factory=lambda: SessionWorkspace(cwd=".")
    )
    settings: SessionSettings = field(
        default_factory=lambda: SessionSettings(agent_id="default")
    )
    title: "str | None" = None
    state: SessionState = SessionState.OPEN
    revision: int = 1

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("session revision must start at one")


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
    "CreateSession",
    "ForkSession",
    "SessionQuery",
    "SessionRecord",
    "SessionSettings",
    "SessionState",
    "SessionTurn",
    "SessionWorkspace",
    "UpdateSession",
    "SeedTurn",
    "SessionContextSeed",
]
